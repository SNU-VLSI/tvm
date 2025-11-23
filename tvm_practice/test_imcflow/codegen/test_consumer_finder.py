import tvm
from tvm import relay

from tvm.relay.backend.contrib.imcflow.transform import UseDefChainParser, ConsumerFinder


def _build_simple_chain():
  x = relay.var("x")
  relu_call = relay.nn.relu(x)
  add_call = relay.add(relu_call, relay.const(1))
  func = relay.Function([x], add_call)
  return func, x, relu_call, add_call


def test_direct_consumer():
  func, x, relu_call, _ = _build_simple_chain()
  parser = UseDefChainParser()
  parser.visit(func.body)
  finder = ConsumerFinder(parser)

  consumers = finder.find_consumers_of_node(x)
  assert len(consumers) == 1
  consumer_call, arg_idx = consumers[0]
  assert consumer_call == relu_call
  assert arg_idx == 0


def test_skip_predicate_reaches_downstream():
  func, x, relu_call, add_call = _build_simple_chain()
  parser = UseDefChainParser()
  parser.visit(func.body)

  def skip_relu(call, _arg_idx):
    return isinstance(call.op, tvm.ir.Op) and call.op == relay.op.get("nn.relu")

  finder = ConsumerFinder(parser, skip_predicates=[skip_relu])
  consumers = finder.find_consumers_of_node(x)
  # relu is skipped, so the downstream add should be the recorded consumer
  assert len(consumers) == 1
  consumer_call, arg_idx = consumers[0]
  assert consumer_call == add_call
  assert arg_idx == 0
  # ensure relu itself was not recorded as a consumer
  assert consumer_call != relu_call


def test_recurses_into_function_when_skipped():
  # Build a small callee function
  a = relay.var("a")
  b = relay.var("b")
  inner_add = relay.add(a, b)
  inner_fn = relay.Function([a, b], inner_add)

  # Top-level function calling the inner function
  x = relay.var("x")
  y = relay.var("y")
  call_inner = relay.Call(inner_fn, [x, y])
  top_func = relay.Function([x, y], call_inner)

  parser = UseDefChainParser()
  parser.visit(top_func.body)

  def skip_function_call(call, _arg_idx):
    return isinstance(call.op, relay.Function)

  finder = ConsumerFinder(parser, skip_predicates=[skip_function_call])
  consumers = finder.find_consumers_of_node(x)

  # The finder should recurse into the callee and report the inner add call
  assert len(consumers) == 1
  consumer_call, arg_idx = consumers[0]
  assert consumer_call == inner_add
  assert arg_idx == 0


if __name__ == "__main__":
  test_direct_consumer()
  test_skip_predicate_reaches_downstream()
  test_recurses_into_function_when_skipped()
  print("All ConsumerFinder tests passed.")
