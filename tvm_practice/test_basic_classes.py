import tvm
from tvm import relay
import numpy as np

def test_constant():
    """
    relay.Constant: 고정된 상수 값을 나타냅니다.
    주로 가중치, 바이어스 등을 표현하는 데 사용됩니다.
    """
    print("--- Testing relay.Constant ---")
    
    # 1. 생성: numpy 배열로부터 relay.Constant 객체를 생성합니다.
    np_data = np.random.rand(3, 4).astype("float32")
    r_const = relay.Constant(np_data)
    
    print("Created Constant:\n", r_const)
    print("Type:", r_const.checked_type)
    
    # 2. 값 확인: .data 속성을 통해 TVM ndarray에 접근하고, .numpy()로 numpy 배열을 얻을 수 있습니다.
    retrieved_data = r_const.data.numpy()
    print("Retrieved numpy data shape:", retrieved_data.shape)
    
    # 3. "수정": Constant는 불변이므로 직접 수정할 수 없습니다.
    # 새로운 값으로 새 Constant를 만들어야 합니다.
    new_np_data = np_data * 2
    new_r_const = relay.Constant(new_np_data)
    print("\nA new 'modified' Constant:\n", new_r_const)
    print("-" * 30 + "\n")


def test_var():
    """
    relay.Var: 데이터가 흘러다니는 통로를 나타내는 변수입니다.
    주로 함수의 입력, 중간 계산 결과 등을 나타냅니다.
    """
    print("--- Testing relay.Var ---")
    
    # 1. 생성: 이름, 타입(TensorType)을 지정하여 생성합니다.
    # TensorType은 (shape, dtype)으로 구성됩니다.
    input_var = relay.Var("x", relay.TensorType((1, 3, 224, 224), "float32"))
    
    print("Created Var:\n", input_var)
    print("Name hint:", input_var.name_hint)
    print("Type annotation:", input_var.type_annotation)
    
    # 2. "수정": Var 자체는 플레이스홀더이므로 "수정"의 개념이 다릅니다.
    # 다른 타입의 Var가 필요하면 새로 생성해야 합니다.
    new_var = relay.Var("y", shape=(), dtype="bool") # shape=()는 스칼라를 의미
    print("\nA new Var for a different purpose:\n", new_var)
    print("-" * 30 + "\n")


def test_call():
    """
    relay.Call: 연산(Operator)을 호출하는 것을 나타냅니다.
    모든 딥러닝 연산(conv2d, relu, add 등)은 Call 노드로 표현됩니다.
    """
    print("--- Testing relay.Call ---")
    
    # 1. 생성: relay.Call(op, args, attrs) 형태로 생성합니다.
    # 보통은 relay.add(arg1, arg2)와 같은 헬퍼 함수를 사용합니다.
    
    # 사용될 변수와 상수 생성
    a = relay.Var("a", shape=(10, 10), dtype="float32")
    b = relay.Var("b", shape=(10, 10), dtype="float32")
    
    # 방법 1: 헬퍼 함수 사용 (권장)
    add_call = relay.add(a, b)
    print("Created Call (with helper):\n", add_call)
    
    # 방법 2: relay.Call 직접 사용
    add_op = relay.op.get("add")
    add_call_direct = relay.Call(add_op, [a, b])
    print("\nCreated Call (direct):\n", add_call_direct)
    
    # Call 노드의 구성 요소 확인
    print("\nOp:", add_call.op)
    print("Args:", add_call.args)
    
    # 2. "수정": Call 노드를 수정하려면 새로운 Call 노드를 만들어야 합니다.
    # 예를 들어, add를 multiply로 바꾸기
    mul_call = relay.multiply(a, b)
    print("\n'Modified' Call (add -> multiply):\n", mul_call)
    print("-" * 30 + "\n")


def test_function():
    """
    relay.Function: 여러 연산을 묶어 하나의 함수로 정의합니다.
    신경망 모델 전체가 하나의 큰 Function으로 표현될 수 있습니다.
    """
    print("--- Testing relay.Function ---")
    
    # 1. 생성: relay.Function([params], body, ret_type, type_params)
    # 입력으로 사용될 Var 생성
    x = relay.Var("x", shape=(10, 10), dtype="float32")
    
    # 함수의 몸통(body)이 될 연산 생성
    body = relay.add(x, x) # x + x
    
    # 함수 생성
    func = relay.Function([x], body)
    print("Created Function:\n", func)
    
    # 함수 구성 요소 확인
    print("\nParams:", func.params)
    print("Body:\n", func.body)
    
    # 2. "수정": body나 params를 바꿔서 새 함수를 만듭니다.
    new_body = relay.multiply(x, x) # x * x
    modified_func = relay.Function([x], new_body)
    print("\n'Modified' Function (body changed):\n", modified_func)
    print("-" * 30 + "\n")


def test_let():
    """
    relay.Let: 지역 변수를 정의하고 싶을 때 사용합니다.
    `let var = value in body` 형태로, body 내에서만 var를 value처럼 사용할 수 있습니다.
    """
    print("--- Testing relay.Let ---")
    
    # 1. 생성: relay.Let(var, value, body)
    x = relay.Var("x", "float32")
    c = relay.Constant(np.array(1.0, "float32"))
    
    # let x = 1.0 in x + x
    let_expr = relay.Let(x, c, relay.add(x, x))
    print("Created Let expression:\n", let_expr)
    
    # Let을 포함하는 함수
    y = relay.Var("y", "float32")
    # f(y) = let x = y in x + x
    func_with_let = relay.Function([y], relay.Let(x, y, relay.add(x, x)))
    print("\nFunction with Let:\n", func_with_let)
    print("-" * 30 + "\n")


def test_tuple():
    """
    relay.Tuple / relay.TupleGetItem: 여러 Expr을 하나로 묶거나, 묶음에서 특정 원소를 꺼낼 때 사용합니다.
    함수가 여러 값을 반환할 때 유용합니다.
    """
    print("--- Testing Tuple ---")
    
    # 1. 생성
    x = relay.Var("x", shape=(3,), dtype="float32")
    c = relay.Constant(np.array([1, 2, 3], dtype="float32"))
    
    # (x, c) 튜플 생성
    tup = relay.Tuple([x, c])
    print("Created Tuple:\n", tup)
    
    # 튜플에서 원소 꺼내기
    item0 = relay.TupleGetItem(tup, 0) # 첫 번째 원소
    item1 = relay.TupleGetItem(tup, 1) # 두 번째 원소
    
    print("\nTupleGetItem(0):\n", item0)
    print("TupleGetItem(1):\n", item1)
    
    # 2. "수정": 새로운 튜플을 만듭니다.
    y = relay.Var("y", shape=(3,), dtype="float32")
    modified_tup = relay.Tuple([y, c])
    print("\n'Modified' Tuple:\n", modified_tup)
    print("-" * 30 + "\n")


def test_ndarray():
    """
    tvm.nd.NDArray: TVM의 다차원 배열 클래스입니다.
    CPU/GPU 메모리에 데이터를 저장하고, 연산을 수행할 수 있습니다.
    """
    print("--- Testing tvm.nd.NDArray ---")
    
    # 1. 생성: numpy 배열로부터 NDArray 생성
    np_data = np.random.rand(2, 3).astype("float32")
    nd_arr:tvm.runtime.NDArray = tvm.nd.array(np_data)
    print("Created NDArray from numpy:\n", nd_arr)
    print("Shape:", nd_arr.shape)
    print("Dtype:", nd_arr.dtype)
    
    # 2. 빈 NDArray 생성
    empty_arr = tvm.nd.empty((2, 3), dtype="float32")
    print("\nEmpty NDArray:\n", empty_arr)
    
    # slicing
    sliced_arr = nd_arr.numpy()[0:1, 0:2]
    print("\nSliced NDArray:\n", sliced_arr)

    print("-" * 30 + "\n")


if __name__ == "__main__":
    # test_constant()
    # test_var()
    # test_call()
    # test_function()
    # test_let()
    # test_tuple()
    test_ndarray()
