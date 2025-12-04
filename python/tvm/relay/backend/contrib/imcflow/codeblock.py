from abc import *
from typing import *
from copy import copy
from tvm.contrib.imcflow import NodeID
from textwrap import indent
from contextlib import contextmanager
from enum import Enum
import pdb

class UniqueVar:
  _instances = {}
  _counter = 0

  def __new__(cls, obj, dtype="short16"):
    """Ensure only one instance per unique obj and dtype combination."""
    # Normalize the key: use _original for CodeBlock instances in tuples
    key = cls._normalize_key(obj)

    if key not in cls._instances:
      instance = super(UniqueVar, cls).__new__(cls)
      cls._instances[key] = instance
      cls._counter += 1

      # set the instance variables
      instance.name = f"var{cls._counter}"
      instance.dtype = dtype
      instance.static = False

    assert cls._instances[key].dtype == dtype, \
        f"UniqueVar {obj} already exists with dtype {cls._instances[key].dtype}"

    return cls._instances[key]

  @staticmethod
  def _normalize_key(obj):
    """Normalize key by using _original reference for CodeBlock instances."""
    # Handle tuple keys like (CodeBlock, index) or (TensorEdge, index)
    if isinstance(obj, tuple):
      normalized = tuple(
        elem._original if hasattr(elem, '_original') else elem
        for elem in obj
      )
      return normalized
    # Handle direct object keys
    return getattr(obj, '_original', obj)

  def set_static(self):
    # FIXME: we don't know if set_static is always done prior to another variable use
    self.static = True

  def __str__(self):
    return self.name

  @classmethod
  def get_decls(cls):
    for obj, value in cls._instances.items():
      yield f"{value.dtype} {value.name}; // {obj}"

  @classmethod
  def get_decls_str(cls):
    return "\n".join(cls.get_decls())

  @classmethod
  def reset(cls):
    cls._instances = {}
    cls._counter = 0


class CodePhase(Enum):
  INIT = "INIT"
  EXEC = "EXEC"
  END = "END"

class CodeBlock(metaclass=ABCMeta):
  def __init__(self):
    self.next = None
    self._original = self  # Track original object for UniqueVar

  def __copy__(self):
    """Shallow copy that preserves reference to original object."""
    cls = self.__class__
    new_obj = cls.__new__(cls)
    new_obj.__dict__.update(self.__dict__)
    new_obj._original = self._original  # Keep reference to original
    # Don't reset next - preserve the linked list chain
    return new_obj

  @abstractmethod
  def content(self) -> str:
    pass

  def __str__(self) -> str:
    if not self.next:
      return str(self.content())
    if not self.content():
      return str(self.next)
    return str(self.content()) + "\n" + str(self.next)

  def __add__(self, other):
    if isinstance(other, str):
      other = TextBlock(other)
    if isinstance(other, CodeBlock):
      new_block = copy(self)
      ptr = new_block
      while ptr.next is not None:
        ptr = ptr.next
      ptr.next = copy(other)
      return new_block
    raise TypeError(
        f"unsupported operand type(s) for +: 'CodeBlock' and '{type(other)}'")

  def __iadd__(self, other):
    if isinstance(other, str):
      other = TextBlock(other)
    if isinstance(other, CodeBlock):
      ptr = self
      while ptr.next is not None:
        ptr = ptr.next
      ptr.next = other
      return self
    raise TypeError(
        f"unsupported operand type(s) for +=: 'CodeBlock' and '{type(other)}'")


class TextBlock(CodeBlock):
  def __init__(self, text: str):
    super().__init__()
    self.text = text

  def content(self) -> str:
    return self.text


class SimpleFor(CodeBlock):
  scope = 0
  count_stack=[]

  def __init__(self, count: int, body: Union[str, CodeBlock], annotation: str = ""):
    super().__init__()
    self.annotation = annotation
    self.count = int(count)
    self.body = body

  @contextmanager
  def manage_scope(self):
    SimpleFor.scope += 1
    SimpleFor.count_stack.append(self.count)
    try:
      yield f"i{SimpleFor.scope}"
    finally:
      SimpleFor.scope -= 1
      SimpleFor.count_stack.pop()
  
  @property
  def annotation_str(self):
    return f" : {self.annotation}" if self.annotation else ""

  def content(self) -> CodeBlock:
    if self.count == 0:
      return TextBlock(f"// loop ignored with loop count == 0{self.annotation_str}\n")

    if self.count == 1:
      formatted_body = self.body(0) if callable(self.body) else str(self.body)
      code = TextBlock("")
      code += f"// generate{self.annotation_str}. loop count == 1"
      code += formatted_body
      code += f"// endgenerate{self.annotation_str}"
      return code

    with self.manage_scope() as var_iter:
      formatted_body = self.body(var_iter) if callable(
          self.body) else str(self.body)

      code = TextBlock("")
      code += f"for (int {var_iter} = 0; {var_iter} < {self.count}; {var_iter}++) {{ // generate{self.annotation_str}"
      # FIXME: explicit str is NOT the right way
      # but currently is necessay for scope to work.
      # since before current content exits, the body's content should be evaluated
      code += indent(formatted_body, '  ')
      code += f"}} // endgenerate{self.annotation_str}"
    return code


class CtrlBlock(CodeBlock):
  """
  DONE, HALT, INTRT, STANDBY, SET_ADDR_CNT, SET_FLAG
  NOP, STEP, STOP
  """
  def __init__(self, ctrl: str = "", annotation: str = ""):
    super().__init__()
    self.annotation = annotation
    self.ctrl = ctrl
    assert ctrl == "STOP", "only STOP CtrlBlock is supported for now. we to extend"

  def content(self) -> CodeBlock:
    if self.annotation:
      code = TextBlock("")
      code += f"// generate: {self.annotation}"
      code += copy(self._content())
      code += f"// endgenerate: {self.annotation}"
      return code
    else:
      return self._content()

  def _content(self) -> CodeBlock:
    code = TextBlock("")
    if self.ctrl == "STOP":
      code += "__builtin_IMCE_STOP();"
    return code



class NodeCodeBlockManager:
  """A class that manages and generates code blocks for each node."""

  def __init__(self):
    # reset UniqueVar for each new instance of codeblocks
    UniqueVar.reset()
    self.blocks = {key: {CodePhase.INIT: [], CodePhase.EXEC: [], CodePhase.END: []}
                   for key in self.nodes}

  @property
  @abstractmethod
  def nodes(self) -> List[NodeID]:
    pass

  @property
  @abstractmethod
  def target(self) -> str:
    pass

  @abstractmethod
  def start_block(self) -> str:
    """
    The subclass should use UniqueVar.get_decls() to declare variables.
    """
    pass

  @abstractmethod
  def end_block(self) -> str:
    pass

  def append(self, hid, block, block_type: CodePhase = CodePhase.EXEC):
    self.blocks[hid][block_type].append(block)

  def generate_body(self) -> str:
    code = ""
    first = True
    for node in self.nodes:
      condition = f"if" if first else f"else if"
      code += f"{condition} (hid == {node.to_coord(0)} && wid == {node.to_coord(1)}) {{ // {node.name}\n"
      # Generate SETUP blocks first
      for codeblock in self.blocks[node][CodePhase.INIT]:
        code += f"{indent(str(codeblock), '  ')}\n"
      # Generate COMPUTE blocks next
      for codeblock in self.blocks[node][CodePhase.EXEC]:
        code += f"{indent(str(codeblock), '  ')}\n"
      # Generate END blocks last
      for codeblock in self.blocks[node][CodePhase.END]:
        code += f"{indent(str(codeblock), '  ')}\n"
      code += "}\n"
      first = False
    return code

  def generate(self) -> str:
    # generate body first to determine variables first
    # then generate start, where variables are declared
    body = self.generate_body()

    start = self.start_block()
    end = self.end_block()

    return start + indent(body, '  ') + end
  
  def get_blocks(self, phase :List[CodePhase] = None , nodes : List[NodeID] = None) -> List[CodeBlock]:
    """Get all INIT code blocks for the specified nodes.

    Args:
        nodes: List of NodeID instances to get INIT blocks for.
               If None, gets INIT blocks for all nodes.
    Returns:
        List of INIT CodeBlock instances.
    """
    if nodes is None:
      nodes = self.nodes
    if phase is None:
      phase = [CodePhase.INIT, CodePhase.EXEC, CodePhase.END]

    init_blocks = []
    for node in nodes:
      for phase_ in phase:
        init_blocks.extend(self.blocks[node][phase_])
    return init_blocks