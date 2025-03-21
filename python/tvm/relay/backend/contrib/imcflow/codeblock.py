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
    if obj not in cls._instances:
      instance = super(UniqueVar, cls).__new__(cls)
      cls._instances[obj] = instance
      cls._counter += 1

      # set the instance variables
      instance.name = f"var{cls._counter}"
      instance.dtype = dtype
      instance.static = False

    assert cls._instances[obj].dtype == dtype, \
        f"UniqueVar {obj} already exists with dtype {cls._instances[obj].dtype}"

    return cls._instances[obj]

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
  def reset(cls):
    cls._instances = {}
    cls._counter = 0


class CodePhase(Enum):
  INIT = "INIT"
  EXEC = "EXEC"


class CodeBlock(metaclass=ABCMeta):
  def __init__(self):
    self.next = None

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

  def __init__(self, count: int, body: Union[str, CodeBlock], annotation: str = ""):
    super().__init__()
    self.annotation = annotation
    self.count = int(count)
    self.body = body

  @contextmanager
  def manage_scope(self):
    SimpleFor.scope += 1
    try:
      yield f"i{SimpleFor.scope}"
    finally:
      SimpleFor.scope -= 1

  def content(self) -> CodeBlock:
    if self.count == 0:
      return TextBlock("")

    if self.count == 1:
      formatted_body = self.body(0) if callable(self.body) else str(self.body)
      return TextBlock(formatted_body)


    with self.manage_scope() as var_iter:
      formatted_body = self.body(var_iter) if callable(self.body) else str(self.body)

      if self.annotation:
        code = TextBlock("")
        code += f"for (int {var_iter} = 0; {var_iter} < {self.count}; {var_iter}++) {{ // generate: {self.annotation}"
        # FIXME: explicit str is NOT the right way
        # but currently is necessay for scope to work.
        # since before current content exits, the body's content should be evaluated
        code += indent(formatted_body, '  ')
        code += f"}} // endgenerate: {self.annotation}"
      else:
        code = TextBlock("")
        code += f"for (int {var_iter} = 0; {var_iter} < {self.count}; {var_iter}++) {{"
        code += indent(formatted_body, '  ')
        code += f"}}"
    return code


class CodeBlockStart(CodeBlock):
  def __init__(self, name: str, target: str):
    super().__init__()
    assert target in ["inode", "imce"], \
        "target must be either 'inode' or 'imce'"
    self.target = target
    self.func_name = name

  def content(self) -> CodeBlock:
    code = TextBlock("")
    code += "#include \"common_decl.h\"\n"
    code += f"void {self.func_name}() {{"
    if self.target == "imce":
      code += f"  int hid = __builtin_IMCE_GET_CORE_HID();"
      code += f"  int wid = __builtin_IMCE_GET_CORE_WID();\n"
    else:
      code += f"  int hid = __builtin_INODE_GET_CORE_HID();"
      code += f"  int wid = 0;\n"
    for decl in UniqueVar.get_decls():
      code += f"  {decl}"
    code += "\n"

    return code


class CodeBlockEnd(CodeBlock):
  def __init__(self):
    super().__init__()

  def content(self) -> str:
    return "}\n"


class CtrlBlock(CodeBlock):
  """
  DONE, HALT, INTRT, STANDBY, SET_ADDR_CNT, SET_FLAG
  NOP, STEP, STOP
  """
  pass


class CodeBlocks:
  """A class that manages and generates code blocks for each node."""

  def __init__(self, name: str, target: str = "imce"):
    # reset UniqueVar for each new instance of codeblocks
    UniqueVar.reset()

    if target == "imce":
      self.nodes = NodeID.imces()
    elif target == "inode":
      self.nodes = NodeID.inodes()
    else:
      raise ValueError(f"Unknown target: {target}")

    self.start = CodeBlockStart(name, target)
    self.blocks = {key: {CodePhase.INIT: [], CodePhase.EXEC: []}
                   for key in self.nodes}
    self.end = CodeBlockEnd()

  def append(self, hid, block, block_type: CodePhase = CodePhase.EXEC):
    self.blocks[hid][block_type].append(block)

  def generate_body(self) -> str:
    code = ""
    first = True
    for node in self.nodes:
      condition = f"if" if first else f"else if"
      code += f"{condition} (hid == {node.to_coord(0)} && wid == {node.to_coord(1)}) {{\n"
      # Generate SETUP blocks first
      for codeblock in self.blocks[node][CodePhase.INIT]:
        code += f"{indent(str(codeblock), '  ')}\n"
      # Generate COMPUTE blocks next
      for codeblock in self.blocks[node][CodePhase.EXEC]:
        code += f"{indent(str(codeblock), '  ')}\n"
      code += "}\n"
      first = False
    return code

  def generate(self) -> str:
    # generate body first to determine variables first
    # then generate start, where variables are declared
    body = self.generate_body()

    start = str(self.start)
    end = str(self.end)

    return start + body + end
