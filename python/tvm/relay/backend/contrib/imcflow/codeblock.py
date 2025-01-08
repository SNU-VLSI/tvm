from abc import *
from typing import *
from copy import deepcopy
from tvm.contrib.imcflow import NodeID
from textwrap import indent
from contextlib import contextmanager
from enum import Enum
import pdb


class UniqueVar:
  _var_map = {}
  _counter = 0

  def __init__(self, obj, dtype="short16"):
    """Generates a unique variable name for the given object or retrieves an existing one."""
    if obj not in UniqueVar._var_map:
      UniqueVar._var_map[obj] = (f"var{UniqueVar._counter}", dtype)
      UniqueVar._counter += 1
    self.name = UniqueVar._var_map[obj][0]
    self.dtype = UniqueVar._var_map[obj][1]

  def __str__(self):
    return self.name

  @staticmethod
  def get_decls():
    for value in UniqueVar._var_map.values():
      yield f"{value[1]} {value[0]};"


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
      new_block = deepcopy(self)
      ptr = new_block
      while ptr.next is not None:
        ptr = ptr.next
      ptr.next = deepcopy(other)
      return new_block
    raise TypeError(f"unsupported operand type(s) for +: 'CodeBlock' and '{type(other)}'")

  def __iadd__(self, other):
    if isinstance(other, str):
      other = TextBlock(other)
    if isinstance(other, CodeBlock):
      ptr = self
      while ptr.next is not None:
        ptr = ptr.next
      ptr.next = other
      return self
    raise TypeError(f"unsupported operand type(s) for +=: 'CodeBlock' and '{type(other)}'")

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
    elif self.count == 1:
      return self.body

    with self.manage_scope() as var_iter:
      if self.annotation:
        code = TextBlock("")
        code += f"for (int {var_iter} = 0; {var_iter} < {self.count}; {var_iter}++) {{ // generate: {self.annotation}"
        # FIXME: explicit str is NOT the right way
        # but currently is necessay for scope to work.
        # since before current content exits, the body's content should be evaluated
        code += indent(str(self.body), '  ')
        code += f"}} // endgenerate: {self.annotation}"
      else:
        code = TextBlock("")
        code += f"for (int {var_iter} = 0; {var_iter} < {self.count}; {var_iter}++) {{"
        code += indent(str(self.body), '  ')
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
    code += f"void {self.func_name}() {{"
    if self.target == "imce":
      code += f"  int hid = __builtin_IMCE_GET_CORE_HID();"
      code += f"  int wid = __builtin_IMCE_GET_CORE_WID();\n"
    else:
      code += f"  int hid = __builtin_INODE_GET_CORE_HID();\n"
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

  def __init__(self, name, target="imce"):
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

  def generate_body(self):
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

  def generate(self):
    # generate body first to determine variables first
    # then generate start, where variables are declared
    body = self.generate_body()

    start = str(self.start)
    end = str(self.end)

    return start + body + end
