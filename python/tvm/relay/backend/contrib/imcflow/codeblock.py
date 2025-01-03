from abc import *
from typing import *
from tvm.contrib.imcflow import NodeID
from textwrap import indent
import pdb


class CodeBlock(metaclass=ABCMeta):
  def __init__(self, annotation: str = ""):
    self.annotation = annotation

  def _content(self) -> str:
    pass

  def __str__(self) -> str:
    _content = str(self._content())
    if self.annotation and _content:
      return (
          f"// generate: {self.annotation}\n"
          f"{_content}"
          f"// endgenerate: {self.annotation}\n"
      )
    else:
      return _content

  def __add__(self, other):
    return str(self) + str(other)

  def __radd__(self, other):
    return str(other) + str(self)


class SimpleFor(CodeBlock):
  def __init__(self, count: int, body: Union[str, CodeBlock], annotation: str = ""):
    super().__init__(annotation)
    self.count = count
    self.body = body

  def __str__(self) -> str:
    if self.count == 0:
      return ""
    elif self.count == 1:
      return f"{self.body}\n"
    elif self.annotation:
      return (
          f"for (int i = 0; i < {self.count}; i++) {{ // generate: {self.annotation}\n"
          f"{indent(self.body, '  ')}\n"
          f"}} // endgenerate: {self.annotation}\n"
      )
    else:
      return (
          f"for (int i = 0; i < {self.count}; i++) {{\n"
          f"{indent(self.body, '  ')}\n"
          f"}}\n"
      )


class CodeBlockStart(CodeBlock):
  def __init__(self, name: str, target: str):
    assert target in ["inode", "imce"], \
        "target must be either 'inode' or 'imce'"
    self.target = target
    self.func_name = name

  def __str__(self) -> str:
    code = f"void {self.func_name}() {{\n"
    if self.target == "imce":
      code += f"  int hid = __builtin_IMCE_GET_CORE_HID();\n"
      code += f"  int wid = __builtin_IMCE_GET_CORE_WID();\n\n"
    else:
      code += f"  int hid = __builtin_INODE_GET_CORE_HID();\n\n"
    return code


class CodeBlockEnd(CodeBlock):
  def __init__(self):
    pass

  def __str__(self) -> str:
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
    self.blocks = {key: [] for key in self.nodes}
    self.end = CodeBlockEnd()

  def append(self, hid, block):
    self.blocks[hid].append(block)

  def generate(self):
    # TODO: code generation should handle duplicate variable names
    code = str(self.start)
    first = True
    for node in self.nodes:
        condition = f"if" if first else f"else if"
        code += f"{condition} (hid == {node.to_coord(0)} && wid == {node.to_coord(1)}) {{\n"
        for codeblock in self.blocks[node]:
            code += f"{indent(str(codeblock), '  ')}\n"
        code += "}\n"
        first = False
    code += str(self.end)

    return code
