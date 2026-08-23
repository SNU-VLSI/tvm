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

  def __new__(cls, obj, dtype="short16", pointer_type=False):
    """Ensure only one instance per unique obj and dtype combination."""
    key = cls._normalize_key(obj)

    if key not in cls._instances:
      instance = super(UniqueVar, cls).__new__(cls)
      cls._instances[key] = instance
      cls._counter += 1
      instance.name = f"var{cls._counter}"
      instance.dtype = dtype
      instance.pointer_type = pointer_type
      instance.static = False

    assert cls._instances[key].dtype == dtype and cls._instances[key].pointer_type == pointer_type, \
        f"UniqueVar {obj} already exists with dtype {cls._instances[key].dtype}"

    return cls._instances[key]

  @staticmethod
  def _normalize_key(obj):
    if isinstance(obj, tuple):
      normalized = tuple(
        elem._original if hasattr(elem, '_original') else elem
        for elem in obj
      )
      return normalized
    return getattr(obj, '_original', obj)

  def set_static(self):
    self.static = True

  def __str__(self):
    return self.name

  @classmethod
  def get_decls(cls):
    for obj, value in cls._instances.items():
      if value.pointer_type:
        yield f"{value.dtype}* {value.name}; // {obj}"
      else:
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
  EXEC_TILE = "EXEC_TILE"
  END = "END"

class CodeBlock(ABC):
  def __init__(self, annotation: str = ""):
    self.annotation = annotation
    self._original = self  # Track original object for UniqueVar

  def render(self) -> str:
    """Render the block to a string. This is the public API."""
    content = self._render()
    if not content:
      return ""
    
    if self.annotation:
      return f"// generate: {self.annotation}\n{content}\n// endgenerate: {self.annotation}"
    return content

  @abstractmethod
  def _render(self) -> str:
    """Implementation of rendering logic. Returns the raw string content."""
    pass

  def dump(self, indent_level=0) -> str:
    """Dump the structure of the block for debugging."""
    indent_str = "  " * indent_level
    return f"{indent_str}{self.__class__.__name__}(annotation='{self.annotation}')"

  def __str__(self) -> str:
    return self.render()

  def __add__(self, other):
    """Combine two blocks into a SequentialBlock."""
    blocks = []
    
    # Flatten if self is already a SequentialBlock without annotation
    if isinstance(self, SequentialBlock) and not self.annotation:
      blocks.extend(self.blocks)
    else:
      blocks.append(self)
      
    if isinstance(other, str):
      other = TextBlock(other)
      
    # Flatten if other is already a SequentialBlock without annotation
    if isinstance(other, SequentialBlock) and not other.annotation:
      blocks.extend(other.blocks)
    else:
      blocks.append(other)
      
    return SequentialBlock(blocks)

  def __iadd__(self, other):
    return self + other


class TextBlock(CodeBlock):
  def __init__(self, text: str):
    super().__init__()
    self.text = text
  
  def _render(self) -> str:
    return self.text
  
  def dump(self, indent_level=0) -> str:
    indent_str = "  " * indent_level
    preview = self.text.replace("\n", "\\n")
    if len(preview) > 50:
      preview = preview[:47] + "..."
    return f"{indent_str}TextBlock(text='{preview}')"


class DeferredTextBlock(CodeBlock):
  """TextBlock whose content is computed lazily at render() time.

  Needed when the content depends on state attached AFTER block construction
  (e.g. SendBlock's pre-send rendezvous requires the pair_manager, which the
  builder wires up only after the block tree is assembled). The callable is
  invoked at render; returning ""/None renders nothing.
  """

  def __init__(self, fn, annotation: str = ""):
    super().__init__(annotation)
    self.fn = fn

  def _render(self) -> str:
    return self.fn() or ""

  def dump(self, indent_level=0) -> str:
    indent_str = "  " * indent_level
    return f"{indent_str}DeferredTextBlock()"


class SequentialBlock(CodeBlock):
  """A block that holds a list of blocks and renders them sequentially."""
  def __init__(self, blocks: List[CodeBlock] = None, annotation: str = ""):
    super().__init__(annotation)
    self.blocks = blocks if blocks is not None else []

  def add(self, block: CodeBlock):
    if isinstance(block, str):
        block = TextBlock(block)
    self.blocks.append(block)
    return self

  def _render(self) -> str:
    return "\n".join([b.render() for b in self.blocks if b])

  def dump(self, indent_level=0) -> str:
    indent_str = "  " * indent_level
    lines = [f"{indent_str}SequentialBlock(annotation='{self.annotation}')"]
    return "\n".join(lines)


class SimpleFor(CodeBlock):
  scope = 0
  count_stack=[]

  def __init__(self, count: int, body: Union[str, CodeBlock], annotation: str = ""):
    super().__init__(annotation)
    if not isinstance(count, UniqueVar):
      self.count = int(count)
      self.const_count = True
    else:
      self.count = count
      self.const_count = False
    if isinstance(body, str):
      self.body = TextBlock(body)
    else:
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

  def _render(self) -> str:
    if self.const_count and self.count == 0:
      return f"// loop ignored with loop count == 0{self.annotation_str}"

    if self.const_count and self.count == 1:
      # COMPATIBILITY FIX: If the user code passes a lambda, we execute it to get the string/block.
      if callable(self.body):
          formatted_body = self.body(0)
          if isinstance(formatted_body, CodeBlock):
              content = formatted_body.render()
          else:
              content = str(formatted_body)
      else:
          content = self.body.render()
      return f"// generate{self.annotation_str}. loop count == 1\n{content}\n// endgenerate{self.annotation_str}"

    with self.manage_scope() as var_iter:
      # COMPATIBILITY FIX: If the user code passes a lambda, we execute it to get the string/block.
      if callable(self.body):
          formatted_body = self.body(var_iter)
          if isinstance(formatted_body, CodeBlock):
              content = formatted_body.render()
          else:
              content = str(formatted_body)
      else:
          content = self.body.render()

      code = f"for (int {var_iter} = 0; {var_iter} < {self.count}; {var_iter}++) {{ // generate{self.annotation_str}\n"
      code += indent(content, '  ')
      code += f"\n}} // endgenerate{self.annotation_str}"
      return code
      
  def dump(self, indent_level=0) -> str:
    indent_str = "  " * indent_level
    lines = [f"{indent_str}SimpleFor(count={self.count}, annotation='{self.annotation}')"]
    return "\n".join(lines)


class CtrlBlock(CodeBlock):
  def __init__(self, ctrl: str = "", annotation: str = ""):
    super().__init__(annotation)
    self.ctrl = ctrl
    assert ctrl == "STOP", "only STOP CtrlBlock is supported for now. we to extend"

  def _render(self) -> str:
    if self.ctrl == "STOP":
      return "__builtin_IMCE_STOP();"
    return ""


class NodeCodeBlockManager:
  """A class that manages and generates code blocks for each node."""

  def __init__(self):
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
    pass

  @abstractmethod
  def end_block(self) -> str:
    pass

  def append(self, hid, block, block_type: CodePhase = CodePhase.EXEC):
    print(f"[IMCE CODE BLOCK MANAGER] append block to hid={hid}, type={block_type}, block={type(block).__name__}")
    # C1b (C) wave-launch: stamp each block with the launch wave the builder is
    # currently emitting (default 0 = single-launch, so this is inert unless the
    # builder sets current_wave under merge mode). Used by generate_body(wave=k)
    # to emit only wave-k blocks into that wave's imce.cpp for per-(core,wave)
    # IMEM. `_wave` is advisory metadata on the block; renderers ignore it.
    try:
      block._wave = getattr(self, "current_wave", 0)
    except Exception:
      pass
    self.blocks[hid][block_type].append(block)

  def generate_body(self, wave=None) -> str:
    """Render the per-core body. When wave is None (default) emit ALL blocks
    (stock single-launch / non-merged path, byte-identical). When wave=k, emit
    ONLY blocks tagged with that wave -> the wave-k IMEM program per core."""
    code = ""
    first = True
    for node in self.nodes:
      condition = f"if" if first else f"else if"
      code += f"{condition} (hid == {node.to_coord(0)} && wid == {node.to_coord(1)}) {{ // {node.name}\n"
      for phase in [CodePhase.INIT, CodePhase.EXEC, CodePhase.END]:
        for codeblock in self.blocks[node][phase]:
          if wave is not None and getattr(codeblock, "_wave", 0) != wave:
            continue
          code += f"{indent(codeblock.render(), '  ')}\n"
      code += "}\n"
      first = False
    return code

  def generate(self, wave=None) -> str:
    body = self.generate_body(wave=wave)
    start = self.start_block()
    end = self.end_block()
    return start + indent(body, '  ') + end

  def wave_indices(self) -> List[int]:
    """C1b (C): the distinct launch waves present across all appended blocks.
    Returns [0] when nothing is wave-tagged (every non-merged region) so callers
    that iterate waves collapse to the single stock program -> byte-identical."""
    waves = set()
    for node in self.nodes:
      # Iterate only the phases this manager actually holds (IMCE managers have
      # no EXEC_TILE bucket, so key off the node's own phase dict).
      for phase, blks in self.blocks[node].items():
        for cb in blks:
          waves.add(getattr(cb, "_wave", 0))
    return sorted(waves) if waves else [0]

  def get_blocks(self, phase :List[CodePhase] = None , nodes : List[NodeID] = None) -> List[CodeBlock]:
    if nodes is None:
      nodes = self.nodes
    if phase is None:
      phase = [CodePhase.INIT, CodePhase.EXEC, CodePhase.EXEC_TILE, CodePhase.END]

    init_blocks = []
    for node in nodes:
      for phase_ in phase:
        init_blocks.extend(self.blocks[node][phase_])
    return init_blocks