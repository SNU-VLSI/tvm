from abc import *
from tvm.contrib.imcflow import *


class CodeBlockBase(metaclass=ABCMeta):
  @abstractmethod
  def generate(self) -> str:
    pass


class PolicyUpdateCodeBlock(CodeBlockBase):
  """ Code block for updating policy table for given inode's hw node id  """
  def __init__(self, inode: NodeID):
    for imce in inode.slaves():
      policytabledict = ImcflowDeviceConfig.PolicyTableDict[imce]


  def generate(self) -> str:
    code = "int policy_table_start_address;\n"
    for entry in self.tensor_edge.code_block["entries"]:
      db_name = entry["data_block"]
      address = self.tensor_edge.dbl_[db_name][0]
      size = self.tensor_edge.dbl_[db_name][1]

      code += f"\npolicy_table_start_address = {address};\n"
      for i in range(0, size, 32):
        code += f"__builtin_INODE_PU(policy_table_start_address, {i}, {int(i / 32)}, {entry['col_id']});\n"

    return ""


# for op:
#   if op == "relu":
#     codeblocks.append(PolicyUpdateBlock(Node.inode_0))
#     codeblocks.append(CtrlBlock())
#     codeblocks.append(Conv2dBlock())

#     codeblocks.generate()
