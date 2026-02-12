#include "../common_decl.h"
void tvmgen_default_tvmgen_default_imcflow_main_46_round_imcflow_region4_main_33() {
  int hid = __builtin_IMCE_GET_CORE_HID();
  int wid = __builtin_IMCE_GET_CORE_WID();
  short16 var1; // (TensorEdge((-101, odata), (115, lhs)), 0)
  short16 var2; // (TensorEdge((-102, odata), (115, rhs)), 0)
  short16 var3; // (TensorEdge((-101, odata), (115, lhs)), 1)
  short16 var4; // (TensorEdge((-102, odata), (115, rhs)), 1)
  short16 var5; // (TensorEdge((-101, odata), (115, lhs)), 2)
  short16 var6; // (TensorEdge((-102, odata), (115, rhs)), 2)
  short16 var7; // (TensorEdge((-101, odata), (115, lhs)), 3)
  short16 var8; // (TensorEdge((-102, odata), (115, rhs)), 3)
  short16 var9; // (AddBlock(gid: 115), 0)
  short16 var10; // (AddBlock(gid: 115), 1)
  short16 var11; // (AddBlock(gid: 115), 2)
  short16 var12; // (AddBlock(gid: 115), 3)
  if (hid == 0 && wid == 1) { // imce_0_1
  }
  else if (hid == 0 && wid == 2) { // imce_0_2
  }
  else if (hid == 0 && wid == 3) { // imce_0_3
  }
  else if (hid == 0 && wid == 4) { // imce_0_4
  }
  else if (hid == 1 && wid == 1) { // imce_1_1
    // generate: call_created_loop
    for (int i1 = 0; i1 < 64; i1++) { // generate : call_created_loop
      // generate: add standalone

      var1 = __builtin_IMCE_RECV(2); // TensorEdge((-101, odata), (115, lhs)), inode_0_0 -> imce_1_1
      var2 = __builtin_IMCE_RECV(3); // TensorEdge((-102, odata), (115, rhs)), inode_1_0 -> imce_1_1
      var3 = __builtin_IMCE_RECV(2); // TensorEdge((-101, odata), (115, lhs)), inode_0_0 -> imce_1_1
      var4 = __builtin_IMCE_RECV(3); // TensorEdge((-102, odata), (115, rhs)), inode_1_0 -> imce_1_1
      var5 = __builtin_IMCE_RECV(2); // TensorEdge((-101, odata), (115, lhs)), inode_0_0 -> imce_1_1
      var6 = __builtin_IMCE_RECV(3); // TensorEdge((-102, odata), (115, rhs)), inode_1_0 -> imce_1_1
      var7 = __builtin_IMCE_RECV(2); // TensorEdge((-101, odata), (115, lhs)), inode_0_0 -> imce_1_1
      var8 = __builtin_IMCE_RECV(3); // TensorEdge((-102, odata), (115, rhs)), inode_1_0 -> imce_1_1
      // generate: add

      var9 = __builtin_IMCE_ADD(var1, var2, 15);
      var10 = __builtin_IMCE_ADD(var3, var4, 15);
      var11 = __builtin_IMCE_ADD(var5, var6, 15);
      var12 = __builtin_IMCE_ADD(var7, var8, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(1, var9, 2, 0); // TensorEdge((115, odata), (116, func_out0)), imce_1_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var10, 2, 0); // TensorEdge((115, odata), (116, func_out0)), imce_1_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var11, 2, 0); // TensorEdge((115, odata), (116, func_out0)), imce_1_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var12, 2, 0); // TensorEdge((115, odata), (116, func_out0)), imce_1_1 -> inode_3_0
      // endgenerate: add standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 1 && wid == 2) { // imce_1_2
  }
  else if (hid == 1 && wid == 3) { // imce_1_3
  }
  else if (hid == 1 && wid == 4) { // imce_1_4
  }
  else if (hid == 2 && wid == 1) { // imce_2_1
  }
  else if (hid == 2 && wid == 2) { // imce_2_2
  }
  else if (hid == 2 && wid == 3) { // imce_2_3
  }
  else if (hid == 2 && wid == 4) { // imce_2_4
  }
  else if (hid == 3 && wid == 1) { // imce_3_1
  }
  else if (hid == 3 && wid == 2) { // imce_3_2
  }
  else if (hid == 3 && wid == 3) { // imce_3_3
  }
  else if (hid == 3 && wid == 4) { // imce_3_4
  }
}
