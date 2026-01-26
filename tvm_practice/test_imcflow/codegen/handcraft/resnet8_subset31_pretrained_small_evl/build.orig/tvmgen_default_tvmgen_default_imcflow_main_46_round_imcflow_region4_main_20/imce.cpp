#include "../common_decl.h"
void tvmgen_default_tvmgen_default_imcflow_main_46_round_imcflow_region4_main_20() {
  int hid = __builtin_IMCE_GET_CORE_HID();
  int wid = __builtin_IMCE_GET_CORE_WID();
  short16 var1; // (TensorEdge((-98, odata), (112, lhs)), 0)
  short16 var2; // (TensorEdge((-99, odata), (112, rhs)), 0)
  short16 var3; // (AddBlock(gid: 112), 0)
  if (hid == 0 && wid == 1) { // imce_0_1
  }
  else if (hid == 0 && wid == 2) { // imce_0_2
  }
  else if (hid == 0 && wid == 3) { // imce_0_3
  }
  else if (hid == 0 && wid == 4) { // imce_0_4
  }
  else if (hid == 1 && wid == 1) { // imce_1_1
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
    // generate: call_created_loop
    for (int i1 = 0; i1 < 16; i1++) { // generate : call_created_loop
      // generate: add standalone

      var1 = __builtin_IMCE_RECV(2); // TensorEdge((-98, odata), (112, lhs)), inode_0_0 -> imce_3_4
      var2 = __builtin_IMCE_RECV(3); // TensorEdge((-99, odata), (112, rhs)), inode_1_0 -> imce_3_4
      // generate: add

      var3 = __builtin_IMCE_ADD(var1, var2, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(1, var3, 2, 0); // TensorEdge((112, odata), (113, func_out0)), imce_3_4 -> inode_3_0
      // endgenerate: add standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
}
