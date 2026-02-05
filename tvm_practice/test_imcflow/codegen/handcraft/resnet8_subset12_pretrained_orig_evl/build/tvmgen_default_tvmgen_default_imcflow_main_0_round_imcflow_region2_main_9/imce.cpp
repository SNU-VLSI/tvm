#include "../common_decl.h"
void tvmgen_default_tvmgen_default_imcflow_main_0_round_imcflow_region2_main_9() {
  int hid = __builtin_IMCE_GET_CORE_HID();
  int wid = __builtin_IMCE_GET_CORE_WID();
  short16 var1; // (ConvBlock(gid: 34), 0)
  short16 var2; // (ConvBlock(gid: 34), 1)
  short16 var3; // (ConvBlock(gid: 34), 2)
  short16 var4; // (ConvBlock(gid: 34), 3)
  short16 var5; // (TensorEdge(((33, -31), min), ((33, 31), min)), 0)
  short16 var6; // (TensorEdge(((33, -32), max), ((33, 31), max)), 0)
  short16 var7; // (TensorEdge((-27, odata), ((33, 30), lhs)), 0)
  short16 var8; // (TensorEdge((-28, odata), ((33, 30), rhs)), 0)
  short16 var9; // (MinmaxQuantBlock(gid: 31), 0)
  short16 var10; // (MinmaxQuantBlock(gid: 31), 1)
  short16 var11; // (MinmaxQuantBlock(gid: 31), 2)
  short16 var12; // (MinmaxQuantBlock(gid: 31), 3)
  short16 var13; // (AddBlock(gid: 30), 0)
  short16 var14; // (TensorEdge((-34, config), (34, config)), 0)
  short16 var15; // (TensorEdge(((33, 31), odata), (34, data)), 0)
  short16 var16; // (TensorEdge(((33, 31), odata), (34, data)), 1)
  short16 var17; // (TensorEdge(((33, 31), odata), (34, data)), 2)
  short16 var18; // (TensorEdge(((33, 31), odata), (34, data)), 3)
  if (hid == 0 && wid == 1) { // imce_0_1
  }
  else if (hid == 0 && wid == 2) { // imce_0_2
  }
  else if (hid == 0 && wid == 3) { // imce_0_3
  }
  else if (hid == 0 && wid == 4) { // imce_0_4
  }
  else if (hid == 1 && wid == 1) { // imce_1_1
    // generate: TensorEdge(((33, -31), min), ((33, 31), min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge(((33, -31), min), ((33, 31), min)), min write
    // generate: TensorEdge(((33, -32), max), ((33, 31), max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge(((33, -32), max), ((33, 31), max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 1024; i1++) { // generate : call_created_loop
      // generate: imcflow.preop-minmax_wrapper

      __builtin_IMCE_SETFLAG(1); 
      __builtin_IMCE_STANDBY(0, 1); 
      __builtin_IMCE_STANDBY(5, 1); 
      __builtin_IMCE_SETFLAG(0); 
      var7 = __builtin_IMCE_RECV(2); // TensorEdge((-27, odata), ((33, 30), lhs)), inode_0_0 -> imce_1_1
      var8 = __builtin_IMCE_RECV(3); // TensorEdge((-28, odata), ((33, 30), rhs)), inode_1_0 -> imce_1_1
      // generate: imcflow.preop-minmax_block
      // generate: add

      var13 = __builtin_IMCE_ADD(var7, var8, 15);
      // endgenerate: add
      // generate: min_max_quantize

      __builtin_IMCE_STANDBY(11, 1); 
      __builtin_IMCE_MM_QUANT(var13, 0, 15, 0);
      var9 = __builtin_IMCE_GET_QREG(0);
      var10 = __builtin_IMCE_GET_QREG(1);
      var11 = __builtin_IMCE_GET_QREG(2);
      var12 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      // endgenerate: imcflow.preop-minmax_block
      __builtin_IMCE_SEND(1, var9, 0, 0); // TensorEdge(((33, 31), odata), (34, data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(1, var10, 0, 0); // TensorEdge(((33, 31), odata), (34, data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(1, var11, 0, 0); // TensorEdge(((33, 31), odata), (34, data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(1, var12, 0, 0); // TensorEdge(((33, 31), odata), (34, data)), imce_1_1 -> imce_2_1
      // endgenerate: imcflow.preop-minmax_wrapper
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
    // generate: TensorEdge((-34, config), (34, config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge((-34, config), (34, config)), config write
    // generate: conv exec2
    // generate: conv exec2_row_group0_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 16; i1++) { // generate : conv exec2_row_group0_outer_loop(iterate row offset)
      // generate: conv exec2_row_group0_col_group0
      // generate : conv exec2_row_group0_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 34; i2++) { // generate : load_block
        __builtin_IMCE_SETFLAG(1);
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge(((33, 31), odata), (34, data)), imce_1_1 -> imce_2_1

        } // endgenerate
        __builtin_IMCE_SETFLAG(0);
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var1 = __builtin_IMCE_GET_CREG((short)0);
      var2 = __builtin_IMCE_GET_CREG((short)1);
      var3 = __builtin_IMCE_GET_CREG((short)2);
      var4 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(1, var1, 2, 0); // TensorEdge((34, odata), (35, func_out0)), imce_2_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var2, 2, 0); // TensorEdge((34, odata), (35, func_out0)), imce_2_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var3, 2, 0); // TensorEdge((34, odata), (35, func_out0)), imce_2_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var4, 2, 0); // TensorEdge((34, odata), (35, func_out0)), imce_2_1 -> inode_3_0
      // endgenerate : conv exec2_row_group0_col_group0
      // endgenerate: conv exec2_row_group0_col_group0
      // generate: conv exec2_row_group0_col_group1
      for (int i2 = 0; i2 < 15; i2++) { // generate : conv exec2_row_group0_col_group1

        // generate: load_block
        for (int i3 = 0; i3 < 2; i3++) { // generate : load_block
          __builtin_IMCE_SETFLAG(1);
          for (int i4 = 0; i4 < 4; i4++) { // generate
            __builtin_IMCE_LOAD_LB(0); // TensorEdge(((33, 31), odata), (34, data)), imce_1_1 -> imce_2_1

          } // endgenerate
          __builtin_IMCE_SETFLAG(0);
        } // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var1 = __builtin_IMCE_GET_CREG((short)0);
        var2 = __builtin_IMCE_GET_CREG((short)1);
        var3 = __builtin_IMCE_GET_CREG((short)2);
        var4 = __builtin_IMCE_GET_CREG((short)3);
        __builtin_IMCE_SEND(1, var1, 2, 0); // TensorEdge((34, odata), (35, func_out0)), imce_2_1 -> inode_3_0
        __builtin_IMCE_SEND(1, var2, 2, 0); // TensorEdge((34, odata), (35, func_out0)), imce_2_1 -> inode_3_0
        __builtin_IMCE_SEND(1, var3, 2, 0); // TensorEdge((34, odata), (35, func_out0)), imce_2_1 -> inode_3_0
        __builtin_IMCE_SEND(1, var4, 2, 0); // TensorEdge((34, odata), (35, func_out0)), imce_2_1 -> inode_3_0
      } // endgenerate : conv exec2_row_group0_col_group1
      // endgenerate: conv exec2_row_group0_col_group1
    } // endgenerate : conv exec2_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec2_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec2
    __builtin_IMCE_STOP();
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
