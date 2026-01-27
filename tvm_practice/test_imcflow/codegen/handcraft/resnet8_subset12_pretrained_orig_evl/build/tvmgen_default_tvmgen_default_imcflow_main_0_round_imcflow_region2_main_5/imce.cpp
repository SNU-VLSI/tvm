#include "../common_decl.h"
void tvmgen_default_tvmgen_default_imcflow_main_0_round_imcflow_region2_main_5() {
  int hid = __builtin_IMCE_GET_CORE_HID();
  int wid = __builtin_IMCE_GET_CORE_WID();
  short16 var1; // (ConvBlock(gid: 34), 0)
  short16 var2; // (ConvBlock(gid: 34), 1)
  short16 var3; // (ConvBlock(gid: 34), 2)
  short16 var4; // (ConvBlock(gid: 34), 3)
  short16 var5; // (TensorEdge((-33, config), (34, config)), 0)
  short16 var6; // (TensorEdge((33, odata), (34, data)), 0)
  short16 var7; // (TensorEdge((33, odata), (34, data)), 1)
  short16 var8; // (TensorEdge((33, odata), (34, data)), 2)
  short16 var9; // (TensorEdge((33, odata), (34, data)), 3)
  short16 var10; // (TensorEdge((-30, min), (33, min)), 0)
  short16 var11; // (TensorEdge((-31, max), (33, max)), 0)
  short16 var12; // (TensorEdge((32, odata), (33, data)), 0)
  short16 var13; // (MinmaxQuantBlock(gid: 33), 0)
  short16 var14; // (MinmaxQuantBlock(gid: 33), 1)
  short16 var15; // (MinmaxQuantBlock(gid: 33), 2)
  short16 var16; // (MinmaxQuantBlock(gid: 33), 3)
  short16 var17; // (TensorEdge((-28, odata), (32, lhs)), 0)
  short16 var18; // (TensorEdge((-29, odata), (32, rhs)), 0)
  short16 var19; // (AddBlock(gid: 32), 0)
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
    // generate: TensorEdge((-33, config), (34, config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge((-33, config), (34, config)), config write
    // generate: conv exec2
    // generate: conv exec2_row_group0_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 16; i1++) { // generate : conv exec2_row_group0_outer_loop(iterate row offset)
      // generate: conv exec2_row_group0_col_group0
      // generate : conv exec2_row_group0_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 34; i2++) { // generate : load_block
        __builtin_IMCE_SETFLAG(1);
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((33, odata), (34, data)), imce_3_3 -> imce_3_2

        } // endgenerate
        __builtin_IMCE_SETFLAG(0);
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var1 = __builtin_IMCE_GET_CREG((short)0);
      var2 = __builtin_IMCE_GET_CREG((short)1);
      var3 = __builtin_IMCE_GET_CREG((short)2);
      var4 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(1, var1, 2, 0); // TensorEdge((34, odata), (35, func_out0)), imce_3_2 -> inode_3_0
      __builtin_IMCE_SEND(1, var2, 2, 0); // TensorEdge((34, odata), (35, func_out0)), imce_3_2 -> inode_3_0
      __builtin_IMCE_SEND(1, var3, 2, 0); // TensorEdge((34, odata), (35, func_out0)), imce_3_2 -> inode_3_0
      __builtin_IMCE_SEND(1, var4, 2, 0); // TensorEdge((34, odata), (35, func_out0)), imce_3_2 -> inode_3_0
      // endgenerate : conv exec2_row_group0_col_group0
      // endgenerate: conv exec2_row_group0_col_group0
      // generate: conv exec2_row_group0_col_group1
      for (int i2 = 0; i2 < 15; i2++) { // generate : conv exec2_row_group0_col_group1

        // generate: load_block
        for (int i3 = 0; i3 < 2; i3++) { // generate : load_block
          __builtin_IMCE_SETFLAG(1);
          for (int i4 = 0; i4 < 4; i4++) { // generate
            __builtin_IMCE_LOAD_LB(0); // TensorEdge((33, odata), (34, data)), imce_3_3 -> imce_3_2

          } // endgenerate
          __builtin_IMCE_SETFLAG(0);
        } // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var1 = __builtin_IMCE_GET_CREG((short)0);
        var2 = __builtin_IMCE_GET_CREG((short)1);
        var3 = __builtin_IMCE_GET_CREG((short)2);
        var4 = __builtin_IMCE_GET_CREG((short)3);
        __builtin_IMCE_SEND(1, var1, 2, 0); // TensorEdge((34, odata), (35, func_out0)), imce_3_2 -> inode_3_0
        __builtin_IMCE_SEND(1, var2, 2, 0); // TensorEdge((34, odata), (35, func_out0)), imce_3_2 -> inode_3_0
        __builtin_IMCE_SEND(1, var3, 2, 0); // TensorEdge((34, odata), (35, func_out0)), imce_3_2 -> inode_3_0
        __builtin_IMCE_SEND(1, var4, 2, 0); // TensorEdge((34, odata), (35, func_out0)), imce_3_2 -> inode_3_0
      } // endgenerate : conv exec2_row_group0_col_group1
      // endgenerate: conv exec2_row_group0_col_group1
    } // endgenerate : conv exec2_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec2_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec2
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 3) { // imce_3_3
    // generate: TensorEdge((-30, min), (33, min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge((-30, min), (33, min)), min write
    // generate: TensorEdge((-31, max), (33, max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge((-31, max), (33, max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 1024; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      __builtin_IMCE_SETFLAG(2); 
      __builtin_IMCE_STANDBY(19, 2); 
      __builtin_IMCE_SETFLAG(0); 
      var12 = __builtin_IMCE_RECV(2); // TensorEdge((32, odata), (33, data)), imce_3_4 -> imce_3_3
      // generate: min_max_quantize

      __builtin_IMCE_STANDBY(17, 1); 
      __builtin_IMCE_MM_QUANT(var12, 0, 15, 0);
      var13 = __builtin_IMCE_GET_QREG(0);
      var14 = __builtin_IMCE_GET_QREG(1);
      var15 = __builtin_IMCE_GET_QREG(2);
      var16 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      __builtin_IMCE_SEND(1, var13, 0, 0); // TensorEdge((33, odata), (34, data)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(1, var14, 0, 0); // TensorEdge((33, odata), (34, data)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(1, var15, 0, 0); // TensorEdge((33, odata), (34, data)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(1, var16, 0, 0); // TensorEdge((33, odata), (34, data)), imce_3_3 -> imce_3_2
      // endgenerate: min_max_quantize_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 4) { // imce_3_4
    // generate: call_created_loop
    for (int i1 = 0; i1 < 1024; i1++) { // generate : call_created_loop
      // generate: add standalone

      __builtin_IMCE_SETFLAG(1); 
      __builtin_IMCE_STANDBY(0, 1); 
      __builtin_IMCE_STANDBY(5, 1); 
      __builtin_IMCE_SETFLAG(0); 
      var17 = __builtin_IMCE_RECV(2); // TensorEdge((-28, odata), (32, lhs)), inode_0_0 -> imce_3_4
      var18 = __builtin_IMCE_RECV(3); // TensorEdge((-29, odata), (32, rhs)), inode_1_0 -> imce_3_4
      // generate: add

      var19 = __builtin_IMCE_ADD(var17, var18, 15);
      // endgenerate: add

        __builtin_IMCE_STANDBY(18, 2);
        __builtin_IMCE_SETFLAG(2);
        __builtin_IMCE_STANDBY(18, 0);
        __builtin_IMCE_SETFLAG(0);
      __builtin_IMCE_SEND(1, var19, 2, 0); // TensorEdge((32, odata), (33, data)), imce_3_4 -> imce_3_3
      // endgenerate: add standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
}
