#include "../common_decl.h"
void tvmgen_default_tvmgen_default_imcflow_main_0_round_imcflow_region1_main_0() {
  int hid = __builtin_IMCE_GET_CORE_HID();
  int wid = __builtin_IMCE_GET_CORE_WID();
  short16 var1; // (ConvBlock(gid: 14), 0)
  short16 var2; // (ConvBlock(gid: 14), 1)
  short16 var3; // (ConvBlock(gid: 14), 2)
  short16 var4; // (ConvBlock(gid: 14), 3)
  short16 var5; // (ConvBlock(gid: 11), 0)
  short16 var6; // (ConvBlock(gid: 11), 1)
  short16 var7; // (ConvBlock(gid: 11), 2)
  short16 var8; // (ConvBlock(gid: 11), 3)
  short16 var9; // (TensorEdge(((20, -13), config), ((20, 11), config)), 0)
  short16 var10; // (TensorEdge(((20, -14), fused_scale), ((20, 12), fused_scale)), 0)
  short16 var11; // (TensorEdge(((20, -15), fused_bias), ((20, 12), fused_bias)), 0)
  short16 var12; // (BatchNormBlock(gid: 12), 0)
  short16 var13; // (BatchNormBlock(gid: 12), 1)
  short16 var14; // (BatchNormBlock(gid: 12), 2)
  short16 var15; // (BatchNormBlock(gid: 12), 3)
  short16 var16; // (TensorEdge(((20, -14), fused_scale), ((20, 12), fused_scale)), 1)
  short16 var17; // (TensorEdge(((20, -15), fused_bias), ((20, 12), fused_bias)), 1)
  short16 var18; // (TensorEdge(((20, -14), fused_scale), ((20, 12), fused_scale)), 2)
  short16 var19; // (TensorEdge(((20, -15), fused_bias), ((20, 12), fused_bias)), 2)
  short16 var20; // (TensorEdge(((20, -14), fused_scale), ((20, 12), fused_scale)), 3)
  short16 var21; // (TensorEdge(((20, -15), fused_bias), ((20, 12), fused_bias)), 3)
  short16 var22; // (TensorEdge((-23, min), (19, min)), 0)
  short16 var23; // (TensorEdge((-24, max), (19, max)), 0)
  short16 var24; // (TensorEdge(((18, 15), odata), (19, data)), 0)
  short16 var25; // (TensorEdge(((18, 15), odata), (19, data)), 1)
  short16 var26; // (TensorEdge(((18, 15), odata), (19, data)), 2)
  short16 var27; // (TensorEdge(((18, 15), odata), (19, data)), 3)
  short16 var28; // (MinmaxQuantBlock(gid: 19), 0)
  short16 var29; // (MinmaxQuantBlock(gid: 19), 1)
  short16 var30; // (MinmaxQuantBlock(gid: 19), 2)
  short16 var31; // (MinmaxQuantBlock(gid: 19), 3)
  short16 var32; // (TensorEdge(((18, -18), config), ((18, 14), config)), 0)
  short16 var33; // (TensorEdge(((18, -19), fused_scale), ((18, 15), fused_scale)), 0)
  short16 var34; // (TensorEdge(((18, -20), fused_bias), ((18, 15), fused_bias)), 0)
  short16 var35; // (BatchNormBlock(gid: 15), 0)
  short16 var36; // (BatchNormBlock(gid: 15), 1)
  short16 var37; // (BatchNormBlock(gid: 15), 2)
  short16 var38; // (BatchNormBlock(gid: 15), 3)
  short16 var39; // (TensorEdge(((18, -19), fused_scale), ((18, 15), fused_scale)), 1)
  short16 var40; // (TensorEdge(((18, -20), fused_bias), ((18, 15), fused_bias)), 1)
  short16 var41; // (TensorEdge(((18, -19), fused_scale), ((18, 15), fused_scale)), 2)
  short16 var42; // (TensorEdge(((18, -20), fused_bias), ((18, 15), fused_bias)), 2)
  short16 var43; // (TensorEdge(((18, -19), fused_scale), ((18, 15), fused_scale)), 3)
  short16 var44; // (TensorEdge(((18, -20), fused_bias), ((18, 15), fused_bias)), 3)
  short16 var45; // (TensorEdge((-21, min), (17, min)), 0)
  short16 var46; // (TensorEdge((-22, max), (17, max)), 0)
  short16 var47; // (TensorEdge((-10, odata), (17, data)), 0)
  short16 var48; // (MinmaxQuantBlock(gid: 17), 0)
  short16 var49; // (MinmaxQuantBlock(gid: 17), 1)
  short16 var50; // (MinmaxQuantBlock(gid: 17), 2)
  short16 var51; // (MinmaxQuantBlock(gid: 17), 3)
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
    // generate: TensorEdge(((20, -13), config), ((20, 11), config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge(((20, -13), config), ((20, 11), config)), config write
    // generate: TensorEdge(((20, -14), fused_scale), ((20, 12), fused_scale)), fused_scale write

    var10 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((20, -14), fused_scale), ((20, 12), fused_scale)), fused_scale write
    // generate: TensorEdge(((20, -15), fused_bias), ((20, 12), fused_bias)), fused_bias write

    var11 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((20, -15), fused_bias), ((20, 12), fused_bias)), fused_bias write
    // generate: conv exec1
    // generate: conv exec1_row_group0_outer_loop(iterate row offset)
    // generate : conv exec1_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec1_row_group0_col_group0
    // generate : conv exec1_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 10; i1++) { // generate : load_block
      __builtin_IMCE_SETFLAG(1);
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((19, odata), ((20, 11), data)), imce_3_2 -> imce_3_1
      } // endgenerate
      __builtin_IMCE_SETFLAG(0);
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var5 = __builtin_IMCE_GET_CREG((short)0);
    var6 = __builtin_IMCE_GET_CREG((short)1);
    var7 = __builtin_IMCE_GET_CREG((short)2);
    var8 = __builtin_IMCE_GET_CREG((short)3);
    // generate: batch_norm

    var12 = __builtin_IMCE_MULTL(var5, var10, 15);
    var12 = __builtin_IMCE_ADD(var12, var11, 15);
    var13 = __builtin_IMCE_MULTL(var6, var16, 15);
    var13 = __builtin_IMCE_ADD(var13, var17, 15);
    var14 = __builtin_IMCE_MULTL(var7, var18, 15);
    var14 = __builtin_IMCE_ADD(var14, var19, 15);
    var15 = __builtin_IMCE_MULTL(var8, var20, 15);
    var15 = __builtin_IMCE_ADD(var15, var21, 15);
    // endgenerate: batch_norm
    __builtin_IMCE_SEND(1, var12, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var13, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var14, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var15, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
    // endgenerate : conv exec1_row_group0_col_group0
    // endgenerate: conv exec1_row_group0_col_group0
    // generate: conv exec1_row_group0_col_group1
    for (int i1 = 0; i1 < 6; i1++) { // generate : conv exec1_row_group0_col_group1

      // generate: load_block
      // generate : load_block. loop count == 1
      __builtin_IMCE_SETFLAG(1);
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((19, odata), ((20, 11), data)), imce_3_2 -> imce_3_1
      } // endgenerate
      __builtin_IMCE_SETFLAG(0);
      // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      // generate: batch_norm

      var12 = __builtin_IMCE_MULTL(var5, var10, 15);
      var12 = __builtin_IMCE_ADD(var12, var11, 15);
      var13 = __builtin_IMCE_MULTL(var6, var16, 15);
      var13 = __builtin_IMCE_ADD(var13, var17, 15);
      var14 = __builtin_IMCE_MULTL(var7, var18, 15);
      var14 = __builtin_IMCE_ADD(var14, var19, 15);
      var15 = __builtin_IMCE_MULTL(var8, var20, 15);
      var15 = __builtin_IMCE_ADD(var15, var21, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var12, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var13, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var14, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var15, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
    } // endgenerate : conv exec1_row_group0_col_group1
    // endgenerate: conv exec1_row_group0_col_group1
    // generate: conv exec1_row_group0_col_group2
    // generate : conv exec1_row_group0_col_group2. loop count == 1

    // generate: load_block
    // loop ignored with loop count == 0 : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var5 = __builtin_IMCE_GET_CREG((short)0);
    var6 = __builtin_IMCE_GET_CREG((short)1);
    var7 = __builtin_IMCE_GET_CREG((short)2);
    var8 = __builtin_IMCE_GET_CREG((short)3);
    // generate: batch_norm

    var12 = __builtin_IMCE_MULTL(var5, var10, 15);
    var12 = __builtin_IMCE_ADD(var12, var11, 15);
    var13 = __builtin_IMCE_MULTL(var6, var16, 15);
    var13 = __builtin_IMCE_ADD(var13, var17, 15);
    var14 = __builtin_IMCE_MULTL(var7, var18, 15);
    var14 = __builtin_IMCE_ADD(var14, var19, 15);
    var15 = __builtin_IMCE_MULTL(var8, var20, 15);
    var15 = __builtin_IMCE_ADD(var15, var21, 15);
    // endgenerate: batch_norm
    __builtin_IMCE_SEND(1, var12, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var13, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var14, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var15, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
    // endgenerate : conv exec1_row_group0_col_group2
    // endgenerate: conv exec1_row_group0_col_group2
    // endgenerate : conv exec1_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec1_row_group0_outer_loop(iterate row offset)
    // generate: conv exec1_row_group1_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 6; i1++) { // generate : conv exec1_row_group1_outer_loop(iterate row offset)
      // generate: conv exec1_row_group1_col_group0
      // generate : conv exec1_row_group1_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 2; i2++) { // generate : load_block
        __builtin_IMCE_SETFLAG(1);
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((19, odata), ((20, 11), data)), imce_3_2 -> imce_3_1
        } // endgenerate
        __builtin_IMCE_SETFLAG(0);
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      // generate: batch_norm

      var12 = __builtin_IMCE_MULTL(var5, var10, 15);
      var12 = __builtin_IMCE_ADD(var12, var11, 15);
      var13 = __builtin_IMCE_MULTL(var6, var16, 15);
      var13 = __builtin_IMCE_ADD(var13, var17, 15);
      var14 = __builtin_IMCE_MULTL(var7, var18, 15);
      var14 = __builtin_IMCE_ADD(var14, var19, 15);
      var15 = __builtin_IMCE_MULTL(var8, var20, 15);
      var15 = __builtin_IMCE_ADD(var15, var21, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var12, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var13, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var14, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var15, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
      // endgenerate : conv exec1_row_group1_col_group0
      // endgenerate: conv exec1_row_group1_col_group0
      // generate: conv exec1_row_group1_col_group1
      for (int i2 = 0; i2 < 6; i2++) { // generate : conv exec1_row_group1_col_group1

        // generate: load_block
        // generate : load_block. loop count == 1
        __builtin_IMCE_SETFLAG(1);
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((19, odata), ((20, 11), data)), imce_3_2 -> imce_3_1
        } // endgenerate
        __builtin_IMCE_SETFLAG(0);
        // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var5 = __builtin_IMCE_GET_CREG((short)0);
        var6 = __builtin_IMCE_GET_CREG((short)1);
        var7 = __builtin_IMCE_GET_CREG((short)2);
        var8 = __builtin_IMCE_GET_CREG((short)3);
        // generate: batch_norm

        var12 = __builtin_IMCE_MULTL(var5, var10, 15);
        var12 = __builtin_IMCE_ADD(var12, var11, 15);
        var13 = __builtin_IMCE_MULTL(var6, var16, 15);
        var13 = __builtin_IMCE_ADD(var13, var17, 15);
        var14 = __builtin_IMCE_MULTL(var7, var18, 15);
        var14 = __builtin_IMCE_ADD(var14, var19, 15);
        var15 = __builtin_IMCE_MULTL(var8, var20, 15);
        var15 = __builtin_IMCE_ADD(var15, var21, 15);
        // endgenerate: batch_norm
        __builtin_IMCE_SEND(1, var12, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
        __builtin_IMCE_SEND(1, var13, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
        __builtin_IMCE_SEND(1, var14, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
        __builtin_IMCE_SEND(1, var15, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
      } // endgenerate : conv exec1_row_group1_col_group1
      // endgenerate: conv exec1_row_group1_col_group1
      // generate: conv exec1_row_group1_col_group2
      // generate : conv exec1_row_group1_col_group2. loop count == 1

      // generate: load_block
      // loop ignored with loop count == 0 : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      // generate: batch_norm

      var12 = __builtin_IMCE_MULTL(var5, var10, 15);
      var12 = __builtin_IMCE_ADD(var12, var11, 15);
      var13 = __builtin_IMCE_MULTL(var6, var16, 15);
      var13 = __builtin_IMCE_ADD(var13, var17, 15);
      var14 = __builtin_IMCE_MULTL(var7, var18, 15);
      var14 = __builtin_IMCE_ADD(var14, var19, 15);
      var15 = __builtin_IMCE_MULTL(var8, var20, 15);
      var15 = __builtin_IMCE_ADD(var15, var21, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var12, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var13, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var14, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var15, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
      // endgenerate : conv exec1_row_group1_col_group2
      // endgenerate: conv exec1_row_group1_col_group2
    } // endgenerate : conv exec1_row_group1_outer_loop(iterate row offset)
    // endgenerate: conv exec1_row_group1_outer_loop(iterate row offset)
    // generate: conv exec1_row_group2_outer_loop(iterate row offset)
    // generate : conv exec1_row_group2_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec1_row_group2_col_group0
    for (int i1 = 0; i1 < 8; i1++) { // generate : conv exec1_row_group2_col_group0

      // generate: load_block
      // loop ignored with loop count == 0 : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      // generate: batch_norm

      var12 = __builtin_IMCE_MULTL(var5, var10, 15);
      var12 = __builtin_IMCE_ADD(var12, var11, 15);
      var13 = __builtin_IMCE_MULTL(var6, var16, 15);
      var13 = __builtin_IMCE_ADD(var13, var17, 15);
      var14 = __builtin_IMCE_MULTL(var7, var18, 15);
      var14 = __builtin_IMCE_ADD(var14, var19, 15);
      var15 = __builtin_IMCE_MULTL(var8, var20, 15);
      var15 = __builtin_IMCE_ADD(var15, var21, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var12, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var13, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var14, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var15, 2, 0); // TensorEdge(((20, 12), odata), (21, func_out0)), imce_3_1 -> inode_3_0
    } // endgenerate : conv exec1_row_group2_col_group0
    // endgenerate: conv exec1_row_group2_col_group0
    // endgenerate : conv exec1_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec1_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec1
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 2) { // imce_3_2
    // generate: TensorEdge((-23, min), (19, min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge((-23, min), (19, min)), min write
    // generate: TensorEdge((-24, max), (19, max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge((-24, max), (19, max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 64; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      __builtin_IMCE_SETFLAG(1);
      var24 = __builtin_IMCE_RECV(2); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
      var25 = __builtin_IMCE_RECV(2); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
      var26 = __builtin_IMCE_RECV(2); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
      var27 = __builtin_IMCE_RECV(2); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SETFLAG(0);
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var24, 0, 15, 0);
      var28 = __builtin_IMCE_GET_QREG(0);
      var29 = __builtin_IMCE_GET_QREG(1);
      var30 = __builtin_IMCE_GET_QREG(2);
      var31 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      __builtin_IMCE_STANDBY(16, 1);
      __builtin_IMCE_SEND(1, var28, 0, 0); // TensorEdge((19, odata), ((20, 11), data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var29, 0, 0); // TensorEdge((19, odata), ((20, 11), data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var30, 0, 0); // TensorEdge((19, odata), ((20, 11), data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var31, 0, 0); // TensorEdge((19, odata), ((20, 11), data)), imce_3_2 -> imce_3_1
      // endgenerate: min_max_quantize_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 3) { // imce_3_3
    // generate: TensorEdge(((18, -18), config), ((18, 14), config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge(((18, -18), config), ((18, 14), config)), config write
    // generate: TensorEdge(((18, -19), fused_scale), ((18, 15), fused_scale)), fused_scale write

    var33 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((18, -19), fused_scale), ((18, 15), fused_scale)), fused_scale write
    // generate: TensorEdge(((18, -20), fused_bias), ((18, 15), fused_bias)), fused_bias write

    var34 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((18, -20), fused_bias), ((18, 15), fused_bias)), fused_bias write
    // generate: conv exec0
    // generate: conv exec0_row_group0_outer_loop(iterate row offset)
    // generate : conv exec0_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec0_row_group0_col_group0
    // generate : conv exec0_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 10; i1++) { // generate : load_block
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((17, odata), ((18, 14), data)), imce_3_4 -> imce_3_3

      } // endgenerate
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var1 = __builtin_IMCE_GET_CREG((short)0);
    var2 = __builtin_IMCE_GET_CREG((short)1);
    var3 = __builtin_IMCE_GET_CREG((short)2);
    var4 = __builtin_IMCE_GET_CREG((short)3);
    // generate: batch_norm

    var35 = __builtin_IMCE_MULTL(var1, var33, 15);
    var35 = __builtin_IMCE_ADD(var35, var34, 15);
    var36 = __builtin_IMCE_MULTL(var2, var39, 15);
    var36 = __builtin_IMCE_ADD(var36, var40, 15);
    var37 = __builtin_IMCE_MULTL(var3, var41, 15);
    var37 = __builtin_IMCE_ADD(var37, var42, 15);
    var38 = __builtin_IMCE_MULTL(var4, var43, 15);
    var38 = __builtin_IMCE_ADD(var38, var44, 15);
    // endgenerate: batch_norm
    __builtin_IMCE_STANDBY(17, 1);
    __builtin_IMCE_SEND(1, var35, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
    __builtin_IMCE_SEND(1, var36, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
    __builtin_IMCE_SEND(1, var37, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
    __builtin_IMCE_SEND(1, var38, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
    // endgenerate : conv exec0_row_group0_col_group0
    // endgenerate: conv exec0_row_group0_col_group0
    // generate: conv exec0_row_group0_col_group1
    for (int i1 = 0; i1 < 6; i1++) { // generate : conv exec0_row_group0_col_group1

      // generate: load_block
      // generate : load_block. loop count == 1
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((17, odata), ((18, 14), data)), imce_3_4 -> imce_3_3

      } // endgenerate
      // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var1 = __builtin_IMCE_GET_CREG((short)0);
      var2 = __builtin_IMCE_GET_CREG((short)1);
      var3 = __builtin_IMCE_GET_CREG((short)2);
      var4 = __builtin_IMCE_GET_CREG((short)3);
      // generate: batch_norm

      var35 = __builtin_IMCE_MULTL(var1, var33, 15);
      var35 = __builtin_IMCE_ADD(var35, var34, 15);
      var36 = __builtin_IMCE_MULTL(var2, var39, 15);
      var36 = __builtin_IMCE_ADD(var36, var40, 15);
      var37 = __builtin_IMCE_MULTL(var3, var41, 15);
      var37 = __builtin_IMCE_ADD(var37, var42, 15);
      var38 = __builtin_IMCE_MULTL(var4, var43, 15);
      var38 = __builtin_IMCE_ADD(var38, var44, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_STANDBY(17, 1);
      __builtin_IMCE_SEND(1, var35, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(1, var36, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(1, var37, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(1, var38, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
    } // endgenerate : conv exec0_row_group0_col_group1
    // endgenerate: conv exec0_row_group0_col_group1
    // generate: conv exec0_row_group0_col_group2
    // generate : conv exec0_row_group0_col_group2. loop count == 1

    // generate: load_block
    // loop ignored with loop count == 0 : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var1 = __builtin_IMCE_GET_CREG((short)0);
    var2 = __builtin_IMCE_GET_CREG((short)1);
    var3 = __builtin_IMCE_GET_CREG((short)2);
    var4 = __builtin_IMCE_GET_CREG((short)3);
    // generate: batch_norm

    var35 = __builtin_IMCE_MULTL(var1, var33, 15);
    var35 = __builtin_IMCE_ADD(var35, var34, 15);
    var36 = __builtin_IMCE_MULTL(var2, var39, 15);
    var36 = __builtin_IMCE_ADD(var36, var40, 15);
    var37 = __builtin_IMCE_MULTL(var3, var41, 15);
    var37 = __builtin_IMCE_ADD(var37, var42, 15);
    var38 = __builtin_IMCE_MULTL(var4, var43, 15);
    var38 = __builtin_IMCE_ADD(var38, var44, 15);
    // endgenerate: batch_norm
    __builtin_IMCE_STANDBY(17, 1);
    __builtin_IMCE_SEND(1, var35, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
    __builtin_IMCE_SEND(1, var36, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
    __builtin_IMCE_SEND(1, var37, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
    __builtin_IMCE_SEND(1, var38, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
    // endgenerate : conv exec0_row_group0_col_group2
    // endgenerate: conv exec0_row_group0_col_group2
    // endgenerate : conv exec0_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec0_row_group0_outer_loop(iterate row offset)
    // generate: conv exec0_row_group1_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 6; i1++) { // generate : conv exec0_row_group1_outer_loop(iterate row offset)
      // generate: conv exec0_row_group1_col_group0
      // generate : conv exec0_row_group1_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 2; i2++) { // generate : load_block
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((17, odata), ((18, 14), data)), imce_3_4 -> imce_3_3

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var1 = __builtin_IMCE_GET_CREG((short)0);
      var2 = __builtin_IMCE_GET_CREG((short)1);
      var3 = __builtin_IMCE_GET_CREG((short)2);
      var4 = __builtin_IMCE_GET_CREG((short)3);
      // generate: batch_norm

      var35 = __builtin_IMCE_MULTL(var1, var33, 15);
      var35 = __builtin_IMCE_ADD(var35, var34, 15);
      var36 = __builtin_IMCE_MULTL(var2, var39, 15);
      var36 = __builtin_IMCE_ADD(var36, var40, 15);
      var37 = __builtin_IMCE_MULTL(var3, var41, 15);
      var37 = __builtin_IMCE_ADD(var37, var42, 15);
      var38 = __builtin_IMCE_MULTL(var4, var43, 15);
      var38 = __builtin_IMCE_ADD(var38, var44, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_STANDBY(17, 1);
      __builtin_IMCE_SEND(1, var35, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(1, var36, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(1, var37, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(1, var38, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
      // endgenerate : conv exec0_row_group1_col_group0
      // endgenerate: conv exec0_row_group1_col_group0
      // generate: conv exec0_row_group1_col_group1
      for (int i2 = 0; i2 < 6; i2++) { // generate : conv exec0_row_group1_col_group1

        // generate: load_block
        // generate : load_block. loop count == 1
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((17, odata), ((18, 14), data)), imce_3_4 -> imce_3_3

        } // endgenerate
        // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var1 = __builtin_IMCE_GET_CREG((short)0);
        var2 = __builtin_IMCE_GET_CREG((short)1);
        var3 = __builtin_IMCE_GET_CREG((short)2);
        var4 = __builtin_IMCE_GET_CREG((short)3);
        // generate: batch_norm

        var35 = __builtin_IMCE_MULTL(var1, var33, 15);
        var35 = __builtin_IMCE_ADD(var35, var34, 15);
        var36 = __builtin_IMCE_MULTL(var2, var39, 15);
        var36 = __builtin_IMCE_ADD(var36, var40, 15);
        var37 = __builtin_IMCE_MULTL(var3, var41, 15);
        var37 = __builtin_IMCE_ADD(var37, var42, 15);
        var38 = __builtin_IMCE_MULTL(var4, var43, 15);
        var38 = __builtin_IMCE_ADD(var38, var44, 15);
        // endgenerate: batch_norm
        __builtin_IMCE_STANDBY(17, 1);
        __builtin_IMCE_SEND(1, var35, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
        __builtin_IMCE_SEND(1, var36, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
        __builtin_IMCE_SEND(1, var37, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
        __builtin_IMCE_SEND(1, var38, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
      } // endgenerate : conv exec0_row_group1_col_group1
      // endgenerate: conv exec0_row_group1_col_group1
      // generate: conv exec0_row_group1_col_group2
      // generate : conv exec0_row_group1_col_group2. loop count == 1

      // generate: load_block
      // loop ignored with loop count == 0 : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var1 = __builtin_IMCE_GET_CREG((short)0);
      var2 = __builtin_IMCE_GET_CREG((short)1);
      var3 = __builtin_IMCE_GET_CREG((short)2);
      var4 = __builtin_IMCE_GET_CREG((short)3);
      // generate: batch_norm

      var35 = __builtin_IMCE_MULTL(var1, var33, 15);
      var35 = __builtin_IMCE_ADD(var35, var34, 15);
      var36 = __builtin_IMCE_MULTL(var2, var39, 15);
      var36 = __builtin_IMCE_ADD(var36, var40, 15);
      var37 = __builtin_IMCE_MULTL(var3, var41, 15);
      var37 = __builtin_IMCE_ADD(var37, var42, 15);
      var38 = __builtin_IMCE_MULTL(var4, var43, 15);
      var38 = __builtin_IMCE_ADD(var38, var44, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_STANDBY(17, 1);
      __builtin_IMCE_SEND(1, var35, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(1, var36, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(1, var37, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(1, var38, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
      // endgenerate : conv exec0_row_group1_col_group2
      // endgenerate: conv exec0_row_group1_col_group2
    } // endgenerate : conv exec0_row_group1_outer_loop(iterate row offset)
    // endgenerate: conv exec0_row_group1_outer_loop(iterate row offset)
    // generate: conv exec0_row_group2_outer_loop(iterate row offset)
    // generate : conv exec0_row_group2_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec0_row_group2_col_group0

    for (int i1 = 0; i1 < 4; i1++) { // generate : conv exec0_row_group2_col_group0
      __builtin_IMCE_STEP();

      var1 = __builtin_IMCE_GET_CREG((short)0);
      var2 = __builtin_IMCE_GET_CREG((short)1);
      var3 = __builtin_IMCE_GET_CREG((short)2);
      var4 = __builtin_IMCE_GET_CREG((short)3);
      // generate: batch_norm

      var35 = __builtin_IMCE_MULTL(var1, var33, 15);
      var35 = __builtin_IMCE_ADD(var35, var34, 15);
      var36 = __builtin_IMCE_MULTL(var2, var39, 15);
      var36 = __builtin_IMCE_ADD(var36, var40, 15);
      var37 = __builtin_IMCE_MULTL(var3, var41, 15);
      var37 = __builtin_IMCE_ADD(var37, var42, 15);
      var38 = __builtin_IMCE_MULTL(var4, var43, 15);
      var38 = __builtin_IMCE_ADD(var38, var44, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var35, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(1, var36, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(1, var37, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(1, var38, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
    }

    for (int i1 = 0; i1 < 4; i1++) { // generate : conv exec0_row_group2_col_group0

      // generate: load_block
      // loop ignored with loop count == 0 : load_block
      // endgenerate: load_block

      __builtin_IMCE_NOP();
      __builtin_IMCE_NOP();
      __builtin_IMCE_NOP();
      __builtin_IMCE_NOP();
      __builtin_IMCE_NOP();
      __builtin_IMCE_NOP();
      __builtin_IMCE_NOP();
      __builtin_IMCE_NOP();
      __builtin_IMCE_NOP();
      __builtin_IMCE_NOP();
      __builtin_IMCE_NOP();
      __builtin_IMCE_NOP();
      __builtin_IMCE_NOP();
      __builtin_IMCE_NOP();
      __builtin_IMCE_NOP();
      __builtin_IMCE_NOP();
      __builtin_IMCE_NOP();
      __builtin_IMCE_NOP();

      var1 = __builtin_IMCE_GET_CREG((short)0);
      var2 = __builtin_IMCE_GET_CREG((short)1);
      var3 = __builtin_IMCE_GET_CREG((short)2);
      var4 = __builtin_IMCE_GET_CREG((short)3);
      // generate: batch_norm

      var35 = __builtin_IMCE_MULTL(var1, var33, 15);
      var35 = __builtin_IMCE_ADD(var35, var34, 15);
      var36 = __builtin_IMCE_MULTL(var2, var39, 15);
      var36 = __builtin_IMCE_ADD(var36, var40, 15);
      var37 = __builtin_IMCE_MULTL(var3, var41, 15);
      var37 = __builtin_IMCE_ADD(var37, var42, 15);
      var38 = __builtin_IMCE_MULTL(var4, var43, 15);
      var38 = __builtin_IMCE_ADD(var38, var44, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var35, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(1, var36, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(1, var37, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(1, var38, 2, 0); // TensorEdge(((18, 15), odata), (19, data)), imce_3_3 -> imce_3_2
    } // endgenerate : conv exec0_row_group2_col_group0
    // endgenerate: conv exec0_row_group2_col_group0
    // endgenerate : conv exec0_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec0_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec0
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 4) { // imce_3_4
    // generate: TensorEdge((-21, min), (17, min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge((-21, min), (17, min)), min write
    // generate: TensorEdge((-22, max), (17, max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge((-22, max), (17, max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 64; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      var47 = __builtin_IMCE_RECV(2); // TensorEdge((-10, odata), (17, data)), inode_0_0 -> imce_3_4
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var47, 0, 15, 0);
      var48 = __builtin_IMCE_GET_QREG(0);
      var49 = __builtin_IMCE_GET_QREG(1);
      var50 = __builtin_IMCE_GET_QREG(2);
      var51 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      __builtin_IMCE_SEND(1, var48, 0, 0); // TensorEdge((17, odata), ((18, 14), data)), imce_3_4 -> imce_3_3
      __builtin_IMCE_SEND(1, var49, 0, 0); // TensorEdge((17, odata), ((18, 14), data)), imce_3_4 -> imce_3_3
      __builtin_IMCE_SEND(1, var50, 0, 0); // TensorEdge((17, odata), ((18, 14), data)), imce_3_4 -> imce_3_3
      __builtin_IMCE_SEND(1, var51, 0, 0); // TensorEdge((17, odata), ((18, 14), data)), imce_3_4 -> imce_3_3
      // endgenerate: min_max_quantize_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
}
