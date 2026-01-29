#include "../common_decl.h"
void tvmgen_default_tvmgen_default_imcflow_main_46_round_imcflow_region1_main_0() {
  int hid = __builtin_IMCE_GET_CORE_HID();
  int wid = __builtin_IMCE_GET_CORE_WID();
  short16 var1; // (ConvBlock(gid: 35), 0)
  short16 var2; // (ConvBlock(gid: 35), 1)
  short16 var3; // (ConvBlock(gid: 35), 2)
  short16 var4; // (ConvBlock(gid: 35), 3)
  short16 var5; // (ConvBlock(gid: 32), 0)
  short16 var6; // (ConvBlock(gid: 32), 1)
  short16 var7; // (ConvBlock(gid: 32), 2)
  short16 var8; // (ConvBlock(gid: 32), 3)
  short16 var9; // (TensorEdge((-24, min), (38, min)), 0)
  short16 var10; // (TensorEdge((-25, max), (38, max)), 0)
  short16 var11; // (TensorEdge((-13, odata), (38, data)), 0)
  short16 var12; // (MinmaxQuantBlock(gid: 38), 0)
  short16 var13; // (MinmaxQuantBlock(gid: 38), 1)
  short16 var14; // (MinmaxQuantBlock(gid: 38), 2)
  short16 var15; // (MinmaxQuantBlock(gid: 38), 3)
  short16 var16; // (TensorEdge(((39, -21), config), ((39, 35), config)), 0)
  short16 var17; // (TensorEdge(((39, -22), fused_scale), ((39, 36), fused_scale)), 0)
  short16 var18; // (TensorEdge(((39, -23), fused_bias), ((39, 36), fused_bias)), 0)
  short16 var19; // (BatchNormBlock(gid: 36), 0)
  short16 var20; // (BatchNormBlock(gid: 36), 1)
  short16 var21; // (BatchNormBlock(gid: 36), 2)
  short16 var22; // (BatchNormBlock(gid: 36), 3)
  short16 var23; // (TensorEdge(((39, -22), fused_scale), ((39, 36), fused_scale)), 1)
  short16 var24; // (TensorEdge(((39, -23), fused_bias), ((39, 36), fused_bias)), 1)
  short16 var25; // (TensorEdge(((39, -22), fused_scale), ((39, 36), fused_scale)), 2)
  short16 var26; // (TensorEdge(((39, -23), fused_bias), ((39, 36), fused_bias)), 2)
  short16 var27; // (TensorEdge(((39, -22), fused_scale), ((39, 36), fused_scale)), 3)
  short16 var28; // (TensorEdge(((39, -23), fused_bias), ((39, 36), fused_bias)), 3)
  short16 var29; // (TensorEdge((-26, min), (40, min)), 0)
  short16 var30; // (TensorEdge((-27, max), (40, max)), 0)
  short16 var31; // (TensorEdge(((39, 36), odata), (40, data)), 0)
  short16 var32; // (TensorEdge(((39, 36), odata), (40, data)), 1)
  short16 var33; // (TensorEdge(((39, 36), odata), (40, data)), 2)
  short16 var34; // (TensorEdge(((39, 36), odata), (40, data)), 3)
  short16 var35; // (MinmaxQuantBlock(gid: 40), 0)
  short16 var36; // (MinmaxQuantBlock(gid: 40), 1)
  short16 var37; // (MinmaxQuantBlock(gid: 40), 2)
  short16 var38; // (MinmaxQuantBlock(gid: 40), 3)
  short16 var39; // (TensorEdge((-28, odata), (42, rhs)), 0)
  short16 var40; // (TensorEdge((-13, odata), (42, lhs)), 0)
  short16 var41; // (MultlBlock(gid: 42), 0)
  short16 var42; // (TensorEdge(((41, -16), config), ((41, 32), config)), 0)
  short16 var43; // (TensorEdge(((41, -17), fused_scale), ((41, 33), fused_scale)), 0)
  short16 var44; // (TensorEdge(((41, -18), fused_bias), ((41, 33), fused_bias)), 0)
  short16 var45; // (BatchNormBlock(gid: 33), 0)
  short16 var46; // (BatchNormBlock(gid: 33), 1)
  short16 var47; // (BatchNormBlock(gid: 33), 2)
  short16 var48; // (BatchNormBlock(gid: 33), 3)
  short16 var49; // (TensorEdge(((41, -17), fused_scale), ((41, 33), fused_scale)), 1)
  short16 var50; // (TensorEdge(((41, -18), fused_bias), ((41, 33), fused_bias)), 1)
  short16 var51; // (TensorEdge(((41, -17), fused_scale), ((41, 33), fused_scale)), 2)
  short16 var52; // (TensorEdge(((41, -18), fused_bias), ((41, 33), fused_bias)), 2)
  short16 var53; // (TensorEdge(((41, -17), fused_scale), ((41, 33), fused_scale)), 3)
  short16 var54; // (TensorEdge(((41, -18), fused_bias), ((41, 33), fused_bias)), 3)
  if (hid == 0 && wid == 1) { // imce_0_1
    // generate: TensorEdge((-24, min), (38, min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge((-24, min), (38, min)), min write
    // generate: TensorEdge((-25, max), (38, max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge((-25, max), (38, max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 1024; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      var11 = __builtin_IMCE_RECV(2); // TensorEdge((-13, odata), (38, data)), inode_0_0 -> imce_0_1
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var11, 0, 15, 0);
      var12 = __builtin_IMCE_GET_QREG(0);
      var13 = __builtin_IMCE_GET_QREG(1);
      var14 = __builtin_IMCE_GET_QREG(2);
      var15 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      __builtin_IMCE_SEND(1, var12, 0, 0); // TensorEdge((38, odata), ((39, 35), data)), imce_0_1 -> imce_1_1
      __builtin_IMCE_SEND(1, var13, 0, 0); // TensorEdge((38, odata), ((39, 35), data)), imce_0_1 -> imce_1_1
      __builtin_IMCE_SEND(1, var14, 0, 0); // TensorEdge((38, odata), ((39, 35), data)), imce_0_1 -> imce_1_1
      __builtin_IMCE_SEND(1, var15, 0, 0); // TensorEdge((38, odata), ((39, 35), data)), imce_0_1 -> imce_1_1
      // endgenerate: min_max_quantize_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 0 && wid == 2) { // imce_0_2
  }
  else if (hid == 0 && wid == 3) { // imce_0_3
  }
  else if (hid == 0 && wid == 4) { // imce_0_4
  }
  else if (hid == 1 && wid == 1) { // imce_1_1
    // generate: TensorEdge(((39, -21), config), ((39, 35), config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge(((39, -21), config), ((39, 35), config)), config write
    // generate: TensorEdge(((39, -22), fused_scale), ((39, 36), fused_scale)), fused_scale write

    var17 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((39, -22), fused_scale), ((39, 36), fused_scale)), fused_scale write
    // generate: TensorEdge(((39, -23), fused_bias), ((39, 36), fused_bias)), fused_bias write

    var18 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((39, -23), fused_bias), ((39, 36), fused_bias)), fused_bias write
    // generate: conv exec0
    // generate: conv exec0_row_group0_outer_loop(iterate row offset)
    // generate : conv exec0_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec0_row_group0_col_group0
    // generate : conv exec0_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 34; i1++) { // generate : load_block
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((38, odata), ((39, 35), data)), imce_0_1 -> imce_1_1

      } // endgenerate
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var1 = __builtin_IMCE_GET_CREG((short)0);
    var2 = __builtin_IMCE_GET_CREG((short)1);
    var3 = __builtin_IMCE_GET_CREG((short)2);
    var4 = __builtin_IMCE_GET_CREG((short)3);
    // generate: batch_norm

    var19 = __builtin_IMCE_MULTL(var1, var17, 15);
    var19 = __builtin_IMCE_ADD(var19, var18, 15);
    var20 = __builtin_IMCE_MULTL(var2, var23, 15);
    var20 = __builtin_IMCE_ADD(var20, var24, 15);
    var21 = __builtin_IMCE_MULTL(var3, var25, 15);
    var21 = __builtin_IMCE_ADD(var21, var26, 15);
    var22 = __builtin_IMCE_MULTL(var4, var27, 15);
    var22 = __builtin_IMCE_ADD(var22, var28, 15);
    // endgenerate: batch_norm
    __builtin_IMCE_SEND(1, var19, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
    __builtin_IMCE_SEND(1, var20, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
    __builtin_IMCE_SEND(1, var21, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
    __builtin_IMCE_SEND(1, var22, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
    // endgenerate : conv exec0_row_group0_col_group0
    // endgenerate: conv exec0_row_group0_col_group0
    // generate: conv exec0_row_group0_col_group1
    for (int i1 = 0; i1 < 30; i1++) { // generate : conv exec0_row_group0_col_group1

      // generate: load_block
      // generate : load_block. loop count == 1
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((38, odata), ((39, 35), data)), imce_0_1 -> imce_1_1

      } // endgenerate
      // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var1 = __builtin_IMCE_GET_CREG((short)0);
      var2 = __builtin_IMCE_GET_CREG((short)1);
      var3 = __builtin_IMCE_GET_CREG((short)2);
      var4 = __builtin_IMCE_GET_CREG((short)3);
      // generate: batch_norm

      var19 = __builtin_IMCE_MULTL(var1, var17, 15);
      var19 = __builtin_IMCE_ADD(var19, var18, 15);
      var20 = __builtin_IMCE_MULTL(var2, var23, 15);
      var20 = __builtin_IMCE_ADD(var20, var24, 15);
      var21 = __builtin_IMCE_MULTL(var3, var25, 15);
      var21 = __builtin_IMCE_ADD(var21, var26, 15);
      var22 = __builtin_IMCE_MULTL(var4, var27, 15);
      var22 = __builtin_IMCE_ADD(var22, var28, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var19, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(1, var20, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(1, var21, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(1, var22, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
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

    var19 = __builtin_IMCE_MULTL(var1, var17, 15);
    var19 = __builtin_IMCE_ADD(var19, var18, 15);
    var20 = __builtin_IMCE_MULTL(var2, var23, 15);
    var20 = __builtin_IMCE_ADD(var20, var24, 15);
    var21 = __builtin_IMCE_MULTL(var3, var25, 15);
    var21 = __builtin_IMCE_ADD(var21, var26, 15);
    var22 = __builtin_IMCE_MULTL(var4, var27, 15);
    var22 = __builtin_IMCE_ADD(var22, var28, 15);
    // endgenerate: batch_norm
    __builtin_IMCE_SEND(1, var19, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
    __builtin_IMCE_SEND(1, var20, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
    __builtin_IMCE_SEND(1, var21, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
    __builtin_IMCE_SEND(1, var22, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
    // endgenerate : conv exec0_row_group0_col_group2
    // endgenerate: conv exec0_row_group0_col_group2
    // endgenerate : conv exec0_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec0_row_group0_outer_loop(iterate row offset)
    // generate: conv exec0_row_group1_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 30; i1++) { // generate : conv exec0_row_group1_outer_loop(iterate row offset)
      // generate: conv exec0_row_group1_col_group0
      // generate : conv exec0_row_group1_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 2; i2++) { // generate : load_block
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((38, odata), ((39, 35), data)), imce_0_1 -> imce_1_1

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var1 = __builtin_IMCE_GET_CREG((short)0);
      var2 = __builtin_IMCE_GET_CREG((short)1);
      var3 = __builtin_IMCE_GET_CREG((short)2);
      var4 = __builtin_IMCE_GET_CREG((short)3);
      // generate: batch_norm

      var19 = __builtin_IMCE_MULTL(var1, var17, 15);
      var19 = __builtin_IMCE_ADD(var19, var18, 15);
      var20 = __builtin_IMCE_MULTL(var2, var23, 15);
      var20 = __builtin_IMCE_ADD(var20, var24, 15);
      var21 = __builtin_IMCE_MULTL(var3, var25, 15);
      var21 = __builtin_IMCE_ADD(var21, var26, 15);
      var22 = __builtin_IMCE_MULTL(var4, var27, 15);
      var22 = __builtin_IMCE_ADD(var22, var28, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var19, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(1, var20, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(1, var21, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(1, var22, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
      // endgenerate : conv exec0_row_group1_col_group0
      // endgenerate: conv exec0_row_group1_col_group0
      // generate: conv exec0_row_group1_col_group1
      for (int i2 = 0; i2 < 30; i2++) { // generate : conv exec0_row_group1_col_group1

        // generate: load_block
        // generate : load_block. loop count == 1
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((38, odata), ((39, 35), data)), imce_0_1 -> imce_1_1

        } // endgenerate
        // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var1 = __builtin_IMCE_GET_CREG((short)0);
        var2 = __builtin_IMCE_GET_CREG((short)1);
        var3 = __builtin_IMCE_GET_CREG((short)2);
        var4 = __builtin_IMCE_GET_CREG((short)3);
        // generate: batch_norm

        var19 = __builtin_IMCE_MULTL(var1, var17, 15);
        var19 = __builtin_IMCE_ADD(var19, var18, 15);
        var20 = __builtin_IMCE_MULTL(var2, var23, 15);
        var20 = __builtin_IMCE_ADD(var20, var24, 15);
        var21 = __builtin_IMCE_MULTL(var3, var25, 15);
        var21 = __builtin_IMCE_ADD(var21, var26, 15);
        var22 = __builtin_IMCE_MULTL(var4, var27, 15);
        var22 = __builtin_IMCE_ADD(var22, var28, 15);
        // endgenerate: batch_norm
        __builtin_IMCE_SEND(1, var19, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
        __builtin_IMCE_SEND(1, var20, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
        __builtin_IMCE_SEND(1, var21, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
        __builtin_IMCE_SEND(1, var22, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
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

      var19 = __builtin_IMCE_MULTL(var1, var17, 15);
      var19 = __builtin_IMCE_ADD(var19, var18, 15);
      var20 = __builtin_IMCE_MULTL(var2, var23, 15);
      var20 = __builtin_IMCE_ADD(var20, var24, 15);
      var21 = __builtin_IMCE_MULTL(var3, var25, 15);
      var21 = __builtin_IMCE_ADD(var21, var26, 15);
      var22 = __builtin_IMCE_MULTL(var4, var27, 15);
      var22 = __builtin_IMCE_ADD(var22, var28, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var19, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(1, var20, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(1, var21, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(1, var22, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
      // endgenerate : conv exec0_row_group1_col_group2
      // endgenerate: conv exec0_row_group1_col_group2
    } // endgenerate : conv exec0_row_group1_outer_loop(iterate row offset)
    // endgenerate: conv exec0_row_group1_outer_loop(iterate row offset)
    // generate: conv exec0_row_group2_outer_loop(iterate row offset)
    // generate : conv exec0_row_group2_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec0_row_group2_col_group0
    for (int i1 = 0; i1 < 32; i1++) { // generate : conv exec0_row_group2_col_group0

      // generate: load_block
      // loop ignored with loop count == 0 : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var1 = __builtin_IMCE_GET_CREG((short)0);
      var2 = __builtin_IMCE_GET_CREG((short)1);
      var3 = __builtin_IMCE_GET_CREG((short)2);
      var4 = __builtin_IMCE_GET_CREG((short)3);
      // generate: batch_norm

      var19 = __builtin_IMCE_MULTL(var1, var17, 15);
      var19 = __builtin_IMCE_ADD(var19, var18, 15);
      var20 = __builtin_IMCE_MULTL(var2, var23, 15);
      var20 = __builtin_IMCE_ADD(var20, var24, 15);
      var21 = __builtin_IMCE_MULTL(var3, var25, 15);
      var21 = __builtin_IMCE_ADD(var21, var26, 15);
      var22 = __builtin_IMCE_MULTL(var4, var27, 15);
      var22 = __builtin_IMCE_ADD(var22, var28, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var19, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(1, var20, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(1, var21, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(1, var22, 2, 0); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
    } // endgenerate : conv exec0_row_group2_col_group0
    // endgenerate: conv exec0_row_group2_col_group0
    // endgenerate : conv exec0_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec0_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec0
    __builtin_IMCE_STOP();
  }
  else if (hid == 1 && wid == 2) { // imce_1_2
  }
  else if (hid == 1 && wid == 3) { // imce_1_3
  }
  else if (hid == 1 && wid == 4) { // imce_1_4
  }
  else if (hid == 2 && wid == 1) { // imce_2_1
    // generate: TensorEdge((-26, min), (40, min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge((-26, min), (40, min)), min write
    // generate: TensorEdge((-27, max), (40, max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge((-27, max), (40, max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 1024; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      var31 = __builtin_IMCE_RECV(2); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
      var32 = __builtin_IMCE_RECV(2); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
      var33 = __builtin_IMCE_RECV(2); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
      var34 = __builtin_IMCE_RECV(2); // TensorEdge(((39, 36), odata), (40, data)), imce_1_1 -> imce_2_1
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var31, 0, 15, 0);
      var35 = __builtin_IMCE_GET_QREG(0);
      var36 = __builtin_IMCE_GET_QREG(1);
      var37 = __builtin_IMCE_GET_QREG(2);
      var38 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      __builtin_IMCE_SEND(2, var35, 0, 0); // TensorEdge((40, odata), ((41, 32), data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(2, var36, 0, 0); // TensorEdge((40, odata), ((41, 32), data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(2, var37, 0, 0); // TensorEdge((40, odata), ((41, 32), data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(2, var38, 0, 0); // TensorEdge((40, odata), ((41, 32), data)), imce_2_1 -> imce_3_1
      // endgenerate: min_max_quantize_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 2 && wid == 2) { // imce_2_2
    // generate: mult const

    var39 = __builtin_IMCE_RECV(3);
    // endgenerate: mult const
    // generate: call_created_loop
    for (int i1 = 0; i1 < 1024; i1++) { // generate : call_created_loop
      // generate: multiply standalone

      var40 = __builtin_IMCE_RECV(2); // TensorEdge((-13, odata), (42, lhs)), inode_0_0 -> imce_2_2
      // generate: multl

      var41 = __builtin_IMCE_MULTL(var40, var39, 15);
      // endgenerate: multl
      __builtin_IMCE_SEND(1, var41, 2, 0); // TensorEdge((42, odata), (43, func_out1), 1), imce_2_2 -> inode_2_0
      // endgenerate: multiply standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 2 && wid == 3) { // imce_2_3
  }
  else if (hid == 2 && wid == 4) { // imce_2_4
  }
  else if (hid == 3 && wid == 1) { // imce_3_1
    // generate: TensorEdge(((41, -16), config), ((41, 32), config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge(((41, -16), config), ((41, 32), config)), config write
    // generate: TensorEdge(((41, -17), fused_scale), ((41, 33), fused_scale)), fused_scale write

    var43 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((41, -17), fused_scale), ((41, 33), fused_scale)), fused_scale write
    // generate: TensorEdge(((41, -18), fused_bias), ((41, 33), fused_bias)), fused_bias write

    var44 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((41, -18), fused_bias), ((41, 33), fused_bias)), fused_bias write
    // generate: conv exec1
    // generate: conv exec1_row_group0_outer_loop(iterate row offset)
    // generate : conv exec1_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec1_row_group0_col_group0
    // generate : conv exec1_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 34; i1++) { // generate : load_block
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((40, odata), ((41, 32), data)), imce_2_1 -> imce_3_1

      } // endgenerate
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var5 = __builtin_IMCE_GET_CREG((short)0);
    var6 = __builtin_IMCE_GET_CREG((short)1);
    var7 = __builtin_IMCE_GET_CREG((short)2);
    var8 = __builtin_IMCE_GET_CREG((short)3);
    // generate: batch_norm

    var45 = __builtin_IMCE_MULTL(var5, var43, 15);
    var45 = __builtin_IMCE_ADD(var45, var44, 15);
    var46 = __builtin_IMCE_MULTL(var6, var49, 15);
    var46 = __builtin_IMCE_ADD(var46, var50, 15);
    var47 = __builtin_IMCE_MULTL(var7, var51, 15);
    var47 = __builtin_IMCE_ADD(var47, var52, 15);
    var48 = __builtin_IMCE_MULTL(var8, var53, 15);
    var48 = __builtin_IMCE_ADD(var48, var54, 15);
    // endgenerate: batch_norm
    __builtin_IMCE_SEND(1, var45, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var46, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var47, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var48, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
    // endgenerate : conv exec1_row_group0_col_group0
    // endgenerate: conv exec1_row_group0_col_group0
    // generate: conv exec1_row_group0_col_group1
    for (int i1 = 0; i1 < 30; i1++) { // generate : conv exec1_row_group0_col_group1

      // generate: load_block
      // generate : load_block. loop count == 1
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((40, odata), ((41, 32), data)), imce_2_1 -> imce_3_1

      } // endgenerate
      // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      // generate: batch_norm

      var45 = __builtin_IMCE_MULTL(var5, var43, 15);
      var45 = __builtin_IMCE_ADD(var45, var44, 15);
      var46 = __builtin_IMCE_MULTL(var6, var49, 15);
      var46 = __builtin_IMCE_ADD(var46, var50, 15);
      var47 = __builtin_IMCE_MULTL(var7, var51, 15);
      var47 = __builtin_IMCE_ADD(var47, var52, 15);
      var48 = __builtin_IMCE_MULTL(var8, var53, 15);
      var48 = __builtin_IMCE_ADD(var48, var54, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var45, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var46, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var47, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var48, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
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

    var45 = __builtin_IMCE_MULTL(var5, var43, 15);
    var45 = __builtin_IMCE_ADD(var45, var44, 15);
    var46 = __builtin_IMCE_MULTL(var6, var49, 15);
    var46 = __builtin_IMCE_ADD(var46, var50, 15);
    var47 = __builtin_IMCE_MULTL(var7, var51, 15);
    var47 = __builtin_IMCE_ADD(var47, var52, 15);
    var48 = __builtin_IMCE_MULTL(var8, var53, 15);
    var48 = __builtin_IMCE_ADD(var48, var54, 15);
    // endgenerate: batch_norm
    __builtin_IMCE_SEND(1, var45, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var46, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var47, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var48, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
    // endgenerate : conv exec1_row_group0_col_group2
    // endgenerate: conv exec1_row_group0_col_group2
    // endgenerate : conv exec1_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec1_row_group0_outer_loop(iterate row offset)
    // generate: conv exec1_row_group1_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 30; i1++) { // generate : conv exec1_row_group1_outer_loop(iterate row offset)
      // generate: conv exec1_row_group1_col_group0
      // generate : conv exec1_row_group1_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 2; i2++) { // generate : load_block
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((40, odata), ((41, 32), data)), imce_2_1 -> imce_3_1

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      // generate: batch_norm

      var45 = __builtin_IMCE_MULTL(var5, var43, 15);
      var45 = __builtin_IMCE_ADD(var45, var44, 15);
      var46 = __builtin_IMCE_MULTL(var6, var49, 15);
      var46 = __builtin_IMCE_ADD(var46, var50, 15);
      var47 = __builtin_IMCE_MULTL(var7, var51, 15);
      var47 = __builtin_IMCE_ADD(var47, var52, 15);
      var48 = __builtin_IMCE_MULTL(var8, var53, 15);
      var48 = __builtin_IMCE_ADD(var48, var54, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var45, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var46, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var47, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var48, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
      // endgenerate : conv exec1_row_group1_col_group0
      // endgenerate: conv exec1_row_group1_col_group0
      // generate: conv exec1_row_group1_col_group1
      for (int i2 = 0; i2 < 30; i2++) { // generate : conv exec1_row_group1_col_group1

        // generate: load_block
        // generate : load_block. loop count == 1
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((40, odata), ((41, 32), data)), imce_2_1 -> imce_3_1

        } // endgenerate
        // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var5 = __builtin_IMCE_GET_CREG((short)0);
        var6 = __builtin_IMCE_GET_CREG((short)1);
        var7 = __builtin_IMCE_GET_CREG((short)2);
        var8 = __builtin_IMCE_GET_CREG((short)3);
        // generate: batch_norm

        var45 = __builtin_IMCE_MULTL(var5, var43, 15);
        var45 = __builtin_IMCE_ADD(var45, var44, 15);
        var46 = __builtin_IMCE_MULTL(var6, var49, 15);
        var46 = __builtin_IMCE_ADD(var46, var50, 15);
        var47 = __builtin_IMCE_MULTL(var7, var51, 15);
        var47 = __builtin_IMCE_ADD(var47, var52, 15);
        var48 = __builtin_IMCE_MULTL(var8, var53, 15);
        var48 = __builtin_IMCE_ADD(var48, var54, 15);
        // endgenerate: batch_norm
        __builtin_IMCE_SEND(1, var45, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
        __builtin_IMCE_SEND(1, var46, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
        __builtin_IMCE_SEND(1, var47, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
        __builtin_IMCE_SEND(1, var48, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
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

      var45 = __builtin_IMCE_MULTL(var5, var43, 15);
      var45 = __builtin_IMCE_ADD(var45, var44, 15);
      var46 = __builtin_IMCE_MULTL(var6, var49, 15);
      var46 = __builtin_IMCE_ADD(var46, var50, 15);
      var47 = __builtin_IMCE_MULTL(var7, var51, 15);
      var47 = __builtin_IMCE_ADD(var47, var52, 15);
      var48 = __builtin_IMCE_MULTL(var8, var53, 15);
      var48 = __builtin_IMCE_ADD(var48, var54, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var45, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var46, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var47, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var48, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
      // endgenerate : conv exec1_row_group1_col_group2
      // endgenerate: conv exec1_row_group1_col_group2
    } // endgenerate : conv exec1_row_group1_outer_loop(iterate row offset)
    // endgenerate: conv exec1_row_group1_outer_loop(iterate row offset)
    // generate: conv exec1_row_group2_outer_loop(iterate row offset)
    // generate : conv exec1_row_group2_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec1_row_group2_col_group0
    for (int i1 = 0; i1 < 32; i1++) { // generate : conv exec1_row_group2_col_group0

      // generate: load_block
      // loop ignored with loop count == 0 : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      // generate: batch_norm

      var45 = __builtin_IMCE_MULTL(var5, var43, 15);
      var45 = __builtin_IMCE_ADD(var45, var44, 15);
      var46 = __builtin_IMCE_MULTL(var6, var49, 15);
      var46 = __builtin_IMCE_ADD(var46, var50, 15);
      var47 = __builtin_IMCE_MULTL(var7, var51, 15);
      var47 = __builtin_IMCE_ADD(var47, var52, 15);
      var48 = __builtin_IMCE_MULTL(var8, var53, 15);
      var48 = __builtin_IMCE_ADD(var48, var54, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var45, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var46, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var47, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var48, 2, 0); // TensorEdge(((41, 33), odata), (43, func_out0), 0), imce_3_1 -> inode_3_0
    } // endgenerate : conv exec1_row_group2_col_group0
    // endgenerate: conv exec1_row_group2_col_group0
    // endgenerate : conv exec1_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec1_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec1
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 2) { // imce_3_2
  }
  else if (hid == 3 && wid == 3) { // imce_3_3
  }
  else if (hid == 3 && wid == 4) { // imce_3_4
  }
}
