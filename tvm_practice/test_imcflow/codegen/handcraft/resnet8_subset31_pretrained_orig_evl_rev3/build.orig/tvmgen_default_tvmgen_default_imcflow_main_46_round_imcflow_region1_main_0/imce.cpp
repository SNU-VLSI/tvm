#include "../common_decl.h"
void tvmgen_default_tvmgen_default_imcflow_main_46_round_imcflow_region1_main_0() {
  int hid = __builtin_IMCE_GET_CORE_HID();
  int wid = __builtin_IMCE_GET_CORE_WID();
  short16 var1; // (ConvBlock(gid: 36), 0)
  short16 var2; // (ConvBlock(gid: 36), 1)
  short16 var3; // (ConvBlock(gid: 36), 2)
  short16 var4; // (ConvBlock(gid: 36), 3)
  short16 var5; // (ConvBlock(gid: 38), 0)
  short16 var6; // (ConvBlock(gid: 38), 1)
  short16 var7; // (ConvBlock(gid: 38), 2)
  short16 var8; // (ConvBlock(gid: 38), 3)
  short16 var9; // (TensorEdge(((43, -28), scale), ((43, 41), rhs)), 0)
  short16 var10; // (TensorEdge((-13, odata), ((43, 40), data)), 0)
  short16 var11; // (MultlBlock(gid: 41), 0)
  short16 var12; // (ReLUBlock(gid: 40), 0)
  short16 var13; // ((MultlBlock(gid: 41), 0), 'L')
  short16 var14; // ((MultlBlock(gid: 41), 0), 'H')
  short16 var15; // ((MultlBlock(gid: 41), 0), 'neg1')
  short16 var16; // ((MultlBlock(gid: 41), 0), 'const_7fff')
  short16 var17; // ((MultlBlock(gid: 41), 0), 'H_sign')
  short16 var18; // ((MultlBlock(gid: 41), 0), 'L_sign')
  short16 var19; // ((MultlBlock(gid: 41), 0), 'mismatch')
  short16 var20; // ((MultlBlock(gid: 41), 0), 'saturate_val')
  short16 var21; // ((MultlBlock(gid: 41), 0), 'not_mismatch')
  short16 var22; // ((MultlBlock(gid: 41), 0), 'part1')
  short16 var23; // ((MultlBlock(gid: 41), 0), 'part2')
  short16 var24; // (TensorEdge((-19, min), (35, min)), 0)
  short16 var25; // (TensorEdge((-20, max), (35, max)), 0)
  short16 var26; // (TensorEdge((-13, odata), (35, data)), 0)
  short16 var27; // (MinmaxQuantBlock(gid: 35), 0)
  short16 var28; // (MinmaxQuantBlock(gid: 35), 1)
  short16 var29; // (MinmaxQuantBlock(gid: 35), 2)
  short16 var30; // (MinmaxQuantBlock(gid: 35), 3)
  short16 var31; // (TensorEdge(((37, -15), fused_scale), ((37, 32), fused_scale)), 0)
  short16 var32; // (TensorEdge(((37, -16), fused_bias), ((37, 32), fused_bias)), 0)
  short16 var33; // (TensorEdge(((37, -17), min), ((37, 33), min)), 0)
  short16 var34; // (TensorEdge(((37, -18), max), ((37, 33), max)), 0)
  short16 var35; // (TensorEdge((36, odata), ((37, 32), data)), 0)
  short16 var36; // (TensorEdge((36, odata), ((37, 32), data)), 1)
  short16 var37; // (TensorEdge((36, odata), ((37, 32), data)), 2)
  short16 var38; // (TensorEdge((36, odata), ((37, 32), data)), 3)
  short16 var39; // (MinmaxQuantBlock(gid: 33), 0)
  short16 var40; // (MinmaxQuantBlock(gid: 33), 1)
  short16 var41; // (MinmaxQuantBlock(gid: 33), 2)
  short16 var42; // (MinmaxQuantBlock(gid: 33), 3)
  short16 var43; // (BatchNormBlock(gid: 32), 0)
  short16 var44; // (BatchNormBlock(gid: 32), 0, 'mult_result')
  short16 var45; // ((BatchNormBlock(gid: 32), 0), 'L')
  short16 var46; // ((BatchNormBlock(gid: 32), 0), 'H')
  short16 var47; // ((BatchNormBlock(gid: 32), 0), 'neg1')
  short16 var48; // ((BatchNormBlock(gid: 32), 0), 'const_7fff')
  short16 var49; // ((BatchNormBlock(gid: 32), 0), 'H_sign')
  short16 var50; // ((BatchNormBlock(gid: 32), 0), 'L_sign')
  short16 var51; // ((BatchNormBlock(gid: 32), 0), 'mismatch')
  short16 var52; // ((BatchNormBlock(gid: 32), 0), 'saturate_val')
  short16 var53; // ((BatchNormBlock(gid: 32), 0), 'not_mismatch')
  short16 var54; // ((BatchNormBlock(gid: 32), 0), 'part1')
  short16 var55; // ((BatchNormBlock(gid: 32), 0), 'part2')
  short16 var56; // (TensorEdge((-22, config), (36, config)), 0)
  short16 var57; // (TensorEdge((35, odata), (36, data)), 0)
  short16 var58; // (TensorEdge((35, odata), (36, data)), 1)
  short16 var59; // (TensorEdge((35, odata), (36, data)), 2)
  short16 var60; // (TensorEdge((35, odata), (36, data)), 3)
  short16 var61; // (TensorEdge((-24, config), (38, config)), 0)
  short16 var62; // (TensorEdge(((37, 33), odata), (38, data)), 0)
  short16 var63; // (TensorEdge(((37, 33), odata), (38, data)), 1)
  short16 var64; // (TensorEdge(((37, 33), odata), (38, data)), 2)
  short16 var65; // (TensorEdge(((37, 33), odata), (38, data)), 3)
  short16 var66; // (TensorEdge((-25, fused_scale), (39, fused_scale)), 0)
  short16 var67; // (TensorEdge((-26, fused_bias), (39, fused_bias)), 0)
  short16 var68; // (TensorEdge((38, odata), (39, data)), 0)
  short16 var69; // (BatchNormBlock(gid: 39), 0)
  short16 var70; // (BatchNormBlock(gid: 39), 0, 'mult_result')
  short16 var71; // ((BatchNormBlock(gid: 39), 0), 'L')
  short16 var72; // ((BatchNormBlock(gid: 39), 0), 'H')
  short16 var73; // ((BatchNormBlock(gid: 39), 0), 'neg1')
  short16 var74; // ((BatchNormBlock(gid: 39), 0), 'const_7fff')
  short16 var75; // ((BatchNormBlock(gid: 39), 0), 'H_sign')
  short16 var76; // ((BatchNormBlock(gid: 39), 0), 'L_sign')
  short16 var77; // ((BatchNormBlock(gid: 39), 0), 'mismatch')
  short16 var78; // ((BatchNormBlock(gid: 39), 0), 'saturate_val')
  short16 var79; // ((BatchNormBlock(gid: 39), 0), 'not_mismatch')
  short16 var80; // ((BatchNormBlock(gid: 39), 0), 'part1')
  short16 var81; // ((BatchNormBlock(gid: 39), 0), 'part2')
  if (hid == 0 && wid == 1) { // imce_0_1
    // generate: mult const

    var9 = __builtin_IMCE_RECV(1);
    // endgenerate: mult const
    // generate: call_created_loop
    for (int i1 = 0; i1 < 1024; i1++) { // generate : call_created_loop
      // generate: imcflow.vecops_wrapper

      var10 = __builtin_IMCE_RECV(2); // TensorEdge((-13, odata), ((43, 40), data)), inode_0_0 -> imce_0_1
      // generate: imcflow.vecops_block
      // generate: relu

      var12 = __builtin_IMCE_MAXI(var10, 0);
      // endgenerate: relu
      // generate: multl


      var13 = __builtin_IMCE_MULTL(var9, var12, 15);
      var14 = __builtin_IMCE_MULTH(var9, var12, 15);
      var15 = __builtin_IMCE_SUBI(0, 1);
      var16 = __builtin_IMCE_SRLI(var15, 1);
      var17 = __builtin_IMCE_SRAI(var14, 15);
      var18 = __builtin_IMCE_SRAI(var13, 15);
      var19 = __builtin_IMCE_XOR(var17, var18, 15);
      var20 = __builtin_IMCE_XOR(var17, var16, 15);
      var21 = __builtin_IMCE_XOR(var19, var15, 15);
      var22 = __builtin_IMCE_AND(var19, var20, 15);
      var23 = __builtin_IMCE_AND(var21, var13, 15);
      var11 = __builtin_IMCE_OR(var22, var23, 15);
      // endgenerate: multl
      // endgenerate: imcflow.vecops_block
      __builtin_IMCE_SEND(1, var11, 2, 0); // TensorEdge(((43, 41), odata), (44, func_out1), 1), imce_0_1 -> inode_2_0
      // endgenerate: imcflow.vecops_wrapper
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 0 && wid == 2) { // imce_0_2
    // generate: TensorEdge((-19, min), (35, min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge((-19, min), (35, min)), min write
    // generate: TensorEdge((-20, max), (35, max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge((-20, max), (35, max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 1024; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      var26 = __builtin_IMCE_RECV(2); // TensorEdge((-13, odata), (35, data)), inode_0_0 -> imce_0_1, imce_0_2
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var26, 0, 15, 0);
      var27 = __builtin_IMCE_GET_QREG(0);
      var28 = __builtin_IMCE_GET_QREG(1);
      var29 = __builtin_IMCE_GET_QREG(2);
      var30 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      __builtin_IMCE_SEND(1, var27, 0, 0); // TensorEdge((35, odata), (36, data)), imce_0_2 -> imce_1_2
      __builtin_IMCE_SEND(1, var28, 0, 0); // TensorEdge((35, odata), (36, data)), imce_0_2 -> imce_1_2
      __builtin_IMCE_SEND(1, var29, 0, 0); // TensorEdge((35, odata), (36, data)), imce_0_2 -> imce_1_2
      __builtin_IMCE_SEND(1, var30, 0, 0); // TensorEdge((35, odata), (36, data)), imce_0_2 -> imce_1_2
      // endgenerate: min_max_quantize_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 0 && wid == 3) { // imce_0_3
  }
  else if (hid == 0 && wid == 4) { // imce_0_4
  }
  else if (hid == 1 && wid == 1) { // imce_1_1
    // generate: TensorEdge(((37, -15), fused_scale), ((37, 32), fused_scale)), fused_scale write

    var31 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((37, -15), fused_scale), ((37, 32), fused_scale)), fused_scale write
    // generate: TensorEdge(((37, -16), fused_bias), ((37, 32), fused_bias)), fused_bias write

    var32 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((37, -16), fused_bias), ((37, 32), fused_bias)), fused_bias write
    // generate: TensorEdge(((37, -17), min), ((37, 33), min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge(((37, -17), min), ((37, 33), min)), min write
    // generate: TensorEdge(((37, -18), max), ((37, 33), max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge(((37, -18), max), ((37, 33), max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 1024; i1++) { // generate : call_created_loop
      // generate: imcflow.preop-minmax_wrapper

      var35 = __builtin_IMCE_RECV(2); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
      var36 = __builtin_IMCE_RECV(2); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
      var37 = __builtin_IMCE_RECV(2); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
      var38 = __builtin_IMCE_RECV(2); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
      // generate: imcflow.preop-minmax_block
      // generate: batch_norm


      var45 = __builtin_IMCE_MULTL(var35, var31, 15);
      var46 = __builtin_IMCE_MULTH(var35, var31, 15);
      var47 = __builtin_IMCE_SUBI(0, 1);
      var48 = __builtin_IMCE_SRLI(var47, 1);
      var49 = __builtin_IMCE_SRAI(var46, 15);
      var50 = __builtin_IMCE_SRAI(var45, 15);
      var51 = __builtin_IMCE_XOR(var49, var50, 15);
      var52 = __builtin_IMCE_XOR(var49, var48, 15);
      var53 = __builtin_IMCE_XOR(var51, var47, 15);
      var54 = __builtin_IMCE_AND(var51, var52, 15);
      var55 = __builtin_IMCE_AND(var53, var45, 15);
      var44 = __builtin_IMCE_OR(var54, var55, 15);
      var43 = __builtin_IMCE_ADD(var44, var32, 15);
      // endgenerate: batch_norm
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var43, 0, 15, 0);
      var39 = __builtin_IMCE_GET_QREG(0);
      var40 = __builtin_IMCE_GET_QREG(1);
      var41 = __builtin_IMCE_GET_QREG(2);
      var42 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      // endgenerate: imcflow.preop-minmax_block
      __builtin_IMCE_SEND(2, var39, 0, 0); // TensorEdge(((37, 33), odata), (38, data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(2, var40, 0, 0); // TensorEdge(((37, 33), odata), (38, data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(2, var41, 0, 0); // TensorEdge(((37, 33), odata), (38, data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(2, var42, 0, 0); // TensorEdge(((37, 33), odata), (38, data)), imce_1_1 -> imce_2_1
      // endgenerate: imcflow.preop-minmax_wrapper
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 1 && wid == 2) { // imce_1_2
    // generate: TensorEdge((-22, config), (36, config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge((-22, config), (36, config)), config write
    // generate: conv exec0
    // generate: conv exec0_row_group0_outer_loop(iterate row offset)
    // generate : conv exec0_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec0_row_group0_col_group0
    // generate : conv exec0_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 34; i1++) { // generate : load_block
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((35, odata), (36, data)), imce_0_2 -> imce_1_2

      } // endgenerate
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var1 = __builtin_IMCE_GET_CREG((short)0);
    var2 = __builtin_IMCE_GET_CREG((short)1);
    var3 = __builtin_IMCE_GET_CREG((short)2);
    var4 = __builtin_IMCE_GET_CREG((short)3);
    __builtin_IMCE_SEND(1, var1, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
    __builtin_IMCE_SEND(1, var2, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
    __builtin_IMCE_SEND(1, var3, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
    __builtin_IMCE_SEND(1, var4, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
    // endgenerate : conv exec0_row_group0_col_group0
    // endgenerate: conv exec0_row_group0_col_group0
    // generate: conv exec0_row_group0_col_group1
    for (int i1 = 0; i1 < 30; i1++) { // generate : conv exec0_row_group0_col_group1

      // generate: load_block
      // generate : load_block. loop count == 1
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((35, odata), (36, data)), imce_0_2 -> imce_1_2

      } // endgenerate
      // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var1 = __builtin_IMCE_GET_CREG((short)0);
      var2 = __builtin_IMCE_GET_CREG((short)1);
      var3 = __builtin_IMCE_GET_CREG((short)2);
      var4 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(1, var1, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var2, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var3, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var4, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
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
    __builtin_IMCE_SEND(1, var1, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
    __builtin_IMCE_SEND(1, var2, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
    __builtin_IMCE_SEND(1, var3, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
    __builtin_IMCE_SEND(1, var4, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
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
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((35, odata), (36, data)), imce_0_2 -> imce_1_2

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var1 = __builtin_IMCE_GET_CREG((short)0);
      var2 = __builtin_IMCE_GET_CREG((short)1);
      var3 = __builtin_IMCE_GET_CREG((short)2);
      var4 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(1, var1, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var2, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var3, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var4, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
      // endgenerate : conv exec0_row_group1_col_group0
      // endgenerate: conv exec0_row_group1_col_group0
      // generate: conv exec0_row_group1_col_group1
      for (int i2 = 0; i2 < 30; i2++) { // generate : conv exec0_row_group1_col_group1

        // generate: load_block
        // generate : load_block. loop count == 1
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((35, odata), (36, data)), imce_0_2 -> imce_1_2

        } // endgenerate
        // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var1 = __builtin_IMCE_GET_CREG((short)0);
        var2 = __builtin_IMCE_GET_CREG((short)1);
        var3 = __builtin_IMCE_GET_CREG((short)2);
        var4 = __builtin_IMCE_GET_CREG((short)3);
        __builtin_IMCE_SEND(1, var1, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
        __builtin_IMCE_SEND(1, var2, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
        __builtin_IMCE_SEND(1, var3, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
        __builtin_IMCE_SEND(1, var4, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
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
      __builtin_IMCE_SEND(1, var1, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var2, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var3, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var4, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
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
      __builtin_IMCE_SEND(1, var1, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var2, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var3, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var4, 2, 0); // TensorEdge((36, odata), ((37, 32), data)), imce_1_2 -> imce_1_1
    } // endgenerate : conv exec0_row_group2_col_group0
    // endgenerate: conv exec0_row_group2_col_group0
    // endgenerate : conv exec0_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec0_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec0
    __builtin_IMCE_STOP();
  }
  else if (hid == 1 && wid == 3) { // imce_1_3
  }
  else if (hid == 1 && wid == 4) { // imce_1_4
  }
  else if (hid == 2 && wid == 1) { // imce_2_1
    // generate: TensorEdge((-24, config), (38, config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge((-24, config), (38, config)), config write
    // generate: conv exec1
    // generate: conv exec1_row_group0_outer_loop(iterate row offset)
    // generate : conv exec1_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec1_row_group0_col_group0
    // generate : conv exec1_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 34; i1++) { // generate : load_block
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge(((37, 33), odata), (38, data)), imce_1_1 -> imce_2_1

      } // endgenerate
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var5 = __builtin_IMCE_GET_CREG((short)0);
    var6 = __builtin_IMCE_GET_CREG((short)1);
    var7 = __builtin_IMCE_GET_CREG((short)2);
    var8 = __builtin_IMCE_GET_CREG((short)3);
    __builtin_IMCE_SEND(1, var5, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
    __builtin_IMCE_SEND(1, var6, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
    __builtin_IMCE_SEND(1, var7, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
    __builtin_IMCE_SEND(1, var8, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
    // endgenerate : conv exec1_row_group0_col_group0
    // endgenerate: conv exec1_row_group0_col_group0
    // generate: conv exec1_row_group0_col_group1
    for (int i1 = 0; i1 < 30; i1++) { // generate : conv exec1_row_group0_col_group1

      // generate: load_block
      // generate : load_block. loop count == 1
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge(((37, 33), odata), (38, data)), imce_1_1 -> imce_2_1

      } // endgenerate
      // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(1, var5, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var6, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var7, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var8, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
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
    __builtin_IMCE_SEND(1, var5, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
    __builtin_IMCE_SEND(1, var6, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
    __builtin_IMCE_SEND(1, var7, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
    __builtin_IMCE_SEND(1, var8, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
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
          __builtin_IMCE_LOAD_LB(0); // TensorEdge(((37, 33), odata), (38, data)), imce_1_1 -> imce_2_1

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(1, var5, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var6, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var7, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var8, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
      // endgenerate : conv exec1_row_group1_col_group0
      // endgenerate: conv exec1_row_group1_col_group0
      // generate: conv exec1_row_group1_col_group1
      for (int i2 = 0; i2 < 30; i2++) { // generate : conv exec1_row_group1_col_group1

        // generate: load_block
        // generate : load_block. loop count == 1
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge(((37, 33), odata), (38, data)), imce_1_1 -> imce_2_1

        } // endgenerate
        // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var5 = __builtin_IMCE_GET_CREG((short)0);
        var6 = __builtin_IMCE_GET_CREG((short)1);
        var7 = __builtin_IMCE_GET_CREG((short)2);
        var8 = __builtin_IMCE_GET_CREG((short)3);
        __builtin_IMCE_SEND(1, var5, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
        __builtin_IMCE_SEND(1, var6, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
        __builtin_IMCE_SEND(1, var7, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
        __builtin_IMCE_SEND(1, var8, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
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
      __builtin_IMCE_SEND(1, var5, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var6, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var7, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var8, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
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
      __builtin_IMCE_SEND(1, var5, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var6, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var7, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var8, 2, 0); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
    } // endgenerate : conv exec1_row_group2_col_group0
    // endgenerate: conv exec1_row_group2_col_group0
    // endgenerate : conv exec1_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec1_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec1
    __builtin_IMCE_STOP();
  }
  else if (hid == 2 && wid == 2) { // imce_2_2
  }
  else if (hid == 2 && wid == 3) { // imce_2_3
  }
  else if (hid == 2 && wid == 4) { // imce_2_4
  }
  else if (hid == 3 && wid == 1) { // imce_3_1
    // generate: TensorEdge((-25, fused_scale), (39, fused_scale)), fused_scale write

    var66 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge((-25, fused_scale), (39, fused_scale)), fused_scale write
    // generate: TensorEdge((-26, fused_bias), (39, fused_bias)), fused_bias write

    var67 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge((-26, fused_bias), (39, fused_bias)), fused_bias write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 4096; i1++) { // generate : call_created_loop
      // generate: bn_standalone

      var68 = __builtin_IMCE_RECV(2); // TensorEdge((38, odata), (39, data)), imce_2_1 -> imce_3_1
      // generate: batch_norm


      var71 = __builtin_IMCE_MULTL(var68, var66, 15);
      var72 = __builtin_IMCE_MULTH(var68, var66, 15);
      var73 = __builtin_IMCE_SUBI(0, 1);
      var74 = __builtin_IMCE_SRLI(var73, 1);
      var75 = __builtin_IMCE_SRAI(var72, 15);
      var76 = __builtin_IMCE_SRAI(var71, 15);
      var77 = __builtin_IMCE_XOR(var75, var76, 15);
      var78 = __builtin_IMCE_XOR(var75, var74, 15);
      var79 = __builtin_IMCE_XOR(var77, var73, 15);
      var80 = __builtin_IMCE_AND(var77, var78, 15);
      var81 = __builtin_IMCE_AND(var79, var71, 15);
      var70 = __builtin_IMCE_OR(var80, var81, 15);
      var69 = __builtin_IMCE_ADD(var70, var67, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var69, 2, 0); // TensorEdge((39, odata), (44, func_out0), 0), imce_3_1 -> inode_3_0
      // endgenerate: bn_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 2) { // imce_3_2
  }
  else if (hid == 3 && wid == 3) { // imce_3_3
  }
  else if (hid == 3 && wid == 4) { // imce_3_4
  }
}
