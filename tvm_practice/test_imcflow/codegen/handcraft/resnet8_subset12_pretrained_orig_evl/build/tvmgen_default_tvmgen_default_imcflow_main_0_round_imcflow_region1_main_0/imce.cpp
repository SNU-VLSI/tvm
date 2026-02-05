#include "../common_decl.h"
void tvmgen_default_tvmgen_default_imcflow_main_0_round_imcflow_region1_main_0() {
  int hid = __builtin_IMCE_GET_CORE_HID();
  int wid = __builtin_IMCE_GET_CORE_WID();
  short16 var1; // (ConvBlock(gid: 19), 0)
  short16 var2; // (ConvBlock(gid: 19), 1)
  short16 var3; // (ConvBlock(gid: 19), 2)
  short16 var4; // (ConvBlock(gid: 19), 3)
  short16 var5; // (ConvBlock(gid: 21), 0)
  short16 var6; // (ConvBlock(gid: 21), 1)
  short16 var7; // (ConvBlock(gid: 21), 2)
  short16 var8; // (ConvBlock(gid: 21), 3)
  short16 var9; // (TensorEdge((-24, scale), (23, rhs)), 0)
  short16 var10; // (TensorEdge((-10, odata), (23, lhs)), 0)
  short16 var11; // (MultlBlock(gid: 23), 0)
  short16 var12; // ((MultlBlock(gid: 23), 0), 'L')
  short16 var13; // ((MultlBlock(gid: 23), 0), 'H')
  short16 var14; // ((MultlBlock(gid: 23), 0), 'neg1')
  short16 var15; // ((MultlBlock(gid: 23), 0), 'const_7fff')
  short16 var16; // ((MultlBlock(gid: 23), 0), 'H_sign')
  short16 var17; // ((MultlBlock(gid: 23), 0), 'L_sign')
  short16 var18; // ((MultlBlock(gid: 23), 0), 'mismatch')
  short16 var19; // ((MultlBlock(gid: 23), 0), 'saturate_val')
  short16 var20; // ((MultlBlock(gid: 23), 0), 'not_mismatch')
  short16 var21; // ((MultlBlock(gid: 23), 0), 'part1')
  short16 var22; // ((MultlBlock(gid: 23), 0), 'part2')
  short16 var23; // (TensorEdge((-16, min), (18, min)), 0)
  short16 var24; // (TensorEdge((-17, max), (18, max)), 0)
  short16 var25; // (TensorEdge((-10, odata), (18, data)), 0)
  short16 var26; // (MinmaxQuantBlock(gid: 18), 0)
  short16 var27; // (MinmaxQuantBlock(gid: 18), 1)
  short16 var28; // (MinmaxQuantBlock(gid: 18), 2)
  short16 var29; // (MinmaxQuantBlock(gid: 18), 3)
  short16 var30; // (TensorEdge(((20, -12), fused_scale), ((20, 15), fused_scale)), 0)
  short16 var31; // (TensorEdge(((20, -13), fused_bias), ((20, 15), fused_bias)), 0)
  short16 var32; // (TensorEdge(((20, -14), min), ((20, 16), min)), 0)
  short16 var33; // (TensorEdge(((20, -15), max), ((20, 16), max)), 0)
  short16 var34; // (TensorEdge((19, odata), ((20, 15), data)), 0)
  short16 var35; // (TensorEdge((19, odata), ((20, 15), data)), 1)
  short16 var36; // (TensorEdge((19, odata), ((20, 15), data)), 2)
  short16 var37; // (TensorEdge((19, odata), ((20, 15), data)), 3)
  short16 var38; // (MinmaxQuantBlock(gid: 16), 0)
  short16 var39; // (MinmaxQuantBlock(gid: 16), 1)
  short16 var40; // (MinmaxQuantBlock(gid: 16), 2)
  short16 var41; // (MinmaxQuantBlock(gid: 16), 3)
  short16 var42; // (BatchNormBlock(gid: 15), 0)
  short16 var43; // (BatchNormBlock(gid: 15), 0, 'mult_result')
  short16 var44; // ((BatchNormBlock(gid: 15), 0), 'L')
  short16 var45; // ((BatchNormBlock(gid: 15), 0), 'H')
  short16 var46; // ((BatchNormBlock(gid: 15), 0), 'neg1')
  short16 var47; // ((BatchNormBlock(gid: 15), 0), 'const_7fff')
  short16 var48; // ((BatchNormBlock(gid: 15), 0), 'H_sign')
  short16 var49; // ((BatchNormBlock(gid: 15), 0), 'L_sign')
  short16 var50; // ((BatchNormBlock(gid: 15), 0), 'mismatch')
  short16 var51; // ((BatchNormBlock(gid: 15), 0), 'saturate_val')
  short16 var52; // ((BatchNormBlock(gid: 15), 0), 'not_mismatch')
  short16 var53; // ((BatchNormBlock(gid: 15), 0), 'part1')
  short16 var54; // ((BatchNormBlock(gid: 15), 0), 'part2')
  short16 var55; // (TensorEdge((-19, config), (19, config)), 0)
  short16 var56; // (TensorEdge((18, odata), (19, data)), 0)
  short16 var57; // (TensorEdge((18, odata), (19, data)), 1)
  short16 var58; // (TensorEdge((18, odata), (19, data)), 2)
  short16 var59; // (TensorEdge((18, odata), (19, data)), 3)
  short16 var60; // (TensorEdge((-21, config), (21, config)), 0)
  short16 var61; // (TensorEdge(((20, 16), odata), (21, data)), 0)
  short16 var62; // (TensorEdge(((20, 16), odata), (21, data)), 1)
  short16 var63; // (TensorEdge(((20, 16), odata), (21, data)), 2)
  short16 var64; // (TensorEdge(((20, 16), odata), (21, data)), 3)
  short16 var65; // (TensorEdge((-22, fused_scale), (22, fused_scale)), 0)
  short16 var66; // (TensorEdge((-23, fused_bias), (22, fused_bias)), 0)
  short16 var67; // (TensorEdge((21, odata), (22, data)), 0)
  short16 var68; // (BatchNormBlock(gid: 22), 0)
  short16 var69; // (BatchNormBlock(gid: 22), 0, 'mult_result')
  short16 var70; // ((BatchNormBlock(gid: 22), 0), 'L')
  short16 var71; // ((BatchNormBlock(gid: 22), 0), 'H')
  short16 var72; // ((BatchNormBlock(gid: 22), 0), 'neg1')
  short16 var73; // ((BatchNormBlock(gid: 22), 0), 'const_7fff')
  short16 var74; // ((BatchNormBlock(gid: 22), 0), 'H_sign')
  short16 var75; // ((BatchNormBlock(gid: 22), 0), 'L_sign')
  short16 var76; // ((BatchNormBlock(gid: 22), 0), 'mismatch')
  short16 var77; // ((BatchNormBlock(gid: 22), 0), 'saturate_val')
  short16 var78; // ((BatchNormBlock(gid: 22), 0), 'not_mismatch')
  short16 var79; // ((BatchNormBlock(gid: 22), 0), 'part1')
  short16 var80; // ((BatchNormBlock(gid: 22), 0), 'part2')
  if (hid == 0 && wid == 1) { // imce_0_1
    // generate: mult const

    var9 = __builtin_IMCE_RECV(1);
    // endgenerate: mult const
    // generate: call_created_loop
    for (int i1 = 0; i1 < 1024; i1++) { // generate : call_created_loop
      // generate: multiply standalone

      var10 = __builtin_IMCE_RECV(2); // TensorEdge((-10, odata), (23, lhs)), inode_0_0 -> imce_0_1
      // generate: multl


      var12 = __builtin_IMCE_MULTL(var10, var9, 15);
      var13 = __builtin_IMCE_MULTH(var10, var9, 15);
      var14 = __builtin_IMCE_SUBI(0, 1);
      var15 = __builtin_IMCE_SRLI(var14, 1);
      var16 = __builtin_IMCE_SRAI(var13, 15);
      var17 = __builtin_IMCE_SRAI(var12, 15);
      var18 = __builtin_IMCE_XOR(var16, var17, 15);
      var19 = __builtin_IMCE_XOR(var16, var15, 15);
      var20 = __builtin_IMCE_XOR(var18, var14, 15);
      var21 = __builtin_IMCE_AND(var18, var19, 15);
      var22 = __builtin_IMCE_AND(var20, var12, 15);
      var11 = __builtin_IMCE_OR(var21, var22, 15);
      // endgenerate: multl
      __builtin_IMCE_SEND(1, var11, 2, 0); // TensorEdge((23, odata), (24, func_out1), 1), imce_0_1 -> inode_2_0
      // endgenerate: multiply standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 0 && wid == 2) { // imce_0_2
    // generate: TensorEdge((-16, min), (18, min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge((-16, min), (18, min)), min write
    // generate: TensorEdge((-17, max), (18, max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge((-17, max), (18, max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 1024; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      __builtin_IMCE_SETFLAG(1);
      __builtin_IMCE_STANDBY(0, 1); 
      __builtin_IMCE_SETFLAG(0); 
      var25 = __builtin_IMCE_RECV(2); // TensorEdge((-10, odata), (18, data)), inode_0_0 -> imce_0_1, imce_0_2
      // generate: min_max_quantize

      __builtin_IMCE_STANDBY(7, 1); // STANDBY is not inserted before SEND but before MM_QUNAT, because of overwritten QREGs for valid=1
      __builtin_IMCE_MM_QUANT(var25, 0, 15, 0);
      var26 = __builtin_IMCE_GET_QREG(0);
      var27 = __builtin_IMCE_GET_QREG(1);
      var28 = __builtin_IMCE_GET_QREG(2);
      var29 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      __builtin_IMCE_SEND(1, var26, 0, 0); // TensorEdge((18, odata), (19, data)), imce_0_2 -> imce_1_2
      __builtin_IMCE_SEND(1, var27, 0, 0); // TensorEdge((18, odata), (19, data)), imce_0_2 -> imce_1_2
      __builtin_IMCE_SEND(1, var28, 0, 0); // TensorEdge((18, odata), (19, data)), imce_0_2 -> imce_1_2
      __builtin_IMCE_SEND(1, var29, 0, 0); // TensorEdge((18, odata), (19, data)), imce_0_2 -> imce_1_2
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
    // generate: TensorEdge(((20, -12), fused_scale), ((20, 15), fused_scale)), fused_scale write

    var30 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((20, -12), fused_scale), ((20, 15), fused_scale)), fused_scale write
    // generate: TensorEdge(((20, -13), fused_bias), ((20, 15), fused_bias)), fused_bias write

    var31 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((20, -13), fused_bias), ((20, 15), fused_bias)), fused_bias write
    // generate: TensorEdge(((20, -14), min), ((20, 16), min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge(((20, -14), min), ((20, 16), min)), min write
    // generate: TensorEdge(((20, -15), max), ((20, 16), max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge(((20, -15), max), ((20, 16), max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 1024; i1++) { // generate : call_created_loop
      // generate: imcflow.preop-minmax_wrapper

      __builtin_IMCE_SETFLAG(1);
      var34 = __builtin_IMCE_RECV(2); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
      var35 = __builtin_IMCE_RECV(2); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
      var36 = __builtin_IMCE_RECV(2); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
      var37 = __builtin_IMCE_RECV(2); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SETFLAG(0);
      // generate: imcflow.preop-minmax_block
      // generate: batch_norm


      var44 = __builtin_IMCE_MULTL(var34, var30, 15);
      var45 = __builtin_IMCE_MULTH(var34, var30, 15);
      var46 = __builtin_IMCE_SUBI(0, 1);
      var47 = __builtin_IMCE_SRLI(var46, 1);
      var48 = __builtin_IMCE_SRAI(var45, 15);
      var49 = __builtin_IMCE_SRAI(var44, 15);
      var50 = __builtin_IMCE_XOR(var48, var49, 15);
      var51 = __builtin_IMCE_XOR(var48, var47, 15);
      var52 = __builtin_IMCE_XOR(var50, var46, 15);
      var53 = __builtin_IMCE_AND(var50, var51, 15);
      var54 = __builtin_IMCE_AND(var52, var44, 15);
      var43 = __builtin_IMCE_OR(var53, var54, 15);
      var42 = __builtin_IMCE_ADD(var43, var31, 15);
      // endgenerate: batch_norm
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var42, 0, 15, 0);
      var38 = __builtin_IMCE_GET_QREG(0);
      var39 = __builtin_IMCE_GET_QREG(1);
      var40 = __builtin_IMCE_GET_QREG(2);
      var41 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      // endgenerate: imcflow.preop-minmax_block
      __builtin_IMCE_STANDBY(11, 1);
      __builtin_IMCE_SEND(2, var38, 0, 0); // TensorEdge(((20, 16), odata), (21, data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(2, var39, 0, 0); // TensorEdge(((20, 16), odata), (21, data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(2, var40, 0, 0); // TensorEdge(((20, 16), odata), (21, data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(2, var41, 0, 0); // TensorEdge(((20, 16), odata), (21, data)), imce_1_1 -> imce_2_1
      // endgenerate: imcflow.preop-minmax_wrapper
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 1 && wid == 2) { // imce_1_2
    // generate: TensorEdge((-19, config), (19, config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge((-19, config), (19, config)), config write
    // generate: conv exec0
    // generate: conv exec0_row_group0_outer_loop(iterate row offset)
    // generate : conv exec0_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec0_row_group0_col_group0
    // generate : conv exec0_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 34; i1++) { // generate : load_block
      __builtin_IMCE_SETFLAG(1);
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((18, odata), (19, data)), imce_0_2 -> imce_1_2

      } // endgenerate
      __builtin_IMCE_SETFLAG(0);
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var1 = __builtin_IMCE_GET_CREG((short)0);
    var2 = __builtin_IMCE_GET_CREG((short)1);
    var3 = __builtin_IMCE_GET_CREG((short)2);
    var4 = __builtin_IMCE_GET_CREG((short)3);

    __builtin_IMCE_STANDBY(6, 1);
    __builtin_IMCE_SEND(1, var1, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
    __builtin_IMCE_SEND(1, var2, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
    __builtin_IMCE_SEND(1, var3, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
    __builtin_IMCE_SEND(1, var4, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
    // endgenerate : conv exec0_row_group0_col_group0
    // endgenerate: conv exec0_row_group0_col_group0
    // generate: conv exec0_row_group0_col_group1
    for (int i1 = 0; i1 < 30; i1++) { // generate : conv exec0_row_group0_col_group1

      // generate: load_block
      // generate : load_block. loop count == 1
      __builtin_IMCE_SETFLAG(1);
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((18, odata), (19, data)), imce_0_2 -> imce_1_2

      } // endgenerate
      __builtin_IMCE_SETFLAG(0);
      // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var1 = __builtin_IMCE_GET_CREG((short)0);
      var2 = __builtin_IMCE_GET_CREG((short)1);
      var3 = __builtin_IMCE_GET_CREG((short)2);
      var4 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_STANDBY(6, 1);
      __builtin_IMCE_SEND(1, var1, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var2, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var3, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var4, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
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
    __builtin_IMCE_STANDBY(6, 1);
    __builtin_IMCE_SEND(1, var1, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
    __builtin_IMCE_SEND(1, var2, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
    __builtin_IMCE_SEND(1, var3, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
    __builtin_IMCE_SEND(1, var4, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
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
        __builtin_IMCE_SETFLAG(1);
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((18, odata), (19, data)), imce_0_2 -> imce_1_2

        } // endgenerate
        __builtin_IMCE_SETFLAG(0);
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var1 = __builtin_IMCE_GET_CREG((short)0);
      var2 = __builtin_IMCE_GET_CREG((short)1);
      var3 = __builtin_IMCE_GET_CREG((short)2);
      var4 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_STANDBY(6, 1);
      __builtin_IMCE_SEND(1, var1, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var2, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var3, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var4, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
      // endgenerate : conv exec0_row_group1_col_group0
      // endgenerate: conv exec0_row_group1_col_group0
      // generate: conv exec0_row_group1_col_group1
      for (int i2 = 0; i2 < 30; i2++) { // generate : conv exec0_row_group1_col_group1

        // generate: load_block
        // generate : load_block. loop count == 1
        __builtin_IMCE_SETFLAG(1);
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((18, odata), (19, data)), imce_0_2 -> imce_1_2

        } // endgenerate
        __builtin_IMCE_SETFLAG(0);
        // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var1 = __builtin_IMCE_GET_CREG((short)0);
        var2 = __builtin_IMCE_GET_CREG((short)1);
        var3 = __builtin_IMCE_GET_CREG((short)2);
        var4 = __builtin_IMCE_GET_CREG((short)3);
        __builtin_IMCE_STANDBY(6, 1);
        __builtin_IMCE_SEND(1, var1, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
        __builtin_IMCE_SEND(1, var2, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
        __builtin_IMCE_SEND(1, var3, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
        __builtin_IMCE_SEND(1, var4, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
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
      __builtin_IMCE_STANDBY(6, 1);
      __builtin_IMCE_SEND(1, var1, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var2, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var3, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var4, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
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
      __builtin_IMCE_SEND(1, 0, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, 0, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, 0, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, 0, 2, 0); // TensorEdge((19, odata), ((20, 15), data)), imce_1_2 -> imce_1_1
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
    // generate: TensorEdge((-21, config), (21, config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge((-21, config), (21, config)), config write
    // generate: conv exec1
    // generate: conv exec1_row_group0_outer_loop(iterate row offset)
    // generate : conv exec1_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec1_row_group0_col_group0
    // generate : conv exec1_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 34; i1++) { // generate : load_block
      __builtin_IMCE_SETFLAG(1);
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge(((20, 16), odata), (21, data)), imce_1_1 -> imce_2_1

      } // endgenerate
      __builtin_IMCE_SETFLAG(0);
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var5 = __builtin_IMCE_GET_CREG((short)0);
    var6 = __builtin_IMCE_GET_CREG((short)1);
    var7 = __builtin_IMCE_GET_CREG((short)2);
    var8 = __builtin_IMCE_GET_CREG((short)3);
    __builtin_IMCE_STANDBY(16, 1);
    __builtin_IMCE_SEND(1, var5, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
    __builtin_IMCE_SEND(1, var6, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
    __builtin_IMCE_SEND(1, var7, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
    __builtin_IMCE_SEND(1, var8, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
    // endgenerate : conv exec1_row_group0_col_group0
    // endgenerate: conv exec1_row_group0_col_group0
    // generate: conv exec1_row_group0_col_group1
    for (int i1 = 0; i1 < 30; i1++) { // generate : conv exec1_row_group0_col_group1

      // generate: load_block
      // generate : load_block. loop count == 1
      __builtin_IMCE_SETFLAG(1);
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge(((20, 16), odata), (21, data)), imce_1_1 -> imce_2_1

      } // endgenerate
      __builtin_IMCE_SETFLAG(0);
      // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_STANDBY(16, 1);
      __builtin_IMCE_SEND(1, var5, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var6, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var7, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var8, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
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
    __builtin_IMCE_STANDBY(16, 1);
    __builtin_IMCE_SEND(1, var5, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
    __builtin_IMCE_SEND(1, var6, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
    __builtin_IMCE_SEND(1, var7, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
    __builtin_IMCE_SEND(1, var8, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
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
        __builtin_IMCE_SETFLAG(1);
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge(((20, 16), odata), (21, data)), imce_1_1 -> imce_2_1

        } // endgenerate
        __builtin_IMCE_SETFLAG(0);
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_STANDBY(16, 1);
      __builtin_IMCE_SEND(1, var5, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var6, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var7, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var8, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
      // endgenerate : conv exec1_row_group1_col_group0
      // endgenerate: conv exec1_row_group1_col_group0
      // generate: conv exec1_row_group1_col_group1
      for (int i2 = 0; i2 < 30; i2++) { // generate : conv exec1_row_group1_col_group1

        // generate: load_block
        // generate : load_block. loop count == 1
        __builtin_IMCE_SETFLAG(1);
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge(((20, 16), odata), (21, data)), imce_1_1 -> imce_2_1

        } // endgenerate
        __builtin_IMCE_SETFLAG(0);
        // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var5 = __builtin_IMCE_GET_CREG((short)0);
        var6 = __builtin_IMCE_GET_CREG((short)1);
        var7 = __builtin_IMCE_GET_CREG((short)2);
        var8 = __builtin_IMCE_GET_CREG((short)3);
        __builtin_IMCE_STANDBY(16, 1);
        __builtin_IMCE_SEND(1, var5, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
        __builtin_IMCE_SEND(1, var6, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
        __builtin_IMCE_SEND(1, var7, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
        __builtin_IMCE_SEND(1, var8, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
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
      __builtin_IMCE_STANDBY(16, 1);
      __builtin_IMCE_SEND(1, var5, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var6, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var7, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var8, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
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
      __builtin_IMCE_STANDBY(16, 1);
      __builtin_IMCE_SEND(1, var5, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var6, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var7, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var8, 2, 0); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
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
    // generate: TensorEdge((-22, fused_scale), (22, fused_scale)), fused_scale write

    var65 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge((-22, fused_scale), (22, fused_scale)), fused_scale write
    // generate: TensorEdge((-23, fused_bias), (22, fused_bias)), fused_bias write

    var66 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge((-23, fused_bias), (22, fused_bias)), fused_bias write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 1024; i1++) { // generate : call_created_loop
      // generate: bn_standalone

      short16 var267, var367, var467;
      __builtin_IMCE_SETFLAG(1);
      var67 = __builtin_IMCE_RECV(2); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
      var267 = __builtin_IMCE_RECV(2); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
      var367 = __builtin_IMCE_RECV(2); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
      var467 = __builtin_IMCE_RECV(2); // TensorEdge((21, odata), (22, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SETFLAG(0);
      // generate: batch_norm


      var70 = __builtin_IMCE_MULTL(var67, var65, 15);
      var71 = __builtin_IMCE_MULTH(var67, var65, 15);
      var72 = __builtin_IMCE_SUBI(0, 1);
      var73 = __builtin_IMCE_SRLI(var72, 1);
      var74 = __builtin_IMCE_SRAI(var71, 15);
      var75 = __builtin_IMCE_SRAI(var70, 15);
      var76 = __builtin_IMCE_XOR(var74, var75, 15);
      var77 = __builtin_IMCE_XOR(var74, var73, 15);
      var78 = __builtin_IMCE_XOR(var76, var72, 15);
      var79 = __builtin_IMCE_AND(var76, var77, 15);
      var80 = __builtin_IMCE_AND(var78, var70, 15);
      var69 = __builtin_IMCE_OR(var79, var80, 15);
      var68 = __builtin_IMCE_ADD(var69, var66, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var68, 2, 0); // TensorEdge((22, odata), (24, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, 0, 2, 0); // TensorEdge((22, odata), (24, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, 0, 2, 0); // TensorEdge((22, odata), (24, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, 0, 2, 0); // TensorEdge((22, odata), (24, func_out0), 0), imce_3_1 -> inode_3_0
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
