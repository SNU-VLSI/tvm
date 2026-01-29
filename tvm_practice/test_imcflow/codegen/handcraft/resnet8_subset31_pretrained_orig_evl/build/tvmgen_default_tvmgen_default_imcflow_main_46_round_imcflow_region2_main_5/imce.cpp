#include "../common_decl.h"
void tvmgen_default_tvmgen_default_imcflow_main_46_round_imcflow_region2_main_5() {
  int hid = __builtin_IMCE_GET_CORE_HID();
  int wid = __builtin_IMCE_GET_CORE_WID();
  short16 var1; // (ConvBlock(gid: 53), 0)
  short16 var2; // (ConvBlock(gid: 53), 1)
  short16 var3; // (ConvBlock(gid: 53), 2)
  short16 var4; // (ConvBlock(gid: 53), 3)
  short16 var5; // (ConvBlock(gid: 61), 0)
  short16 var6; // (ConvBlock(gid: 61), 1)
  short16 var7; // (ConvBlock(gid: 61), 2)
  short16 var8; // (ConvBlock(gid: 61), 3)
  short16 var9; // (ConvBlock(gid: 49), 0)
  short16 var10; // (ConvBlock(gid: 49), 1)
  short16 var11; // (ConvBlock(gid: 49), 2)
  short16 var12; // (ConvBlock(gid: 49), 3)
  short16 var13; // (TensorEdge((61, odata), ((62, 50), rhs)), 0)
  short16 var14; // (TensorEdge((61, odata), ((62, 50), rhs)), 1)
  short16 var15; // (TensorEdge((61, odata), ((62, 50), rhs)), 2)
  short16 var16; // (TensorEdge((61, odata), ((62, 50), rhs)), 3)
  short16 var17; // (ConvBlock(gid: 63), 0)
  short16 var18; // (ConvBlock(gid: 63), 1)
  short16 var19; // (ConvBlock(gid: 63), 2)
  short16 var20; // (ConvBlock(gid: 63), 3)
  short16 var21; // (TensorEdge(((58, -41), config), ((58, 53), config)), 0)
  short16 var22; // (TensorEdge(((58, -42), fused_scale), ((58, 54), fused_scale)), 0)
  short16 var23; // (TensorEdge(((58, -42), fused_scale), ((58, 54), fused_scale)), 1)
  short16 var24; // (TensorEdge(((58, -43), fused_bias), ((58, 54), fused_bias)), 0)
  short16 var25; // (TensorEdge(((58, -43), fused_bias), ((58, 54), fused_bias)), 1)
  short16 var26; // (BatchNormBlock(gid: 54), 0)
  short16 var27; // (BatchNormBlock(gid: 54), 1)
  short16 var28; // (BatchNormBlock(gid: 54), 2)
  short16 var29; // (BatchNormBlock(gid: 54), 3)
  short16 var30; // (TensorEdge(((58, -42), fused_scale), ((58, 54), fused_scale)), 2)
  short16 var31; // (TensorEdge(((58, -43), fused_bias), ((58, 54), fused_bias)), 2)
  short16 var32; // (TensorEdge(((58, -42), fused_scale), ((58, 54), fused_scale)), 3)
  short16 var33; // (TensorEdge(((58, -43), fused_bias), ((58, 54), fused_bias)), 3)
  short16 var34; // (TensorEdge((-44, min), (57, min)), 0)
  short16 var35; // (TensorEdge((-45, max), (57, max)), 0)
  short16 var36; // (TensorEdge((56, odata), (57, data)), 0)
  short16 var37; // (MinmaxQuantBlock(gid: 57), 0)
  short16 var38; // (MinmaxQuantBlock(gid: 57), 1)
  short16 var39; // (MinmaxQuantBlock(gid: 57), 2)
  short16 var40; // (MinmaxQuantBlock(gid: 57), 3)
  short16 var41; // (TensorEdge((-46, min), (59, min)), 0)
  short16 var42; // (TensorEdge((-47, max), (59, max)), 0)
  short16 var43; // (TensorEdge(((58, 54), odata), (59, data)), 0)
  short16 var44; // (TensorEdge(((58, 54), odata), (59, data)), 1)
  short16 var45; // (TensorEdge(((58, 54), odata), (59, data)), 2)
  short16 var46; // (TensorEdge(((58, 54), odata), (59, data)), 3)
  short16 var47; // (MinmaxQuantBlock(gid: 59), 0)
  short16 var48; // (MinmaxQuantBlock(gid: 59), 1)
  short16 var49; // (MinmaxQuantBlock(gid: 59), 2)
  short16 var50; // (MinmaxQuantBlock(gid: 59), 3)
  short16 var51; // (TensorEdge((-31, odata), (56, lhs)), 0)
  short16 var52; // (TensorEdge((-32, odata), (56, rhs)), 0)
  short16 var53; // (AddBlock(gid: 56), 0)
  short16 var54; // (TensorEdge(((69, -53), config), ((69, 63), config)), 0)
  short16 var55; // (TensorEdge(((69, -54), fused_scale), ((69, 64), fused_scale)), 0)
  short16 var56; // (TensorEdge(((69, -54), fused_scale), ((69, 64), fused_scale)), 1)
  short16 var57; // (TensorEdge(((69, -55), fused_bias), ((69, 64), fused_bias)), 0)
  short16 var58; // (TensorEdge(((69, -55), fused_bias), ((69, 64), fused_bias)), 1)
  short16 var59; // (TensorEdge(((69, -51), odata), ((69, 65), lhs)), 0)
  short16 var60; // (TensorEdge(((69, -51), odata), ((69, 65), lhs)), 1)
  short16 var61; // (TensorEdge(((69, -56), odata), ((69, 66), rhs)), 0)
  short16 var62; // (TensorEdge(((69, -56), odata), ((69, 66), rhs)), 1)
  short16 var63; // (AddBlock(gid: 66), 0)
  short16 var64; // (AddBlock(gid: 66), 1)
  short16 var65; // (AddBlock(gid: 66), 2)
  short16 var66; // (AddBlock(gid: 66), 3)
  short16 var67; // (BatchNormBlock(gid: 64), 0)
  short16 var68; // (BatchNormBlock(gid: 64), 1)
  short16 var69; // (TensorEdge(((69, -54), fused_scale), ((69, 64), fused_scale)), 2)
  short16 var70; // (TensorEdge(((69, -55), fused_bias), ((69, 64), fused_bias)), 2)
  short16 var71; // (BatchNormBlock(gid: 64), 2)
  short16 var72; // (TensorEdge(((69, -54), fused_scale), ((69, 64), fused_scale)), 3)
  short16 var73; // (TensorEdge(((69, -55), fused_bias), ((69, 64), fused_bias)), 3)
  short16 var74; // (BatchNormBlock(gid: 64), 3)
  short16 var75; // (MultlBlock(gid: 65), 0)
  short16 var76; // (MultlBlock(gid: 65), 1)
  short16 var77; // (TensorEdge(((69, -51), odata), ((69, 65), lhs)), 2)
  short16 var78; // (MultlBlock(gid: 65), 2)
  short16 var79; // (TensorEdge(((69, -51), odata), ((69, 65), lhs)), 3)
  short16 var80; // (MultlBlock(gid: 65), 3)
  short16 var81; // (TensorEdge(((69, -56), odata), ((69, 66), rhs)), 2)
  short16 var82; // (TensorEdge(((69, -56), odata), ((69, 66), rhs)), 3)
  short16 var83; // (TensorEdge((-57, min), (68, min)), 0)
  short16 var84; // (TensorEdge((-58, max), (68, max)), 0)
  short16 var85; // (TensorEdge((56, odata), (68, data)), 0)
  short16 var86; // (MinmaxQuantBlock(gid: 68), 0)
  short16 var87; // (MinmaxQuantBlock(gid: 68), 1)
  short16 var88; // (MinmaxQuantBlock(gid: 68), 2)
  short16 var89; // (MinmaxQuantBlock(gid: 68), 3)
  short16 var90; // (TensorEdge(((62, -36), config), ((62, 49), config)), 0)
  short16 var91; // (TensorEdge(((62, -37), fused_scale), ((62, 51), fused_scale)), 0)
  short16 var92; // (TensorEdge(((62, -37), fused_scale), ((62, 51), fused_scale)), 1)
  short16 var93; // (TensorEdge(((62, -38), fused_bias), ((62, 51), fused_bias)), 0)
  short16 var94; // (TensorEdge(((62, -38), fused_bias), ((62, 51), fused_bias)), 1)
  short16 var95; // (BatchNormBlock(gid: 51), 0)
  short16 var96; // (BatchNormBlock(gid: 51), 1)
  short16 var97; // (BatchNormBlock(gid: 51), 2)
  short16 var98; // (BatchNormBlock(gid: 51), 3)
  short16 var99; // (AddBlock(gid: 50), 0)
  short16 var100; // (AddBlock(gid: 50), 1)
  short16 var101; // (AddBlock(gid: 50), 2)
  short16 var102; // (AddBlock(gid: 50), 3)
  short16 var103; // (TensorEdge(((62, -37), fused_scale), ((62, 51), fused_scale)), 2)
  short16 var104; // (TensorEdge(((62, -38), fused_bias), ((62, 51), fused_bias)), 2)
  short16 var105; // (TensorEdge(((62, -37), fused_scale), ((62, 51), fused_scale)), 3)
  short16 var106; // (TensorEdge(((62, -38), fused_bias), ((62, 51), fused_bias)), 3)
  short16 var107; // (TensorEdge((-49, config), (61, config)), 0)
  short16 var108; // (TensorEdge((60, odata), (61, data), 1), 0)
  short16 var109; // (TensorEdge((60, odata), (61, data), 1), 1)
  short16 var110; // (TensorEdge((60, odata), (61, data), 1), 2)
  short16 var111; // (TensorEdge((60, odata), (61, data), 1), 3)
  if (hid == 0 && wid == 1) { // imce_0_1
    // generate: TensorEdge(((58, -41), config), ((58, 53), config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge(((58, -41), config), ((58, 53), config)), config write
    // generate: TensorEdge(((58, -42), fused_scale), ((58, 54), fused_scale)), fused_scale write

    var22 = __builtin_IMCE_RECV(1);
    var23 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((58, -42), fused_scale), ((58, 54), fused_scale)), fused_scale write
    // generate: TensorEdge(((58, -43), fused_bias), ((58, 54), fused_bias)), fused_bias write

    var24 = __builtin_IMCE_RECV(1);
    var25 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((58, -43), fused_bias), ((58, 54), fused_bias)), fused_bias write
    // generate: conv exec2
    // generate: conv exec2_row_group0_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 16; i1++) { // generate : conv exec2_row_group0_outer_loop(iterate row offset)
      // generate: conv exec2_row_group0_col_group0
      // generate : conv exec2_row_group0_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 34; i2++) { // generate : load_block
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((57, odata), ((58, 53), data)), imce_0_2 -> imce_0_1

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var1 = __builtin_IMCE_GET_CREG((short)0);
      var2 = __builtin_IMCE_GET_CREG((short)1);
      var3 = __builtin_IMCE_GET_CREG((short)2);
      var4 = __builtin_IMCE_GET_CREG((short)3);
      // generate: batch_norm

      var26 = __builtin_IMCE_MULTL(var1, var22, 15);
      var26 = __builtin_IMCE_ADD(var26, var24, 15);
      var27 = __builtin_IMCE_MULTL(var2, var23, 15);
      var27 = __builtin_IMCE_ADD(var27, var25, 15);
      var28 = __builtin_IMCE_MULTL(var3, var30, 15);
      var28 = __builtin_IMCE_ADD(var28, var31, 15);
      var29 = __builtin_IMCE_MULTL(var4, var32, 15);
      var29 = __builtin_IMCE_ADD(var29, var33, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var26, 2, 0); // TensorEdge(((58, 54), odata), (59, data)), imce_0_1 -> imce_1_1
      __builtin_IMCE_SEND(1, var27, 2, 0); // TensorEdge(((58, 54), odata), (59, data)), imce_0_1 -> imce_1_1
      __builtin_IMCE_SEND(1, var28, 2, 0); // TensorEdge(((58, 54), odata), (59, data)), imce_0_1 -> imce_1_1
      __builtin_IMCE_SEND(1, var29, 2, 0); // TensorEdge(((58, 54), odata), (59, data)), imce_0_1 -> imce_1_1
      // endgenerate : conv exec2_row_group0_col_group0
      // endgenerate: conv exec2_row_group0_col_group0
      // generate: conv exec2_row_group0_col_group1
      for (int i2 = 0; i2 < 15; i2++) { // generate : conv exec2_row_group0_col_group1

        // generate: load_block
        for (int i3 = 0; i3 < 2; i3++) { // generate : load_block
          for (int i4 = 0; i4 < 4; i4++) { // generate
            __builtin_IMCE_LOAD_LB(0); // TensorEdge((57, odata), ((58, 53), data)), imce_0_2 -> imce_0_1

          } // endgenerate
        } // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var1 = __builtin_IMCE_GET_CREG((short)0);
        var2 = __builtin_IMCE_GET_CREG((short)1);
        var3 = __builtin_IMCE_GET_CREG((short)2);
        var4 = __builtin_IMCE_GET_CREG((short)3);
        // generate: batch_norm

        var26 = __builtin_IMCE_MULTL(var1, var22, 15);
        var26 = __builtin_IMCE_ADD(var26, var24, 15);
        var27 = __builtin_IMCE_MULTL(var2, var23, 15);
        var27 = __builtin_IMCE_ADD(var27, var25, 15);
        var28 = __builtin_IMCE_MULTL(var3, var30, 15);
        var28 = __builtin_IMCE_ADD(var28, var31, 15);
        var29 = __builtin_IMCE_MULTL(var4, var32, 15);
        var29 = __builtin_IMCE_ADD(var29, var33, 15);
        // endgenerate: batch_norm
        __builtin_IMCE_SEND(1, var26, 2, 0); // TensorEdge(((58, 54), odata), (59, data)), imce_0_1 -> imce_1_1
        __builtin_IMCE_SEND(1, var27, 2, 0); // TensorEdge(((58, 54), odata), (59, data)), imce_0_1 -> imce_1_1
        __builtin_IMCE_SEND(1, var28, 2, 0); // TensorEdge(((58, 54), odata), (59, data)), imce_0_1 -> imce_1_1
        __builtin_IMCE_SEND(1, var29, 2, 0); // TensorEdge(((58, 54), odata), (59, data)), imce_0_1 -> imce_1_1
      } // endgenerate : conv exec2_row_group0_col_group1
      // endgenerate: conv exec2_row_group0_col_group1
    } // endgenerate : conv exec2_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec2_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec2
    __builtin_IMCE_STOP();
  }
  else if (hid == 0 && wid == 2) { // imce_0_2
    // generate: TensorEdge((-44, min), (57, min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge((-44, min), (57, min)), min write
    // generate: TensorEdge((-45, max), (57, max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge((-45, max), (57, max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 1024; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      var36 = __builtin_IMCE_RECV(2); // TensorEdge((56, odata), (57, data)), imce_1_2 -> imce_0_2
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var36, 0, 15, 0);
      var37 = __builtin_IMCE_GET_QREG(0);
      var38 = __builtin_IMCE_GET_QREG(1);
      var39 = __builtin_IMCE_GET_QREG(2);
      var40 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      __builtin_IMCE_SEND(1, var37, 0, 0); // TensorEdge((57, odata), ((58, 53), data)), imce_0_2 -> imce_0_1
      __builtin_IMCE_SEND(1, var38, 0, 0); // TensorEdge((57, odata), ((58, 53), data)), imce_0_2 -> imce_0_1
      __builtin_IMCE_SEND(1, var39, 0, 0); // TensorEdge((57, odata), ((58, 53), data)), imce_0_2 -> imce_0_1
      __builtin_IMCE_SEND(1, var40, 0, 0); // TensorEdge((57, odata), ((58, 53), data)), imce_0_2 -> imce_0_1
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
    // generate: TensorEdge((-46, min), (59, min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge((-46, min), (59, min)), min write
    // generate: TensorEdge((-47, max), (59, max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge((-47, max), (59, max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 256; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      var43 = __builtin_IMCE_RECV(2); // TensorEdge(((58, 54), odata), (59, data)), imce_0_1 -> imce_1_1
      var44 = __builtin_IMCE_RECV(2); // TensorEdge(((58, 54), odata), (59, data)), imce_0_1 -> imce_1_1
      var45 = __builtin_IMCE_RECV(2); // TensorEdge(((58, 54), odata), (59, data)), imce_0_1 -> imce_1_1
      var46 = __builtin_IMCE_RECV(2); // TensorEdge(((58, 54), odata), (59, data)), imce_0_1 -> imce_1_1
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var43, 0, 15, 0);
      __builtin_IMCE_MM_QUANT(var44, 0, 15, 1);
      var47 = __builtin_IMCE_GET_QREG(0);
      var48 = __builtin_IMCE_GET_QREG(1);
      var49 = __builtin_IMCE_GET_QREG(2);
      var50 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      __builtin_IMCE_SEND(1, var47, 0, 0); // TensorEdge((59, odata), (60, data)), imce_1_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var48, 0, 0); // TensorEdge((59, odata), (60, data)), imce_1_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var49, 0, 0); // TensorEdge((59, odata), (60, data)), imce_1_1 -> imce_3_1
      __builtin_IMCE_SEND(1, var50, 0, 0); // TensorEdge((59, odata), (60, data)), imce_1_1 -> imce_3_1
      // endgenerate: min_max_quantize_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 1 && wid == 2) { // imce_1_2
    // generate: call_created_loop
    for (int i1 = 0; i1 < 1024; i1++) { // generate : call_created_loop
      // generate: add standalone

      var51 = __builtin_IMCE_RECV(2); // TensorEdge((-31, odata), (56, lhs)), inode_0_0 -> imce_1_2
      var52 = __builtin_IMCE_RECV(3); // TensorEdge((-32, odata), (56, rhs)), inode_1_0 -> imce_1_2
      // generate: add

      var53 = __builtin_IMCE_ADD(var51, var52, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(1, var53, 2, 0); // TensorEdge((56, odata), (57, data)),TensorEdge((56, odata), (68, data)), imce_1_2 -> imce_0_2
      // endgenerate: add standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 1 && wid == 3) { // imce_1_3
  }
  else if (hid == 1 && wid == 4) { // imce_1_4
  }
  else if (hid == 2 && wid == 1) { // imce_2_1
    // generate: TensorEdge(((69, -53), config), ((69, 63), config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge(((69, -53), config), ((69, 63), config)), config write
    // generate: TensorEdge(((69, -54), fused_scale), ((69, 64), fused_scale)), fused_scale write

    var55 = __builtin_IMCE_RECV(1);
    var56 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((69, -54), fused_scale), ((69, 64), fused_scale)), fused_scale write
    // generate: TensorEdge(((69, -55), fused_bias), ((69, 64), fused_bias)), fused_bias write

    var57 = __builtin_IMCE_RECV(1);
    var58 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((69, -55), fused_bias), ((69, 64), fused_bias)), fused_bias write
    // generate: mult const

    var59 = __builtin_IMCE_RECV(3);
    var60 = __builtin_IMCE_RECV(3);
    // endgenerate: mult const
    // generate: add const

    var61 = __builtin_IMCE_RECV(2);
    var62 = __builtin_IMCE_RECV(2);
    // endgenerate: add const
    // generate: conv exec5
    // generate: conv exec5_row_group0_outer_loop(iterate row offset)
    // generate : conv exec5_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec5_row_group0_col_group0
    // generate : conv exec5_row_group0_col_group0. loop count == 1

    // generate: load_block
    // generate : load_block. loop count == 1
    for (int i1 = 0; i1 < 4; i1++) { // generate
      __builtin_IMCE_LOAD_LB(0); // TensorEdge((68, odata), ((69, 63), data)), imce_2_2 -> imce_2_1

    } // endgenerate
    // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var17 = __builtin_IMCE_GET_CREG((short)0);
    var18 = __builtin_IMCE_GET_CREG((short)1);
    var19 = __builtin_IMCE_GET_CREG((short)2);
    var20 = __builtin_IMCE_GET_CREG((short)3);
    // generate: batch_norm

    var67 = __builtin_IMCE_MULTL(var17, var55, 15);
    var67 = __builtin_IMCE_ADD(var67, var57, 15);
    var68 = __builtin_IMCE_MULTL(var18, var56, 15);
    var68 = __builtin_IMCE_ADD(var68, var58, 15);
    var71 = __builtin_IMCE_MULTL(var19, var69, 15);
    var71 = __builtin_IMCE_ADD(var71, var70, 15);
    var74 = __builtin_IMCE_MULTL(var20, var72, 15);
    var74 = __builtin_IMCE_ADD(var74, var73, 15);
    // endgenerate: batch_norm
    // generate: multl

    var75 = __builtin_IMCE_MULTL(var59, var67, 15);
    var76 = __builtin_IMCE_MULTL(var60, var68, 15);
    var78 = __builtin_IMCE_MULTL(var77, var71, 15);
    var80 = __builtin_IMCE_MULTL(var79, var74, 15);
    // endgenerate: multl
    // generate: add

    var63 = __builtin_IMCE_ADD(var61, var75, 15);
    var64 = __builtin_IMCE_ADD(var62, var76, 15);
    var65 = __builtin_IMCE_ADD(var81, var78, 15);
    var66 = __builtin_IMCE_ADD(var82, var80, 15);
    // endgenerate: add
    __builtin_IMCE_SEND(1, var63, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
    __builtin_IMCE_SEND(1, var64, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
    __builtin_IMCE_SEND(1, var65, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
    __builtin_IMCE_SEND(1, var66, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
    // endgenerate : conv exec5_row_group0_col_group0
    // endgenerate: conv exec5_row_group0_col_group0
    // generate: conv exec5_row_group0_col_group1
    for (int i1 = 0; i1 < 15; i1++) { // generate : conv exec5_row_group0_col_group1

      // generate: load_block
      for (int i2 = 0; i2 < 2; i2++) { // generate : load_block
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((68, odata), ((69, 63), data)), imce_2_2 -> imce_2_1

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var17 = __builtin_IMCE_GET_CREG((short)0);
      var18 = __builtin_IMCE_GET_CREG((short)1);
      var19 = __builtin_IMCE_GET_CREG((short)2);
      var20 = __builtin_IMCE_GET_CREG((short)3);
      // generate: batch_norm

      var67 = __builtin_IMCE_MULTL(var17, var55, 15);
      var67 = __builtin_IMCE_ADD(var67, var57, 15);
      var68 = __builtin_IMCE_MULTL(var18, var56, 15);
      var68 = __builtin_IMCE_ADD(var68, var58, 15);
      var71 = __builtin_IMCE_MULTL(var19, var69, 15);
      var71 = __builtin_IMCE_ADD(var71, var70, 15);
      var74 = __builtin_IMCE_MULTL(var20, var72, 15);
      var74 = __builtin_IMCE_ADD(var74, var73, 15);
      // endgenerate: batch_norm
      // generate: multl

      var75 = __builtin_IMCE_MULTL(var59, var67, 15);
      var76 = __builtin_IMCE_MULTL(var60, var68, 15);
      var78 = __builtin_IMCE_MULTL(var77, var71, 15);
      var80 = __builtin_IMCE_MULTL(var79, var74, 15);
      // endgenerate: multl
      // generate: add

      var63 = __builtin_IMCE_ADD(var61, var75, 15);
      var64 = __builtin_IMCE_ADD(var62, var76, 15);
      var65 = __builtin_IMCE_ADD(var81, var78, 15);
      var66 = __builtin_IMCE_ADD(var82, var80, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(1, var63, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
      __builtin_IMCE_SEND(1, var64, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
      __builtin_IMCE_SEND(1, var65, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
      __builtin_IMCE_SEND(1, var66, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
    } // endgenerate : conv exec5_row_group0_col_group1
    // endgenerate: conv exec5_row_group0_col_group1
    // endgenerate : conv exec5_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec5_row_group0_outer_loop(iterate row offset)
    // generate: conv exec5_row_group1_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 15; i1++) { // generate : conv exec5_row_group1_outer_loop(iterate row offset)
      // generate: conv exec5_row_group1_col_group0
      // generate : conv exec5_row_group1_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 34; i2++) { // generate : load_block
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((68, odata), ((69, 63), data)), imce_2_2 -> imce_2_1

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var17 = __builtin_IMCE_GET_CREG((short)0);
      var18 = __builtin_IMCE_GET_CREG((short)1);
      var19 = __builtin_IMCE_GET_CREG((short)2);
      var20 = __builtin_IMCE_GET_CREG((short)3);
      // generate: batch_norm

      var67 = __builtin_IMCE_MULTL(var17, var55, 15);
      var67 = __builtin_IMCE_ADD(var67, var57, 15);
      var68 = __builtin_IMCE_MULTL(var18, var56, 15);
      var68 = __builtin_IMCE_ADD(var68, var58, 15);
      var71 = __builtin_IMCE_MULTL(var19, var69, 15);
      var71 = __builtin_IMCE_ADD(var71, var70, 15);
      var74 = __builtin_IMCE_MULTL(var20, var72, 15);
      var74 = __builtin_IMCE_ADD(var74, var73, 15);
      // endgenerate: batch_norm
      // generate: multl

      var75 = __builtin_IMCE_MULTL(var59, var67, 15);
      var76 = __builtin_IMCE_MULTL(var60, var68, 15);
      var78 = __builtin_IMCE_MULTL(var77, var71, 15);
      var80 = __builtin_IMCE_MULTL(var79, var74, 15);
      // endgenerate: multl
      // generate: add

      var63 = __builtin_IMCE_ADD(var61, var75, 15);
      var64 = __builtin_IMCE_ADD(var62, var76, 15);
      var65 = __builtin_IMCE_ADD(var81, var78, 15);
      var66 = __builtin_IMCE_ADD(var82, var80, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(1, var63, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
      __builtin_IMCE_SEND(1, var64, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
      __builtin_IMCE_SEND(1, var65, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
      __builtin_IMCE_SEND(1, var66, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
      // endgenerate : conv exec5_row_group1_col_group0
      // endgenerate: conv exec5_row_group1_col_group0
      // generate: conv exec5_row_group1_col_group1
      for (int i2 = 0; i2 < 15; i2++) { // generate : conv exec5_row_group1_col_group1

        // generate: load_block
        for (int i3 = 0; i3 < 2; i3++) { // generate : load_block
          for (int i4 = 0; i4 < 4; i4++) { // generate
            __builtin_IMCE_LOAD_LB(0); // TensorEdge((68, odata), ((69, 63), data)), imce_2_2 -> imce_2_1

          } // endgenerate
        } // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var17 = __builtin_IMCE_GET_CREG((short)0);
        var18 = __builtin_IMCE_GET_CREG((short)1);
        var19 = __builtin_IMCE_GET_CREG((short)2);
        var20 = __builtin_IMCE_GET_CREG((short)3);
        // generate: batch_norm

        var67 = __builtin_IMCE_MULTL(var17, var55, 15);
        var67 = __builtin_IMCE_ADD(var67, var57, 15);
        var68 = __builtin_IMCE_MULTL(var18, var56, 15);
        var68 = __builtin_IMCE_ADD(var68, var58, 15);
        var71 = __builtin_IMCE_MULTL(var19, var69, 15);
        var71 = __builtin_IMCE_ADD(var71, var70, 15);
        var74 = __builtin_IMCE_MULTL(var20, var72, 15);
        var74 = __builtin_IMCE_ADD(var74, var73, 15);
        // endgenerate: batch_norm
        // generate: multl

        var75 = __builtin_IMCE_MULTL(var59, var67, 15);
        var76 = __builtin_IMCE_MULTL(var60, var68, 15);
        var78 = __builtin_IMCE_MULTL(var77, var71, 15);
        var80 = __builtin_IMCE_MULTL(var79, var74, 15);
        // endgenerate: multl
        // generate: add

        var63 = __builtin_IMCE_ADD(var61, var75, 15);
        var64 = __builtin_IMCE_ADD(var62, var76, 15);
        var65 = __builtin_IMCE_ADD(var81, var78, 15);
        var66 = __builtin_IMCE_ADD(var82, var80, 15);
        // endgenerate: add
        __builtin_IMCE_SEND(1, var63, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
        __builtin_IMCE_SEND(1, var64, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
        __builtin_IMCE_SEND(1, var65, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
        __builtin_IMCE_SEND(1, var66, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
      } // endgenerate : conv exec5_row_group1_col_group1
      // endgenerate: conv exec5_row_group1_col_group1
    } // endgenerate : conv exec5_row_group1_outer_loop(iterate row offset)
    // endgenerate: conv exec5_row_group1_outer_loop(iterate row offset)
    // generate: conv exec5_tail_loop
    for (int i1 = 0; i1 < 132; i1++) { // generate : conv exec5_tail_loop
      __builtin_IMCE_RECV(0);
    } // endgenerate : conv exec5_tail_loop
    // endgenerate: conv exec5_tail_loop
    // endgenerate: conv exec5
    __builtin_IMCE_STOP();
  }
  else if (hid == 2 && wid == 2) { // imce_2_2
    // generate: TensorEdge((-57, min), (68, min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge((-57, min), (68, min)), min write
    // generate: TensorEdge((-58, max), (68, max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge((-58, max), (68, max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 1024; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      var85 = __builtin_IMCE_RECV(2); // TensorEdge((56, odata), (68, data)), imce_1_2 -> imce_2_2
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var85, 0, 15, 0);
      var86 = __builtin_IMCE_GET_QREG(0);
      var87 = __builtin_IMCE_GET_QREG(1);
      var88 = __builtin_IMCE_GET_QREG(2);
      var89 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      __builtin_IMCE_SEND(3, var86, 0, 0); // TensorEdge((68, odata), ((69, 63), data)), imce_2_2 -> imce_2_1
      __builtin_IMCE_SEND(3, var87, 0, 0); // TensorEdge((68, odata), ((69, 63), data)), imce_2_2 -> imce_2_1
      __builtin_IMCE_SEND(3, var88, 0, 0); // TensorEdge((68, odata), ((69, 63), data)), imce_2_2 -> imce_2_1
      __builtin_IMCE_SEND(3, var89, 0, 0); // TensorEdge((68, odata), ((69, 63), data)), imce_2_2 -> imce_2_1
      // endgenerate: min_max_quantize_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 2 && wid == 3) { // imce_2_3
  }
  else if (hid == 2 && wid == 4) { // imce_2_4
  }
  else if (hid == 3 && wid == 1) { // imce_3_1
    // generate: TensorEdge(((62, -36), config), ((62, 49), config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge(((62, -36), config), ((62, 49), config)), config write
    // generate: TensorEdge(((62, -37), fused_scale), ((62, 51), fused_scale)), fused_scale write

    var91 = __builtin_IMCE_RECV(1);
    var92 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((62, -37), fused_scale), ((62, 51), fused_scale)), fused_scale write
    // generate: TensorEdge(((62, -38), fused_bias), ((62, 51), fused_bias)), fused_bias write

    var93 = __builtin_IMCE_RECV(1);
    var94 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((62, -38), fused_bias), ((62, 51), fused_bias)), fused_bias write
    // generate: conv exec4
    // generate: conv exec4_row_group0_outer_loop(iterate row offset)
    // generate : conv exec4_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec4_row_group0_col_group0
    // generate : conv exec4_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 18; i1++) { // generate : load_block
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((60, odata), ((62, 49), data), 0), imce_1_1 -> imce_3_1

      } // endgenerate
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var9 = __builtin_IMCE_GET_CREG((short)0);
    var10 = __builtin_IMCE_GET_CREG((short)1);
    var11 = __builtin_IMCE_GET_CREG((short)2);
    var12 = __builtin_IMCE_GET_CREG((short)3);
    // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    var13 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    var14 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    var15 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    var16 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    // generate: add

    var99 = __builtin_IMCE_ADD(var9, var13, 15);
    var100 = __builtin_IMCE_ADD(var10, var14, 15);
    var101 = __builtin_IMCE_ADD(var11, var15, 15);
    var102 = __builtin_IMCE_ADD(var12, var16, 15);
    // endgenerate: add
    // generate: batch_norm

    var95 = __builtin_IMCE_MULTL(var99, var91, 15);
    var95 = __builtin_IMCE_ADD(var95, var93, 15);
    var96 = __builtin_IMCE_MULTL(var100, var92, 15);
    var96 = __builtin_IMCE_ADD(var96, var94, 15);
    var97 = __builtin_IMCE_MULTL(var101, var103, 15);
    var97 = __builtin_IMCE_ADD(var97, var104, 15);
    var98 = __builtin_IMCE_MULTL(var102, var105, 15);
    var98 = __builtin_IMCE_ADD(var98, var106, 15);
    // endgenerate: batch_norm
    __builtin_IMCE_SEND(1, var95, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var96, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var97, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var98, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
    // endgenerate : conv exec4_row_group0_col_group0
    // endgenerate: conv exec4_row_group0_col_group0
    // generate: conv exec4_row_group0_col_group1
    for (int i1 = 0; i1 < 14; i1++) { // generate : conv exec4_row_group0_col_group1

      // generate: load_block
      // generate : load_block. loop count == 1
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((60, odata), ((62, 49), data), 0), imce_1_1 -> imce_3_1

      } // endgenerate
      // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var9 = __builtin_IMCE_GET_CREG((short)0);
      var10 = __builtin_IMCE_GET_CREG((short)1);
      var11 = __builtin_IMCE_GET_CREG((short)2);
      var12 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      var13 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      var14 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      var15 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      var16 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // generate: add

      var99 = __builtin_IMCE_ADD(var9, var13, 15);
      var100 = __builtin_IMCE_ADD(var10, var14, 15);
      var101 = __builtin_IMCE_ADD(var11, var15, 15);
      var102 = __builtin_IMCE_ADD(var12, var16, 15);
      // endgenerate: add
      // generate: batch_norm

      var95 = __builtin_IMCE_MULTL(var99, var91, 15);
      var95 = __builtin_IMCE_ADD(var95, var93, 15);
      var96 = __builtin_IMCE_MULTL(var100, var92, 15);
      var96 = __builtin_IMCE_ADD(var96, var94, 15);
      var97 = __builtin_IMCE_MULTL(var101, var103, 15);
      var97 = __builtin_IMCE_ADD(var97, var104, 15);
      var98 = __builtin_IMCE_MULTL(var102, var105, 15);
      var98 = __builtin_IMCE_ADD(var98, var106, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var95, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var96, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var97, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var98, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
    } // endgenerate : conv exec4_row_group0_col_group1
    // endgenerate: conv exec4_row_group0_col_group1
    // generate: conv exec4_row_group0_col_group2
    // generate : conv exec4_row_group0_col_group2. loop count == 1

    // generate: load_block
    // loop ignored with loop count == 0 : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var9 = __builtin_IMCE_GET_CREG((short)0);
    var10 = __builtin_IMCE_GET_CREG((short)1);
    var11 = __builtin_IMCE_GET_CREG((short)2);
    var12 = __builtin_IMCE_GET_CREG((short)3);
    // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    var13 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    var14 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    var15 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    var16 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    // generate: add

    var99 = __builtin_IMCE_ADD(var9, var13, 15);
    var100 = __builtin_IMCE_ADD(var10, var14, 15);
    var101 = __builtin_IMCE_ADD(var11, var15, 15);
    var102 = __builtin_IMCE_ADD(var12, var16, 15);
    // endgenerate: add
    // generate: batch_norm

    var95 = __builtin_IMCE_MULTL(var99, var91, 15);
    var95 = __builtin_IMCE_ADD(var95, var93, 15);
    var96 = __builtin_IMCE_MULTL(var100, var92, 15);
    var96 = __builtin_IMCE_ADD(var96, var94, 15);
    var97 = __builtin_IMCE_MULTL(var101, var103, 15);
    var97 = __builtin_IMCE_ADD(var97, var104, 15);
    var98 = __builtin_IMCE_MULTL(var102, var105, 15);
    var98 = __builtin_IMCE_ADD(var98, var106, 15);
    // endgenerate: batch_norm
    __builtin_IMCE_SEND(1, var95, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var96, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var97, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var98, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
    // endgenerate : conv exec4_row_group0_col_group2
    // endgenerate: conv exec4_row_group0_col_group2
    // endgenerate : conv exec4_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec4_row_group0_outer_loop(iterate row offset)
    // generate: conv exec4_row_group1_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 14; i1++) { // generate : conv exec4_row_group1_outer_loop(iterate row offset)
      // generate: conv exec4_row_group1_col_group0
      // generate : conv exec4_row_group1_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 2; i2++) { // generate : load_block
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((60, odata), ((62, 49), data), 0), imce_1_1 -> imce_3_1

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var9 = __builtin_IMCE_GET_CREG((short)0);
      var10 = __builtin_IMCE_GET_CREG((short)1);
      var11 = __builtin_IMCE_GET_CREG((short)2);
      var12 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      var13 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      var14 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      var15 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      var16 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // generate: add

      var99 = __builtin_IMCE_ADD(var9, var13, 15);
      var100 = __builtin_IMCE_ADD(var10, var14, 15);
      var101 = __builtin_IMCE_ADD(var11, var15, 15);
      var102 = __builtin_IMCE_ADD(var12, var16, 15);
      // endgenerate: add
      // generate: batch_norm

      var95 = __builtin_IMCE_MULTL(var99, var91, 15);
      var95 = __builtin_IMCE_ADD(var95, var93, 15);
      var96 = __builtin_IMCE_MULTL(var100, var92, 15);
      var96 = __builtin_IMCE_ADD(var96, var94, 15);
      var97 = __builtin_IMCE_MULTL(var101, var103, 15);
      var97 = __builtin_IMCE_ADD(var97, var104, 15);
      var98 = __builtin_IMCE_MULTL(var102, var105, 15);
      var98 = __builtin_IMCE_ADD(var98, var106, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var95, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var96, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var97, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var98, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
      // endgenerate : conv exec4_row_group1_col_group0
      // endgenerate: conv exec4_row_group1_col_group0
      // generate: conv exec4_row_group1_col_group1
      for (int i2 = 0; i2 < 14; i2++) { // generate : conv exec4_row_group1_col_group1

        // generate: load_block
        // generate : load_block. loop count == 1
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((60, odata), ((62, 49), data), 0), imce_1_1 -> imce_3_1

        } // endgenerate
        // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var9 = __builtin_IMCE_GET_CREG((short)0);
        var10 = __builtin_IMCE_GET_CREG((short)1);
        var11 = __builtin_IMCE_GET_CREG((short)2);
        var12 = __builtin_IMCE_GET_CREG((short)3);
        // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
        var13 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
        // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
        // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
        var14 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
        // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
        // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
        var15 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
        // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
        // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
        var16 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
        // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
        // generate: add

        var99 = __builtin_IMCE_ADD(var9, var13, 15);
        var100 = __builtin_IMCE_ADD(var10, var14, 15);
        var101 = __builtin_IMCE_ADD(var11, var15, 15);
        var102 = __builtin_IMCE_ADD(var12, var16, 15);
        // endgenerate: add
        // generate: batch_norm

        var95 = __builtin_IMCE_MULTL(var99, var91, 15);
        var95 = __builtin_IMCE_ADD(var95, var93, 15);
        var96 = __builtin_IMCE_MULTL(var100, var92, 15);
        var96 = __builtin_IMCE_ADD(var96, var94, 15);
        var97 = __builtin_IMCE_MULTL(var101, var103, 15);
        var97 = __builtin_IMCE_ADD(var97, var104, 15);
        var98 = __builtin_IMCE_MULTL(var102, var105, 15);
        var98 = __builtin_IMCE_ADD(var98, var106, 15);
        // endgenerate: batch_norm
        __builtin_IMCE_SEND(1, var95, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
        __builtin_IMCE_SEND(1, var96, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
        __builtin_IMCE_SEND(1, var97, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
        __builtin_IMCE_SEND(1, var98, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
      } // endgenerate : conv exec4_row_group1_col_group1
      // endgenerate: conv exec4_row_group1_col_group1
      // generate: conv exec4_row_group1_col_group2
      // generate : conv exec4_row_group1_col_group2. loop count == 1

      // generate: load_block
      // loop ignored with loop count == 0 : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var9 = __builtin_IMCE_GET_CREG((short)0);
      var10 = __builtin_IMCE_GET_CREG((short)1);
      var11 = __builtin_IMCE_GET_CREG((short)2);
      var12 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      var13 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      var14 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      var15 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      var16 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // generate: add

      var99 = __builtin_IMCE_ADD(var9, var13, 15);
      var100 = __builtin_IMCE_ADD(var10, var14, 15);
      var101 = __builtin_IMCE_ADD(var11, var15, 15);
      var102 = __builtin_IMCE_ADD(var12, var16, 15);
      // endgenerate: add
      // generate: batch_norm

      var95 = __builtin_IMCE_MULTL(var99, var91, 15);
      var95 = __builtin_IMCE_ADD(var95, var93, 15);
      var96 = __builtin_IMCE_MULTL(var100, var92, 15);
      var96 = __builtin_IMCE_ADD(var96, var94, 15);
      var97 = __builtin_IMCE_MULTL(var101, var103, 15);
      var97 = __builtin_IMCE_ADD(var97, var104, 15);
      var98 = __builtin_IMCE_MULTL(var102, var105, 15);
      var98 = __builtin_IMCE_ADD(var98, var106, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var95, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var96, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var97, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var98, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
      // endgenerate : conv exec4_row_group1_col_group2
      // endgenerate: conv exec4_row_group1_col_group2
    } // endgenerate : conv exec4_row_group1_outer_loop(iterate row offset)
    // endgenerate: conv exec4_row_group1_outer_loop(iterate row offset)
    // generate: conv exec4_row_group2_outer_loop(iterate row offset)
    // generate : conv exec4_row_group2_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec4_row_group2_col_group0
    for (int i1 = 0; i1 < 16; i1++) { // generate : conv exec4_row_group2_col_group0

      // generate: load_block
      // loop ignored with loop count == 0 : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var9 = __builtin_IMCE_GET_CREG((short)0);
      var10 = __builtin_IMCE_GET_CREG((short)1);
      var11 = __builtin_IMCE_GET_CREG((short)2);
      var12 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      var13 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      var14 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      var15 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // generate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      var16 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // generate: add

      var99 = __builtin_IMCE_ADD(var9, var13, 15);
      var100 = __builtin_IMCE_ADD(var10, var14, 15);
      var101 = __builtin_IMCE_ADD(var11, var15, 15);
      var102 = __builtin_IMCE_ADD(var12, var16, 15);
      // endgenerate: add
      // generate: batch_norm

      var95 = __builtin_IMCE_MULTL(var99, var91, 15);
      var95 = __builtin_IMCE_ADD(var95, var93, 15);
      var96 = __builtin_IMCE_MULTL(var100, var92, 15);
      var96 = __builtin_IMCE_ADD(var96, var94, 15);
      var97 = __builtin_IMCE_MULTL(var101, var103, 15);
      var97 = __builtin_IMCE_ADD(var97, var104, 15);
      var98 = __builtin_IMCE_MULTL(var102, var105, 15);
      var98 = __builtin_IMCE_ADD(var98, var106, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var95, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var96, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var97, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var98, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
    } // endgenerate : conv exec4_row_group2_col_group0
    // endgenerate: conv exec4_row_group2_col_group0
    // endgenerate : conv exec4_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec4_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec4
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 2) { // imce_3_2
    // generate: TensorEdge((-49, config), (61, config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge((-49, config), (61, config)), config write
    // generate: conv exec3
    // generate: conv exec3_row_group0_outer_loop(iterate row offset)
    // generate : conv exec3_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec3_row_group0_col_group0
    // generate : conv exec3_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 18; i1++) { // generate : load_block
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((60, odata), (61, data), 1), imce_1_1 -> imce_3_2

      } // endgenerate
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var5 = __builtin_IMCE_GET_CREG((short)0);
    var6 = __builtin_IMCE_GET_CREG((short)1);
    var7 = __builtin_IMCE_GET_CREG((short)2);
    var8 = __builtin_IMCE_GET_CREG((short)3);
    __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    // endgenerate : conv exec3_row_group0_col_group0
    // endgenerate: conv exec3_row_group0_col_group0
    // generate: conv exec3_row_group0_col_group1
    for (int i1 = 0; i1 < 14; i1++) { // generate : conv exec3_row_group0_col_group1

      // generate: load_block
      // generate : load_block. loop count == 1
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((60, odata), (61, data), 1), imce_1_1 -> imce_3_2

      } // endgenerate
      // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    } // endgenerate : conv exec3_row_group0_col_group1
    // endgenerate: conv exec3_row_group0_col_group1
    // generate: conv exec3_row_group0_col_group2
    // generate : conv exec3_row_group0_col_group2. loop count == 1

    // generate: load_block
    // loop ignored with loop count == 0 : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var5 = __builtin_IMCE_GET_CREG((short)0);
    var6 = __builtin_IMCE_GET_CREG((short)1);
    var7 = __builtin_IMCE_GET_CREG((short)2);
    var8 = __builtin_IMCE_GET_CREG((short)3);
    __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    // endgenerate : conv exec3_row_group0_col_group2
    // endgenerate: conv exec3_row_group0_col_group2
    // endgenerate : conv exec3_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec3_row_group0_outer_loop(iterate row offset)
    // generate: conv exec3_row_group1_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 14; i1++) { // generate : conv exec3_row_group1_outer_loop(iterate row offset)
      // generate: conv exec3_row_group1_col_group0
      // generate : conv exec3_row_group1_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 2; i2++) { // generate : load_block
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((60, odata), (61, data), 1), imce_1_1 -> imce_3_2

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // endgenerate : conv exec3_row_group1_col_group0
      // endgenerate: conv exec3_row_group1_col_group0
      // generate: conv exec3_row_group1_col_group1
      for (int i2 = 0; i2 < 14; i2++) { // generate : conv exec3_row_group1_col_group1

        // generate: load_block
        // generate : load_block. loop count == 1
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((60, odata), (61, data), 1), imce_1_1 -> imce_3_2

        } // endgenerate
        // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var5 = __builtin_IMCE_GET_CREG((short)0);
        var6 = __builtin_IMCE_GET_CREG((short)1);
        var7 = __builtin_IMCE_GET_CREG((short)2);
        var8 = __builtin_IMCE_GET_CREG((short)3);
        __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
        __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
        __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
        __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      } // endgenerate : conv exec3_row_group1_col_group1
      // endgenerate: conv exec3_row_group1_col_group1
      // generate: conv exec3_row_group1_col_group2
      // generate : conv exec3_row_group1_col_group2. loop count == 1

      // generate: load_block
      // loop ignored with loop count == 0 : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      // endgenerate : conv exec3_row_group1_col_group2
      // endgenerate: conv exec3_row_group1_col_group2
    } // endgenerate : conv exec3_row_group1_outer_loop(iterate row offset)
    // endgenerate: conv exec3_row_group1_outer_loop(iterate row offset)
    // generate: conv exec3_row_group2_outer_loop(iterate row offset)
    // generate : conv exec3_row_group2_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec3_row_group2_col_group0
    for (int i1 = 0; i1 < 16; i1++) { // generate : conv exec3_row_group2_col_group0

      // generate: load_block
      // loop ignored with loop count == 0 : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_3_2 -> imce_3_1
    } // endgenerate : conv exec3_row_group2_col_group0
    // endgenerate: conv exec3_row_group2_col_group0
    // endgenerate : conv exec3_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec3_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec3
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 3) { // imce_3_3
  }
  else if (hid == 3 && wid == 4) { // imce_3_4
  }
}
