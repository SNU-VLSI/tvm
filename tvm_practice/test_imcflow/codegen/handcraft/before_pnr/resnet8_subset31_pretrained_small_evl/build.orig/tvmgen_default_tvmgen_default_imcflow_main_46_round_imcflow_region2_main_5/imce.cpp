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
  short16 var21; // (TensorEdge(((69, -53), config), ((69, 63), config)), 0)
  short16 var22; // (TensorEdge(((69, -54), fused_scale), ((69, 64), fused_scale)), 0)
  short16 var23; // (TensorEdge(((69, -54), fused_scale), ((69, 64), fused_scale)), 1)
  short16 var24; // (TensorEdge(((69, -55), fused_bias), ((69, 64), fused_bias)), 0)
  short16 var25; // (TensorEdge(((69, -55), fused_bias), ((69, 64), fused_bias)), 1)
  short16 var26; // (TensorEdge(((69, -51), odata), ((69, 65), lhs)), 0)
  short16 var27; // (TensorEdge(((69, -51), odata), ((69, 65), lhs)), 1)
  short16 var28; // (TensorEdge(((69, -56), odata), ((69, 66), rhs)), 0)
  short16 var29; // (TensorEdge(((69, -56), odata), ((69, 66), rhs)), 1)
  short16 var30; // (AddBlock(gid: 66), 0)
  short16 var31; // (AddBlock(gid: 66), 1)
  short16 var32; // (AddBlock(gid: 66), 2)
  short16 var33; // (AddBlock(gid: 66), 3)
  short16 var34; // (BatchNormBlock(gid: 64), 0)
  short16 var35; // (BatchNormBlock(gid: 64), 1)
  short16 var36; // (TensorEdge(((69, -54), fused_scale), ((69, 64), fused_scale)), 2)
  short16 var37; // (TensorEdge(((69, -55), fused_bias), ((69, 64), fused_bias)), 2)
  short16 var38; // (BatchNormBlock(gid: 64), 2)
  short16 var39; // (TensorEdge(((69, -54), fused_scale), ((69, 64), fused_scale)), 3)
  short16 var40; // (TensorEdge(((69, -55), fused_bias), ((69, 64), fused_bias)), 3)
  short16 var41; // (BatchNormBlock(gid: 64), 3)
  short16 var42; // (MultlBlock(gid: 65), 0)
  short16 var43; // (MultlBlock(gid: 65), 1)
  short16 var44; // (TensorEdge(((69, -51), odata), ((69, 65), lhs)), 2)
  short16 var45; // (MultlBlock(gid: 65), 2)
  short16 var46; // (TensorEdge(((69, -51), odata), ((69, 65), lhs)), 3)
  short16 var47; // (MultlBlock(gid: 65), 3)
  short16 var48; // (TensorEdge(((69, -56), odata), ((69, 66), rhs)), 2)
  short16 var49; // (TensorEdge(((69, -56), odata), ((69, 66), rhs)), 3)
  short16 var50; // (TensorEdge((-57, min), (68, min)), 0)
  short16 var51; // (TensorEdge((-58, max), (68, max)), 0)
  short16 var52; // (TensorEdge((56, odata), (68, data)), 0)
  short16 var53; // (MinmaxQuantBlock(gid: 68), 0)
  short16 var54; // (MinmaxQuantBlock(gid: 68), 1)
  short16 var55; // (MinmaxQuantBlock(gid: 68), 2)
  short16 var56; // (MinmaxQuantBlock(gid: 68), 3)
  short16 var57; // (TensorEdge(((62, -36), config), ((62, 49), config)), 0)
  short16 var58; // (TensorEdge(((62, -37), fused_scale), ((62, 51), fused_scale)), 0)
  short16 var59; // (TensorEdge(((62, -37), fused_scale), ((62, 51), fused_scale)), 1)
  short16 var60; // (TensorEdge(((62, -38), fused_bias), ((62, 51), fused_bias)), 0)
  short16 var61; // (TensorEdge(((62, -38), fused_bias), ((62, 51), fused_bias)), 1)
  short16 var62; // (BatchNormBlock(gid: 51), 0)
  short16 var63; // (BatchNormBlock(gid: 51), 1)
  short16 var64; // (BatchNormBlock(gid: 51), 2)
  short16 var65; // (BatchNormBlock(gid: 51), 3)
  short16 var66; // (AddBlock(gid: 50), 0)
  short16 var67; // (AddBlock(gid: 50), 1)
  short16 var68; // (AddBlock(gid: 50), 2)
  short16 var69; // (AddBlock(gid: 50), 3)
  short16 var70; // (TensorEdge(((62, -37), fused_scale), ((62, 51), fused_scale)), 2)
  short16 var71; // (TensorEdge(((62, -38), fused_bias), ((62, 51), fused_bias)), 2)
  short16 var72; // (TensorEdge(((62, -37), fused_scale), ((62, 51), fused_scale)), 3)
  short16 var73; // (TensorEdge(((62, -38), fused_bias), ((62, 51), fused_bias)), 3)
  short16 var74; // (TensorEdge((-49, config), (61, config)), 0)
  short16 var75; // (TensorEdge((60, odata), (61, data), 1), 0)
  short16 var76; // (TensorEdge((60, odata), (61, data), 1), 1)
  short16 var77; // (TensorEdge((60, odata), (61, data), 1), 2)
  short16 var78; // (TensorEdge((60, odata), (61, data), 1), 3)
  short16 var79; // (TensorEdge((-46, min), (59, min)), 0)
  short16 var80; // (TensorEdge((-47, max), (59, max)), 0)
  short16 var81; // (TensorEdge(((58, 54), odata), (59, data)), 0)
  short16 var82; // (TensorEdge(((58, 54), odata), (59, data)), 1)
  short16 var83; // (TensorEdge(((58, 54), odata), (59, data)), 2)
  short16 var84; // (TensorEdge(((58, 54), odata), (59, data)), 3)
  short16 var85; // (MinmaxQuantBlock(gid: 59), 0)
  short16 var86; // (MinmaxQuantBlock(gid: 59), 1)
  short16 var87; // (MinmaxQuantBlock(gid: 59), 2)
  short16 var88; // (MinmaxQuantBlock(gid: 59), 3)
  short16 var89; // (TensorEdge(((58, -41), config), ((58, 53), config)), 0)
  short16 var90; // (TensorEdge(((58, -42), fused_scale), ((58, 54), fused_scale)), 0)
  short16 var91; // (TensorEdge(((58, -42), fused_scale), ((58, 54), fused_scale)), 1)
  short16 var92; // (TensorEdge(((58, -43), fused_bias), ((58, 54), fused_bias)), 0)
  short16 var93; // (TensorEdge(((58, -43), fused_bias), ((58, 54), fused_bias)), 1)
  short16 var94; // (BatchNormBlock(gid: 54), 0)
  short16 var95; // (BatchNormBlock(gid: 54), 1)
  short16 var96; // (BatchNormBlock(gid: 54), 2)
  short16 var97; // (BatchNormBlock(gid: 54), 3)
  short16 var98; // (TensorEdge(((58, -42), fused_scale), ((58, 54), fused_scale)), 2)
  short16 var99; // (TensorEdge(((58, -43), fused_bias), ((58, 54), fused_bias)), 2)
  short16 var100; // (TensorEdge(((58, -42), fused_scale), ((58, 54), fused_scale)), 3)
  short16 var101; // (TensorEdge(((58, -43), fused_bias), ((58, 54), fused_bias)), 3)
  short16 var102; // (TensorEdge((-44, min), (57, min)), 0)
  short16 var103; // (TensorEdge((-45, max), (57, max)), 0)
  short16 var104; // (TensorEdge((56, odata), (57, data)), 0)
  short16 var105; // (MinmaxQuantBlock(gid: 57), 0)
  short16 var106; // (MinmaxQuantBlock(gid: 57), 1)
  short16 var107; // (MinmaxQuantBlock(gid: 57), 2)
  short16 var108; // (MinmaxQuantBlock(gid: 57), 3)
  short16 var109; // (TensorEdge((-31, odata), (56, lhs)), 0)
  short16 var110; // (TensorEdge((-32, odata), (56, rhs)), 0)
  short16 var111; // (AddBlock(gid: 56), 0)
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
    // generate: TensorEdge(((69, -53), config), ((69, 63), config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge(((69, -53), config), ((69, 63), config)), config write
    // generate: TensorEdge(((69, -54), fused_scale), ((69, 64), fused_scale)), fused_scale write

    var22 = __builtin_IMCE_RECV(1);
    var23 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((69, -54), fused_scale), ((69, 64), fused_scale)), fused_scale write
    // generate: TensorEdge(((69, -55), fused_bias), ((69, 64), fused_bias)), fused_bias write

    var24 = __builtin_IMCE_RECV(1);
    var25 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((69, -55), fused_bias), ((69, 64), fused_bias)), fused_bias write
    // generate: mult const

    var26 = __builtin_IMCE_RECV(3);
    var27 = __builtin_IMCE_RECV(3);
    // endgenerate: mult const
    // generate: add const

    var28 = __builtin_IMCE_RECV(2);
    var29 = __builtin_IMCE_RECV(2);
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

    var34 = __builtin_IMCE_MULTL(var17, var22, 15);
    var34 = __builtin_IMCE_ADD(var34, var24, 15);
    var35 = __builtin_IMCE_MULTL(var18, var23, 15);
    var35 = __builtin_IMCE_ADD(var35, var25, 15);
    var38 = __builtin_IMCE_MULTL(var19, var36, 15);
    var38 = __builtin_IMCE_ADD(var38, var37, 15);
    var41 = __builtin_IMCE_MULTL(var20, var39, 15);
    var41 = __builtin_IMCE_ADD(var41, var40, 15);
    // endgenerate: batch_norm
    // generate: multl

    var42 = __builtin_IMCE_MULTL(var26, var34, 15);
    var43 = __builtin_IMCE_MULTL(var27, var35, 15);
    var45 = __builtin_IMCE_MULTL(var44, var38, 15);
    var47 = __builtin_IMCE_MULTL(var46, var41, 15);
    // endgenerate: multl
    // generate: add

    var30 = __builtin_IMCE_ADD(var28, var42, 15);
    var31 = __builtin_IMCE_ADD(var29, var43, 15);
    var32 = __builtin_IMCE_ADD(var48, var45, 15);
    var33 = __builtin_IMCE_ADD(var49, var47, 15);
    // endgenerate: add
    __builtin_IMCE_SEND(2, var30, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
    __builtin_IMCE_SEND(2, var31, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
    __builtin_IMCE_SEND(2, var32, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
    __builtin_IMCE_SEND(2, var33, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
    // endgenerate : conv exec5_row_group0_col_group0
    // endgenerate: conv exec5_row_group0_col_group0
    // generate: conv exec5_row_group0_col_group1
    for (int i1 = 0; i1 < 3; i1++) { // generate : conv exec5_row_group0_col_group1

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

      var34 = __builtin_IMCE_MULTL(var17, var22, 15);
      var34 = __builtin_IMCE_ADD(var34, var24, 15);
      var35 = __builtin_IMCE_MULTL(var18, var23, 15);
      var35 = __builtin_IMCE_ADD(var35, var25, 15);
      var38 = __builtin_IMCE_MULTL(var19, var36, 15);
      var38 = __builtin_IMCE_ADD(var38, var37, 15);
      var41 = __builtin_IMCE_MULTL(var20, var39, 15);
      var41 = __builtin_IMCE_ADD(var41, var40, 15);
      // endgenerate: batch_norm
      // generate: multl

      var42 = __builtin_IMCE_MULTL(var26, var34, 15);
      var43 = __builtin_IMCE_MULTL(var27, var35, 15);
      var45 = __builtin_IMCE_MULTL(var44, var38, 15);
      var47 = __builtin_IMCE_MULTL(var46, var41, 15);
      // endgenerate: multl
      // generate: add

      var30 = __builtin_IMCE_ADD(var28, var42, 15);
      var31 = __builtin_IMCE_ADD(var29, var43, 15);
      var32 = __builtin_IMCE_ADD(var48, var45, 15);
      var33 = __builtin_IMCE_ADD(var49, var47, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(2, var30, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
      __builtin_IMCE_SEND(2, var31, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
      __builtin_IMCE_SEND(2, var32, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
      __builtin_IMCE_SEND(2, var33, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
    } // endgenerate : conv exec5_row_group0_col_group1
    // endgenerate: conv exec5_row_group0_col_group1
    // endgenerate : conv exec5_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec5_row_group0_outer_loop(iterate row offset)
    // generate: conv exec5_row_group1_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 3; i1++) { // generate : conv exec5_row_group1_outer_loop(iterate row offset)
      // generate: conv exec5_row_group1_col_group0
      // generate : conv exec5_row_group1_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 10; i2++) { // generate : load_block
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

      var34 = __builtin_IMCE_MULTL(var17, var22, 15);
      var34 = __builtin_IMCE_ADD(var34, var24, 15);
      var35 = __builtin_IMCE_MULTL(var18, var23, 15);
      var35 = __builtin_IMCE_ADD(var35, var25, 15);
      var38 = __builtin_IMCE_MULTL(var19, var36, 15);
      var38 = __builtin_IMCE_ADD(var38, var37, 15);
      var41 = __builtin_IMCE_MULTL(var20, var39, 15);
      var41 = __builtin_IMCE_ADD(var41, var40, 15);
      // endgenerate: batch_norm
      // generate: multl

      var42 = __builtin_IMCE_MULTL(var26, var34, 15);
      var43 = __builtin_IMCE_MULTL(var27, var35, 15);
      var45 = __builtin_IMCE_MULTL(var44, var38, 15);
      var47 = __builtin_IMCE_MULTL(var46, var41, 15);
      // endgenerate: multl
      // generate: add

      var30 = __builtin_IMCE_ADD(var28, var42, 15);
      var31 = __builtin_IMCE_ADD(var29, var43, 15);
      var32 = __builtin_IMCE_ADD(var48, var45, 15);
      var33 = __builtin_IMCE_ADD(var49, var47, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(2, var30, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
      __builtin_IMCE_SEND(2, var31, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
      __builtin_IMCE_SEND(2, var32, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
      __builtin_IMCE_SEND(2, var33, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
      // endgenerate : conv exec5_row_group1_col_group0
      // endgenerate: conv exec5_row_group1_col_group0
      // generate: conv exec5_row_group1_col_group1
      for (int i2 = 0; i2 < 3; i2++) { // generate : conv exec5_row_group1_col_group1

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

        var34 = __builtin_IMCE_MULTL(var17, var22, 15);
        var34 = __builtin_IMCE_ADD(var34, var24, 15);
        var35 = __builtin_IMCE_MULTL(var18, var23, 15);
        var35 = __builtin_IMCE_ADD(var35, var25, 15);
        var38 = __builtin_IMCE_MULTL(var19, var36, 15);
        var38 = __builtin_IMCE_ADD(var38, var37, 15);
        var41 = __builtin_IMCE_MULTL(var20, var39, 15);
        var41 = __builtin_IMCE_ADD(var41, var40, 15);
        // endgenerate: batch_norm
        // generate: multl

        var42 = __builtin_IMCE_MULTL(var26, var34, 15);
        var43 = __builtin_IMCE_MULTL(var27, var35, 15);
        var45 = __builtin_IMCE_MULTL(var44, var38, 15);
        var47 = __builtin_IMCE_MULTL(var46, var41, 15);
        // endgenerate: multl
        // generate: add

        var30 = __builtin_IMCE_ADD(var28, var42, 15);
        var31 = __builtin_IMCE_ADD(var29, var43, 15);
        var32 = __builtin_IMCE_ADD(var48, var45, 15);
        var33 = __builtin_IMCE_ADD(var49, var47, 15);
        // endgenerate: add
        __builtin_IMCE_SEND(2, var30, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
        __builtin_IMCE_SEND(2, var31, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
        __builtin_IMCE_SEND(2, var32, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
        __builtin_IMCE_SEND(2, var33, 2, 0); // TensorEdge(((69, 66), odata), (70, func_out1), 1), imce_2_1 -> inode_2_0
      } // endgenerate : conv exec5_row_group1_col_group1
      // endgenerate: conv exec5_row_group1_col_group1
    } // endgenerate : conv exec5_row_group1_outer_loop(iterate row offset)
    // endgenerate: conv exec5_row_group1_outer_loop(iterate row offset)
    // generate: conv exec5_tail_loop
    for (int i1 = 0; i1 < 36; i1++) { // generate : conv exec5_tail_loop
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
    for (int i1 = 0; i1 < 64; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      var52 = __builtin_IMCE_RECV(2); // TensorEdge((56, odata), (68, data)), imce_3_4 -> imce_3_3, imce_2_2
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var52, 0, 15, 0);
      var53 = __builtin_IMCE_GET_QREG(0);
      var54 = __builtin_IMCE_GET_QREG(1);
      var55 = __builtin_IMCE_GET_QREG(2);
      var56 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      __builtin_IMCE_SEND(4, var53, 0, 0); // TensorEdge((68, odata), ((69, 63), data)), imce_2_2 -> imce_2_1
      __builtin_IMCE_SEND(4, var54, 0, 0); // TensorEdge((68, odata), ((69, 63), data)), imce_2_2 -> imce_2_1
      __builtin_IMCE_SEND(4, var55, 0, 0); // TensorEdge((68, odata), ((69, 63), data)), imce_2_2 -> imce_2_1
      __builtin_IMCE_SEND(4, var56, 0, 0); // TensorEdge((68, odata), ((69, 63), data)), imce_2_2 -> imce_2_1
      // endgenerate: min_max_quantize_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 2 && wid == 3) { // imce_2_3
    // generate: TensorEdge(((62, -36), config), ((62, 49), config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge(((62, -36), config), ((62, 49), config)), config write
    // generate: TensorEdge(((62, -37), fused_scale), ((62, 51), fused_scale)), fused_scale write

    var58 = __builtin_IMCE_RECV(1);
    var59 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((62, -37), fused_scale), ((62, 51), fused_scale)), fused_scale write
    // generate: TensorEdge(((62, -38), fused_bias), ((62, 51), fused_bias)), fused_bias write

    var60 = __builtin_IMCE_RECV(1);
    var61 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((62, -38), fused_bias), ((62, 51), fused_bias)), fused_bias write
    // generate: conv exec4
    // generate: conv exec4_row_group0_outer_loop(iterate row offset)
    // generate : conv exec4_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec4_row_group0_col_group0
    // generate : conv exec4_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 6; i1++) { // generate : load_block
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((60, odata), ((62, 49), data), 0), imce_3_1 -> imce_2_3

      } // endgenerate
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var9 = __builtin_IMCE_GET_CREG((short)0);
    var10 = __builtin_IMCE_GET_CREG((short)1);
    var11 = __builtin_IMCE_GET_CREG((short)2);
    var12 = __builtin_IMCE_GET_CREG((short)3);
    var13 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
    var14 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
    var15 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
    var16 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
    // generate: add

    var66 = __builtin_IMCE_ADD(var9, var13, 15);
    var67 = __builtin_IMCE_ADD(var10, var14, 15);
    var68 = __builtin_IMCE_ADD(var11, var15, 15);
    var69 = __builtin_IMCE_ADD(var12, var16, 15);
    // endgenerate: add
    // generate: batch_norm

    var62 = __builtin_IMCE_MULTL(var66, var58, 15);
    var62 = __builtin_IMCE_ADD(var62, var60, 15);
    var63 = __builtin_IMCE_MULTL(var67, var59, 15);
    var63 = __builtin_IMCE_ADD(var63, var61, 15);
    var64 = __builtin_IMCE_MULTL(var68, var70, 15);
    var64 = __builtin_IMCE_ADD(var64, var71, 15);
    var65 = __builtin_IMCE_MULTL(var69, var72, 15);
    var65 = __builtin_IMCE_ADD(var65, var73, 15);
    // endgenerate: batch_norm
    __builtin_IMCE_SEND(1, var62, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
    __builtin_IMCE_SEND(1, var63, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
    __builtin_IMCE_SEND(1, var64, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
    __builtin_IMCE_SEND(1, var65, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
    // endgenerate : conv exec4_row_group0_col_group0
    // endgenerate: conv exec4_row_group0_col_group0
    // generate: conv exec4_row_group0_col_group1
    for (int i1 = 0; i1 < 2; i1++) { // generate : conv exec4_row_group0_col_group1

      // generate: load_block
      // generate : load_block. loop count == 1
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((60, odata), ((62, 49), data), 0), imce_3_1 -> imce_2_3

      } // endgenerate
      // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var9 = __builtin_IMCE_GET_CREG((short)0);
      var10 = __builtin_IMCE_GET_CREG((short)1);
      var11 = __builtin_IMCE_GET_CREG((short)2);
      var12 = __builtin_IMCE_GET_CREG((short)3);
      var13 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      var14 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      var15 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      var16 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      // generate: add

      var66 = __builtin_IMCE_ADD(var9, var13, 15);
      var67 = __builtin_IMCE_ADD(var10, var14, 15);
      var68 = __builtin_IMCE_ADD(var11, var15, 15);
      var69 = __builtin_IMCE_ADD(var12, var16, 15);
      // endgenerate: add
      // generate: batch_norm

      var62 = __builtin_IMCE_MULTL(var66, var58, 15);
      var62 = __builtin_IMCE_ADD(var62, var60, 15);
      var63 = __builtin_IMCE_MULTL(var67, var59, 15);
      var63 = __builtin_IMCE_ADD(var63, var61, 15);
      var64 = __builtin_IMCE_MULTL(var68, var70, 15);
      var64 = __builtin_IMCE_ADD(var64, var71, 15);
      var65 = __builtin_IMCE_MULTL(var69, var72, 15);
      var65 = __builtin_IMCE_ADD(var65, var73, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var62, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
      __builtin_IMCE_SEND(1, var63, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
      __builtin_IMCE_SEND(1, var64, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
      __builtin_IMCE_SEND(1, var65, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
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
    var13 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
    var14 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
    var15 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
    var16 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
    // generate: add

    var66 = __builtin_IMCE_ADD(var9, var13, 15);
    var67 = __builtin_IMCE_ADD(var10, var14, 15);
    var68 = __builtin_IMCE_ADD(var11, var15, 15);
    var69 = __builtin_IMCE_ADD(var12, var16, 15);
    // endgenerate: add
    // generate: batch_norm

    var62 = __builtin_IMCE_MULTL(var66, var58, 15);
    var62 = __builtin_IMCE_ADD(var62, var60, 15);
    var63 = __builtin_IMCE_MULTL(var67, var59, 15);
    var63 = __builtin_IMCE_ADD(var63, var61, 15);
    var64 = __builtin_IMCE_MULTL(var68, var70, 15);
    var64 = __builtin_IMCE_ADD(var64, var71, 15);
    var65 = __builtin_IMCE_MULTL(var69, var72, 15);
    var65 = __builtin_IMCE_ADD(var65, var73, 15);
    // endgenerate: batch_norm
    __builtin_IMCE_SEND(1, var62, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
    __builtin_IMCE_SEND(1, var63, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
    __builtin_IMCE_SEND(1, var64, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
    __builtin_IMCE_SEND(1, var65, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
    // endgenerate : conv exec4_row_group0_col_group2
    // endgenerate: conv exec4_row_group0_col_group2
    // endgenerate : conv exec4_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec4_row_group0_outer_loop(iterate row offset)
    // generate: conv exec4_row_group1_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 2; i1++) { // generate : conv exec4_row_group1_outer_loop(iterate row offset)
      // generate: conv exec4_row_group1_col_group0
      // generate : conv exec4_row_group1_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 2; i2++) { // generate : load_block
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((60, odata), ((62, 49), data), 0), imce_3_1 -> imce_2_3

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var9 = __builtin_IMCE_GET_CREG((short)0);
      var10 = __builtin_IMCE_GET_CREG((short)1);
      var11 = __builtin_IMCE_GET_CREG((short)2);
      var12 = __builtin_IMCE_GET_CREG((short)3);
      var13 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      var14 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      var15 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      var16 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      // generate: add

      var66 = __builtin_IMCE_ADD(var9, var13, 15);
      var67 = __builtin_IMCE_ADD(var10, var14, 15);
      var68 = __builtin_IMCE_ADD(var11, var15, 15);
      var69 = __builtin_IMCE_ADD(var12, var16, 15);
      // endgenerate: add
      // generate: batch_norm

      var62 = __builtin_IMCE_MULTL(var66, var58, 15);
      var62 = __builtin_IMCE_ADD(var62, var60, 15);
      var63 = __builtin_IMCE_MULTL(var67, var59, 15);
      var63 = __builtin_IMCE_ADD(var63, var61, 15);
      var64 = __builtin_IMCE_MULTL(var68, var70, 15);
      var64 = __builtin_IMCE_ADD(var64, var71, 15);
      var65 = __builtin_IMCE_MULTL(var69, var72, 15);
      var65 = __builtin_IMCE_ADD(var65, var73, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var62, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
      __builtin_IMCE_SEND(1, var63, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
      __builtin_IMCE_SEND(1, var64, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
      __builtin_IMCE_SEND(1, var65, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
      // endgenerate : conv exec4_row_group1_col_group0
      // endgenerate: conv exec4_row_group1_col_group0
      // generate: conv exec4_row_group1_col_group1
      for (int i2 = 0; i2 < 2; i2++) { // generate : conv exec4_row_group1_col_group1

        // generate: load_block
        // generate : load_block. loop count == 1
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((60, odata), ((62, 49), data), 0), imce_3_1 -> imce_2_3

        } // endgenerate
        // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var9 = __builtin_IMCE_GET_CREG((short)0);
        var10 = __builtin_IMCE_GET_CREG((short)1);
        var11 = __builtin_IMCE_GET_CREG((short)2);
        var12 = __builtin_IMCE_GET_CREG((short)3);
        var13 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
        var14 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
        var15 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
        var16 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
        // generate: add

        var66 = __builtin_IMCE_ADD(var9, var13, 15);
        var67 = __builtin_IMCE_ADD(var10, var14, 15);
        var68 = __builtin_IMCE_ADD(var11, var15, 15);
        var69 = __builtin_IMCE_ADD(var12, var16, 15);
        // endgenerate: add
        // generate: batch_norm

        var62 = __builtin_IMCE_MULTL(var66, var58, 15);
        var62 = __builtin_IMCE_ADD(var62, var60, 15);
        var63 = __builtin_IMCE_MULTL(var67, var59, 15);
        var63 = __builtin_IMCE_ADD(var63, var61, 15);
        var64 = __builtin_IMCE_MULTL(var68, var70, 15);
        var64 = __builtin_IMCE_ADD(var64, var71, 15);
        var65 = __builtin_IMCE_MULTL(var69, var72, 15);
        var65 = __builtin_IMCE_ADD(var65, var73, 15);
        // endgenerate: batch_norm
        __builtin_IMCE_SEND(1, var62, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
        __builtin_IMCE_SEND(1, var63, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
        __builtin_IMCE_SEND(1, var64, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
        __builtin_IMCE_SEND(1, var65, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
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
      var13 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      var14 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      var15 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      var16 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      // generate: add

      var66 = __builtin_IMCE_ADD(var9, var13, 15);
      var67 = __builtin_IMCE_ADD(var10, var14, 15);
      var68 = __builtin_IMCE_ADD(var11, var15, 15);
      var69 = __builtin_IMCE_ADD(var12, var16, 15);
      // endgenerate: add
      // generate: batch_norm

      var62 = __builtin_IMCE_MULTL(var66, var58, 15);
      var62 = __builtin_IMCE_ADD(var62, var60, 15);
      var63 = __builtin_IMCE_MULTL(var67, var59, 15);
      var63 = __builtin_IMCE_ADD(var63, var61, 15);
      var64 = __builtin_IMCE_MULTL(var68, var70, 15);
      var64 = __builtin_IMCE_ADD(var64, var71, 15);
      var65 = __builtin_IMCE_MULTL(var69, var72, 15);
      var65 = __builtin_IMCE_ADD(var65, var73, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var62, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
      __builtin_IMCE_SEND(1, var63, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
      __builtin_IMCE_SEND(1, var64, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
      __builtin_IMCE_SEND(1, var65, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
      // endgenerate : conv exec4_row_group1_col_group2
      // endgenerate: conv exec4_row_group1_col_group2
    } // endgenerate : conv exec4_row_group1_outer_loop(iterate row offset)
    // endgenerate: conv exec4_row_group1_outer_loop(iterate row offset)
    // generate: conv exec4_row_group2_outer_loop(iterate row offset)
    // generate : conv exec4_row_group2_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec4_row_group2_col_group0
    for (int i1 = 0; i1 < 4; i1++) { // generate : conv exec4_row_group2_col_group0

      // generate: load_block
      // loop ignored with loop count == 0 : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var9 = __builtin_IMCE_GET_CREG((short)0);
      var10 = __builtin_IMCE_GET_CREG((short)1);
      var11 = __builtin_IMCE_GET_CREG((short)2);
      var12 = __builtin_IMCE_GET_CREG((short)3);
      var13 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      var14 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      var15 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      var16 = __builtin_IMCE_RECV(2); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      // generate: add

      var66 = __builtin_IMCE_ADD(var9, var13, 15);
      var67 = __builtin_IMCE_ADD(var10, var14, 15);
      var68 = __builtin_IMCE_ADD(var11, var15, 15);
      var69 = __builtin_IMCE_ADD(var12, var16, 15);
      // endgenerate: add
      // generate: batch_norm

      var62 = __builtin_IMCE_MULTL(var66, var58, 15);
      var62 = __builtin_IMCE_ADD(var62, var60, 15);
      var63 = __builtin_IMCE_MULTL(var67, var59, 15);
      var63 = __builtin_IMCE_ADD(var63, var61, 15);
      var64 = __builtin_IMCE_MULTL(var68, var70, 15);
      var64 = __builtin_IMCE_ADD(var64, var71, 15);
      var65 = __builtin_IMCE_MULTL(var69, var72, 15);
      var65 = __builtin_IMCE_ADD(var65, var73, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var62, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
      __builtin_IMCE_SEND(1, var63, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
      __builtin_IMCE_SEND(1, var64, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
      __builtin_IMCE_SEND(1, var65, 2, 0); // TensorEdge(((62, 51), odata), (70, func_out0), 0), imce_2_3 -> inode_3_0
    } // endgenerate : conv exec4_row_group2_col_group0
    // endgenerate: conv exec4_row_group2_col_group0
    // endgenerate : conv exec4_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec4_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec4
    __builtin_IMCE_STOP();
  }
  else if (hid == 2 && wid == 4) { // imce_2_4
    // generate: TensorEdge((-49, config), (61, config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge((-49, config), (61, config)), config write
    // generate: conv exec3
    // generate: conv exec3_row_group0_outer_loop(iterate row offset)
    // generate : conv exec3_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec3_row_group0_col_group0
    // generate : conv exec3_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 6; i1++) { // generate : load_block
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((60, odata), (61, data), 1), imce_3_1 -> imce_2_3, imce_2_4

      } // endgenerate
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var5 = __builtin_IMCE_GET_CREG((short)0);
    var6 = __builtin_IMCE_GET_CREG((short)1);
    var7 = __builtin_IMCE_GET_CREG((short)2);
    var8 = __builtin_IMCE_GET_CREG((short)3);
    __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
    __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
    __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
    __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
    // endgenerate : conv exec3_row_group0_col_group0
    // endgenerate: conv exec3_row_group0_col_group0
    // generate: conv exec3_row_group0_col_group1
    for (int i1 = 0; i1 < 2; i1++) { // generate : conv exec3_row_group0_col_group1

      // generate: load_block
      // generate : load_block. loop count == 1
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((60, odata), (61, data), 1), imce_3_1 -> imce_2_3, imce_2_4

      } // endgenerate
      // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
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
    __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
    __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
    __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
    __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
    // endgenerate : conv exec3_row_group0_col_group2
    // endgenerate: conv exec3_row_group0_col_group2
    // endgenerate : conv exec3_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec3_row_group0_outer_loop(iterate row offset)
    // generate: conv exec3_row_group1_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 2; i1++) { // generate : conv exec3_row_group1_outer_loop(iterate row offset)
      // generate: conv exec3_row_group1_col_group0
      // generate : conv exec3_row_group1_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 2; i2++) { // generate : load_block
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((60, odata), (61, data), 1), imce_3_1 -> imce_2_3, imce_2_4

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      // endgenerate : conv exec3_row_group1_col_group0
      // endgenerate: conv exec3_row_group1_col_group0
      // generate: conv exec3_row_group1_col_group1
      for (int i2 = 0; i2 < 2; i2++) { // generate : conv exec3_row_group1_col_group1

        // generate: load_block
        // generate : load_block. loop count == 1
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((60, odata), (61, data), 1), imce_3_1 -> imce_2_3, imce_2_4

        } // endgenerate
        // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var5 = __builtin_IMCE_GET_CREG((short)0);
        var6 = __builtin_IMCE_GET_CREG((short)1);
        var7 = __builtin_IMCE_GET_CREG((short)2);
        var8 = __builtin_IMCE_GET_CREG((short)3);
        __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
        __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
        __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
        __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
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
      __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      // endgenerate : conv exec3_row_group1_col_group2
      // endgenerate: conv exec3_row_group1_col_group2
    } // endgenerate : conv exec3_row_group1_outer_loop(iterate row offset)
    // endgenerate: conv exec3_row_group1_outer_loop(iterate row offset)
    // generate: conv exec3_row_group2_outer_loop(iterate row offset)
    // generate : conv exec3_row_group2_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec3_row_group2_col_group0
    for (int i1 = 0; i1 < 4; i1++) { // generate : conv exec3_row_group2_col_group0

      // generate: load_block
      // loop ignored with loop count == 0 : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
      __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((61, odata), ((62, 50), rhs)), imce_2_4 -> imce_2_3
    } // endgenerate : conv exec3_row_group2_col_group0
    // endgenerate: conv exec3_row_group2_col_group0
    // endgenerate : conv exec3_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec3_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec3
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 1) { // imce_3_1
    // generate: TensorEdge((-46, min), (59, min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge((-46, min), (59, min)), min write
    // generate: TensorEdge((-47, max), (59, max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge((-47, max), (59, max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 16; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      var81 = __builtin_IMCE_RECV(2); // TensorEdge(((58, 54), odata), (59, data)), imce_3_2 -> imce_3_1
      var82 = __builtin_IMCE_RECV(2); // TensorEdge(((58, 54), odata), (59, data)), imce_3_2 -> imce_3_1
      var83 = __builtin_IMCE_RECV(2); // TensorEdge(((58, 54), odata), (59, data)), imce_3_2 -> imce_3_1
      var84 = __builtin_IMCE_RECV(2); // TensorEdge(((58, 54), odata), (59, data)), imce_3_2 -> imce_3_1
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var81, 0, 15, 0);
      __builtin_IMCE_MM_QUANT(var82, 0, 15, 1);
      var85 = __builtin_IMCE_GET_QREG(0);
      var86 = __builtin_IMCE_GET_QREG(1);
      var87 = __builtin_IMCE_GET_QREG(2);
      var88 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      __builtin_IMCE_SEND(2, var85, 0, 0); // TensorEdge((59, odata), (60, data)), imce_3_1 -> imce_2_3
      __builtin_IMCE_SEND(2, var86, 0, 0); // TensorEdge((59, odata), (60, data)), imce_3_1 -> imce_2_3
      __builtin_IMCE_SEND(2, var87, 0, 0); // TensorEdge((59, odata), (60, data)), imce_3_1 -> imce_2_3
      __builtin_IMCE_SEND(2, var88, 0, 0); // TensorEdge((59, odata), (60, data)), imce_3_1 -> imce_2_3
      // endgenerate: min_max_quantize_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 2) { // imce_3_2
    // generate: TensorEdge(((58, -41), config), ((58, 53), config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge(((58, -41), config), ((58, 53), config)), config write
    // generate: TensorEdge(((58, -42), fused_scale), ((58, 54), fused_scale)), fused_scale write

    var90 = __builtin_IMCE_RECV(1);
    var91 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((58, -42), fused_scale), ((58, 54), fused_scale)), fused_scale write
    // generate: TensorEdge(((58, -43), fused_bias), ((58, 54), fused_bias)), fused_bias write

    var92 = __builtin_IMCE_RECV(1);
    var93 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((58, -43), fused_bias), ((58, 54), fused_bias)), fused_bias write
    // generate: conv exec2
    // generate: conv exec2_row_group0_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 4; i1++) { // generate : conv exec2_row_group0_outer_loop(iterate row offset)
      // generate: conv exec2_row_group0_col_group0
      // generate : conv exec2_row_group0_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 10; i2++) { // generate : load_block
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((57, odata), ((58, 53), data)), imce_3_3 -> imce_3_2

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var1 = __builtin_IMCE_GET_CREG((short)0);
      var2 = __builtin_IMCE_GET_CREG((short)1);
      var3 = __builtin_IMCE_GET_CREG((short)2);
      var4 = __builtin_IMCE_GET_CREG((short)3);
      // generate: batch_norm

      var94 = __builtin_IMCE_MULTL(var1, var90, 15);
      var94 = __builtin_IMCE_ADD(var94, var92, 15);
      var95 = __builtin_IMCE_MULTL(var2, var91, 15);
      var95 = __builtin_IMCE_ADD(var95, var93, 15);
      var96 = __builtin_IMCE_MULTL(var3, var98, 15);
      var96 = __builtin_IMCE_ADD(var96, var99, 15);
      var97 = __builtin_IMCE_MULTL(var4, var100, 15);
      var97 = __builtin_IMCE_ADD(var97, var101, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(2, var94, 2, 0); // TensorEdge(((58, 54), odata), (59, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var95, 2, 0); // TensorEdge(((58, 54), odata), (59, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var96, 2, 0); // TensorEdge(((58, 54), odata), (59, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var97, 2, 0); // TensorEdge(((58, 54), odata), (59, data)), imce_3_2 -> imce_3_1
      // endgenerate : conv exec2_row_group0_col_group0
      // endgenerate: conv exec2_row_group0_col_group0
      // generate: conv exec2_row_group0_col_group1
      for (int i2 = 0; i2 < 3; i2++) { // generate : conv exec2_row_group0_col_group1

        // generate: load_block
        for (int i3 = 0; i3 < 2; i3++) { // generate : load_block
          for (int i4 = 0; i4 < 4; i4++) { // generate
            __builtin_IMCE_LOAD_LB(0); // TensorEdge((57, odata), ((58, 53), data)), imce_3_3 -> imce_3_2

          } // endgenerate
        } // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var1 = __builtin_IMCE_GET_CREG((short)0);
        var2 = __builtin_IMCE_GET_CREG((short)1);
        var3 = __builtin_IMCE_GET_CREG((short)2);
        var4 = __builtin_IMCE_GET_CREG((short)3);
        // generate: batch_norm

        var94 = __builtin_IMCE_MULTL(var1, var90, 15);
        var94 = __builtin_IMCE_ADD(var94, var92, 15);
        var95 = __builtin_IMCE_MULTL(var2, var91, 15);
        var95 = __builtin_IMCE_ADD(var95, var93, 15);
        var96 = __builtin_IMCE_MULTL(var3, var98, 15);
        var96 = __builtin_IMCE_ADD(var96, var99, 15);
        var97 = __builtin_IMCE_MULTL(var4, var100, 15);
        var97 = __builtin_IMCE_ADD(var97, var101, 15);
        // endgenerate: batch_norm
        __builtin_IMCE_SEND(2, var94, 2, 0); // TensorEdge(((58, 54), odata), (59, data)), imce_3_2 -> imce_3_1
        __builtin_IMCE_SEND(2, var95, 2, 0); // TensorEdge(((58, 54), odata), (59, data)), imce_3_2 -> imce_3_1
        __builtin_IMCE_SEND(2, var96, 2, 0); // TensorEdge(((58, 54), odata), (59, data)), imce_3_2 -> imce_3_1
        __builtin_IMCE_SEND(2, var97, 2, 0); // TensorEdge(((58, 54), odata), (59, data)), imce_3_2 -> imce_3_1
      } // endgenerate : conv exec2_row_group0_col_group1
      // endgenerate: conv exec2_row_group0_col_group1
    } // endgenerate : conv exec2_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec2_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec2
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 3) { // imce_3_3
    // generate: TensorEdge((-44, min), (57, min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge((-44, min), (57, min)), min write
    // generate: TensorEdge((-45, max), (57, max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge((-45, max), (57, max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 64; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      var104 = __builtin_IMCE_RECV(2); // TensorEdge((56, odata), (57, data)), imce_3_4 -> imce_3_3
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var104, 0, 15, 0);
      var105 = __builtin_IMCE_GET_QREG(0);
      var106 = __builtin_IMCE_GET_QREG(1);
      var107 = __builtin_IMCE_GET_QREG(2);
      var108 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      __builtin_IMCE_SEND(1, var105, 0, 0); // TensorEdge((57, odata), ((58, 53), data)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(1, var106, 0, 0); // TensorEdge((57, odata), ((58, 53), data)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(1, var107, 0, 0); // TensorEdge((57, odata), ((58, 53), data)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(1, var108, 0, 0); // TensorEdge((57, odata), ((58, 53), data)), imce_3_3 -> imce_3_2
      // endgenerate: min_max_quantize_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 4) { // imce_3_4
    // generate: call_created_loop
    for (int i1 = 0; i1 < 64; i1++) { // generate : call_created_loop
      // generate: add standalone

      var109 = __builtin_IMCE_RECV(2); // TensorEdge((-31, odata), (56, lhs)), inode_0_0 -> imce_3_4
      var110 = __builtin_IMCE_RECV(3); // TensorEdge((-32, odata), (56, rhs)), inode_1_0 -> imce_3_4
      // generate: add

      var111 = __builtin_IMCE_ADD(var109, var110, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(1, var111, 2, 0); // TensorEdge((56, odata), (57, data)),TensorEdge((56, odata), (68, data)), imce_3_4 -> imce_3_3
      // endgenerate: add standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
}
