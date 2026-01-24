#include "../common_decl.h"
void tvmgen_default_tvmgen_default_imcflow_main_46_round_imcflow_region3_main_12() {
  int hid = __builtin_IMCE_GET_CORE_HID();
  int wid = __builtin_IMCE_GET_CORE_WID();
  short16 var1; // (ConvBlock(gid: 88), 0)
  short16 var2; // (ConvBlock(gid: 88), 1)
  short16 var3; // (ConvBlock(gid: 88), 2)
  short16 var4; // (ConvBlock(gid: 88), 3)
  short16 var5; // (ConvBlock(gid: 81), 0)
  short16 var6; // (ConvBlock(gid: 81), 1)
  short16 var7; // (ConvBlock(gid: 81), 2)
  short16 var8; // (ConvBlock(gid: 81), 3)
  short16 var9; // (TensorEdge((88, odata), ((89, 82), rhs)), 0)
  short16 var10; // (TensorEdge((88, odata), ((89, 82), rhs)), 1)
  short16 var11; // (TensorEdge((88, odata), ((89, 82), rhs)), 2)
  short16 var12; // (TensorEdge((88, odata), ((89, 82), rhs)), 3)
  short16 var13; // (ConvBlock(gid: 95), 0)
  short16 var14; // (ConvBlock(gid: 95), 1)
  short16 var15; // (ConvBlock(gid: 95), 2)
  short16 var16; // (ConvBlock(gid: 95), 3)
  short16 var17; // (ConvBlock(gid: 92), 0)
  short16 var18; // (ConvBlock(gid: 92), 1)
  short16 var19; // (ConvBlock(gid: 92), 2)
  short16 var20; // (ConvBlock(gid: 92), 3)
  short16 var21; // (TensorEdge((95, odata), ((96, 93), rhs)), 0)
  short16 var22; // (TensorEdge((95, odata), ((96, 93), rhs)), 1)
  short16 var23; // (TensorEdge((95, odata), ((96, 93), rhs)), 2)
  short16 var24; // (TensorEdge((95, odata), ((96, 93), rhs)), 3)
  short16 var25; // (ConvBlock(gid: 77), 0)
  short16 var26; // (ConvBlock(gid: 77), 1)
  short16 var27; // (ConvBlock(gid: 77), 2)
  short16 var28; // (ConvBlock(gid: 77), 3)
  short16 var29; // (TensorEdge(((96, 93), odata), ((97, 78), lhs)), 0)
  short16 var30; // (TensorEdge(((96, 93), odata), ((97, 78), lhs)), 1)
  short16 var31; // (TensorEdge(((96, 93), odata), ((97, 78), lhs)), 2)
  short16 var32; // (TensorEdge(((96, 93), odata), ((97, 78), lhs)), 3)
  short16 var33; // (ConvBlock(gid: 98), 0)
  short16 var34; // (ConvBlock(gid: 98), 1)
  short16 var35; // (ConvBlock(gid: 98), 2)
  short16 var36; // (ConvBlock(gid: 98), 3)
  short16 var37; // (TensorEdge(((104, -90), config), ((104, 98), config)), 0)
  short16 var38; // (TensorEdge(((104, -91), fused_scale), ((104, 99), fused_scale)), 0)
  short16 var39; // (TensorEdge(((104, -91), fused_scale), ((104, 99), fused_scale)), 1)
  short16 var40; // (TensorEdge(((104, -91), fused_scale), ((104, 99), fused_scale)), 2)
  short16 var41; // (TensorEdge(((104, -91), fused_scale), ((104, 99), fused_scale)), 3)
  short16 var42; // (TensorEdge(((104, -92), fused_bias), ((104, 99), fused_bias)), 0)
  short16 var43; // (TensorEdge(((104, -92), fused_bias), ((104, 99), fused_bias)), 1)
  short16 var44; // (TensorEdge(((104, -92), fused_bias), ((104, 99), fused_bias)), 2)
  short16 var45; // (TensorEdge(((104, -92), fused_bias), ((104, 99), fused_bias)), 3)
  short16 var46; // (TensorEdge(((104, -88), odata), ((104, 100), lhs)), 0)
  short16 var47; // (TensorEdge(((104, -88), odata), ((104, 100), lhs)), 1)
  short16 var48; // (TensorEdge(((104, -88), odata), ((104, 100), lhs)), 2)
  short16 var49; // (TensorEdge(((104, -88), odata), ((104, 100), lhs)), 3)
  short16 var50; // (TensorEdge(((104, -93), odata), ((104, 101), rhs)), 0)
  short16 var51; // (TensorEdge(((104, -93), odata), ((104, 101), rhs)), 1)
  short16 var52; // (TensorEdge(((104, -93), odata), ((104, 101), rhs)), 2)
  short16 var53; // (TensorEdge(((104, -93), odata), ((104, 101), rhs)), 3)
  short16 var54; // (AddBlock(gid: 101), 0)
  short16 var55; // (AddBlock(gid: 101), 1)
  short16 var56; // (AddBlock(gid: 101), 2)
  short16 var57; // (AddBlock(gid: 101), 3)
  short16 var58; // (BatchNormBlock(gid: 99), 0)
  short16 var59; // (BatchNormBlock(gid: 99), 1)
  short16 var60; // (BatchNormBlock(gid: 99), 2)
  short16 var61; // (BatchNormBlock(gid: 99), 3)
  short16 var62; // (MultlBlock(gid: 100), 0)
  short16 var63; // (MultlBlock(gid: 100), 1)
  short16 var64; // (MultlBlock(gid: 100), 2)
  short16 var65; // (MultlBlock(gid: 100), 3)
  short16 var66; // (TensorEdge((-94, min), (103, min)), 0)
  short16 var67; // (TensorEdge((-95, max), (103, max)), 0)
  short16 var68; // (TensorEdge((85, odata), (103, data)), 0)
  short16 var69; // (TensorEdge((85, odata), (103, data)), 1)
  short16 var70; // (MinmaxQuantBlock(gid: 103), 0)
  short16 var71; // (MinmaxQuantBlock(gid: 103), 1)
  short16 var72; // (MinmaxQuantBlock(gid: 103), 2)
  short16 var73; // (MinmaxQuantBlock(gid: 103), 3)
  short16 var74; // (TensorEdge(((97, -66), config), ((97, 77), config)), 0)
  short16 var75; // (TensorEdge(((97, -67), fused_scale), ((97, 79), fused_scale)), 0)
  short16 var76; // (TensorEdge(((97, -67), fused_scale), ((97, 79), fused_scale)), 1)
  short16 var77; // (TensorEdge(((97, -67), fused_scale), ((97, 79), fused_scale)), 2)
  short16 var78; // (TensorEdge(((97, -67), fused_scale), ((97, 79), fused_scale)), 3)
  short16 var79; // (TensorEdge(((97, -68), fused_bias), ((97, 79), fused_bias)), 0)
  short16 var80; // (TensorEdge(((97, -68), fused_bias), ((97, 79), fused_bias)), 1)
  short16 var81; // (TensorEdge(((97, -68), fused_bias), ((97, 79), fused_bias)), 2)
  short16 var82; // (TensorEdge(((97, -68), fused_bias), ((97, 79), fused_bias)), 3)
  short16 var83; // (BatchNormBlock(gid: 79), 0)
  short16 var84; // (BatchNormBlock(gid: 79), 1)
  short16 var85; // (BatchNormBlock(gid: 79), 2)
  short16 var86; // (BatchNormBlock(gid: 79), 3)
  short16 var87; // (AddBlock(gid: 78), 0)
  short16 var88; // (AddBlock(gid: 78), 1)
  short16 var89; // (AddBlock(gid: 78), 2)
  short16 var90; // (AddBlock(gid: 78), 3)
  short16 var91; // (TensorEdge(((96, -84), config), ((96, 92), config)), 0)
  short16 var92; // (AddBlock(gid: 93), 0)
  short16 var93; // (AddBlock(gid: 93), 1)
  short16 var94; // (AddBlock(gid: 93), 2)
  short16 var95; // (AddBlock(gid: 93), 3)
  short16 var96; // (TensorEdge((-86, config), (95, config)), 0)
  short16 var97; // (TensorEdge((91, odata), (95, data), 1), 0)
  short16 var98; // (TensorEdge((91, odata), (95, data), 1), 1)
  short16 var99; // (TensorEdge((91, odata), (95, data), 1), 2)
  short16 var100; // (TensorEdge((91, odata), (95, data), 1), 3)
  short16 var101; // (TensorEdge((-79, min), (90, min)), 0)
  short16 var102; // (TensorEdge((-80, max), (90, max)), 0)
  short16 var103; // (TensorEdge(((89, 83), odata), (90, data)), 0)
  short16 var104; // (TensorEdge(((89, 83), odata), (90, data)), 1)
  short16 var105; // (TensorEdge(((89, 83), odata), (90, data)), 2)
  short16 var106; // (TensorEdge(((89, 83), odata), (90, data)), 3)
  short16 var107; // (MinmaxQuantBlock(gid: 90), 0)
  short16 var108; // (MinmaxQuantBlock(gid: 90), 1)
  short16 var109; // (MinmaxQuantBlock(gid: 90), 2)
  short16 var110; // (MinmaxQuantBlock(gid: 90), 3)
  short16 var111; // (TensorEdge(((89, -72), config), ((89, 81), config)), 0)
  short16 var112; // (TensorEdge(((89, -73), fused_scale), ((89, 83), fused_scale)), 0)
  short16 var113; // (TensorEdge(((89, -73), fused_scale), ((89, 83), fused_scale)), 1)
  short16 var114; // (TensorEdge(((89, -73), fused_scale), ((89, 83), fused_scale)), 2)
  short16 var115; // (TensorEdge(((89, -73), fused_scale), ((89, 83), fused_scale)), 3)
  short16 var116; // (TensorEdge(((89, -74), fused_bias), ((89, 83), fused_bias)), 0)
  short16 var117; // (TensorEdge(((89, -74), fused_bias), ((89, 83), fused_bias)), 1)
  short16 var118; // (TensorEdge(((89, -74), fused_bias), ((89, 83), fused_bias)), 2)
  short16 var119; // (TensorEdge(((89, -74), fused_bias), ((89, 83), fused_bias)), 3)
  short16 var120; // (BatchNormBlock(gid: 83), 0)
  short16 var121; // (BatchNormBlock(gid: 83), 1)
  short16 var122; // (BatchNormBlock(gid: 83), 2)
  short16 var123; // (BatchNormBlock(gid: 83), 3)
  short16 var124; // (AddBlock(gid: 82), 0)
  short16 var125; // (AddBlock(gid: 82), 1)
  short16 var126; // (AddBlock(gid: 82), 2)
  short16 var127; // (AddBlock(gid: 82), 3)
  short16 var128; // (TensorEdge((-78, config), (88, config)), 0)
  short16 var129; // (TensorEdge((87, odata), (88, data), 1), 0)
  short16 var130; // (TensorEdge((87, odata), (88, data), 1), 1)
  short16 var131; // (TensorEdge((87, odata), (88, data), 1), 2)
  short16 var132; // (TensorEdge((87, odata), (88, data), 1), 3)
  short16 var133; // (TensorEdge((-75, min), (86, min)), 0)
  short16 var134; // (TensorEdge((-76, max), (86, max)), 0)
  short16 var135; // (TensorEdge((85, odata), (86, data)), 0)
  short16 var136; // (TensorEdge((85, odata), (86, data)), 1)
  short16 var137; // (MinmaxQuantBlock(gid: 86), 0)
  short16 var138; // (MinmaxQuantBlock(gid: 86), 1)
  short16 var139; // (MinmaxQuantBlock(gid: 86), 2)
  short16 var140; // (MinmaxQuantBlock(gid: 86), 3)
  short16 var141; // (TensorEdge((-61, odata), (85, lhs)), 0)
  short16 var142; // (TensorEdge((-62, odata), (85, rhs)), 0)
  short16 var143; // (AddBlock(gid: 85), 0)
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
    // generate: TensorEdge(((104, -90), config), ((104, 98), config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge(((104, -90), config), ((104, 98), config)), config write
    // generate: TensorEdge(((104, -91), fused_scale), ((104, 99), fused_scale)), fused_scale write

    var38 = __builtin_IMCE_RECV(1);
    var39 = __builtin_IMCE_RECV(1);
    var40 = __builtin_IMCE_RECV(1);
    var41 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((104, -91), fused_scale), ((104, 99), fused_scale)), fused_scale write
    // generate: TensorEdge(((104, -92), fused_bias), ((104, 99), fused_bias)), fused_bias write

    var42 = __builtin_IMCE_RECV(1);
    var43 = __builtin_IMCE_RECV(1);
    var44 = __builtin_IMCE_RECV(1);
    var45 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((104, -92), fused_bias), ((104, 99), fused_bias)), fused_bias write
    // generate: mult const

    var46 = __builtin_IMCE_RECV(3);
    var47 = __builtin_IMCE_RECV(3);
    var48 = __builtin_IMCE_RECV(3);
    var49 = __builtin_IMCE_RECV(3);
    // endgenerate: mult const
    // generate: add const

    var50 = __builtin_IMCE_RECV(2);
    var51 = __builtin_IMCE_RECV(2);
    var52 = __builtin_IMCE_RECV(2);
    var53 = __builtin_IMCE_RECV(2);
    // endgenerate: add const
    // generate: conv exec11
    // generate: conv exec11_row_group0_outer_loop(iterate row offset)
    // generate : conv exec11_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec11_row_group0_col_group0
    // generate : conv exec11_row_group0_col_group0. loop count == 1

    // generate: load_block
    // generate : load_block. loop count == 1
    for (int i1 = 0; i1 < 4; i1++) { // generate
      __builtin_IMCE_LOAD_LB(0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_4 -> imce_1_3

    } // endgenerate
    // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var33 = __builtin_IMCE_GET_CREG((short)0);
    var34 = __builtin_IMCE_GET_CREG((short)1);
    var35 = __builtin_IMCE_GET_CREG((short)2);
    var36 = __builtin_IMCE_GET_CREG((short)3);
    // generate: batch_norm

    var58 = __builtin_IMCE_MULTL(var33, var38, 15);
    var58 = __builtin_IMCE_ADD(var58, var42, 15);
    var59 = __builtin_IMCE_MULTL(var34, var39, 15);
    var59 = __builtin_IMCE_ADD(var59, var43, 15);
    var60 = __builtin_IMCE_MULTL(var35, var40, 15);
    var60 = __builtin_IMCE_ADD(var60, var44, 15);
    var61 = __builtin_IMCE_MULTL(var36, var41, 15);
    var61 = __builtin_IMCE_ADD(var61, var45, 15);
    // endgenerate: batch_norm
    // generate: multl

    var62 = __builtin_IMCE_MULTL(var46, var58, 15);
    var63 = __builtin_IMCE_MULTL(var47, var59, 15);
    var64 = __builtin_IMCE_MULTL(var48, var60, 15);
    var65 = __builtin_IMCE_MULTL(var49, var61, 15);
    // endgenerate: multl
    // generate: add

    var54 = __builtin_IMCE_ADD(var50, var62, 15);
    var55 = __builtin_IMCE_ADD(var51, var63, 15);
    var56 = __builtin_IMCE_ADD(var52, var64, 15);
    var57 = __builtin_IMCE_ADD(var53, var65, 15);
    // endgenerate: add
    __builtin_IMCE_SEND(1, var54, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_1_3 -> inode_2_0
    __builtin_IMCE_SEND(1, var55, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_1_3 -> inode_2_0
    __builtin_IMCE_SEND(1, var56, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_1_3 -> inode_2_0
    __builtin_IMCE_SEND(1, var57, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_1_3 -> inode_2_0
    // endgenerate : conv exec11_row_group0_col_group0
    // endgenerate: conv exec11_row_group0_col_group0
    // generate: conv exec11_row_group0_col_group1
    // generate : conv exec11_row_group0_col_group1. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 2; i1++) { // generate : load_block
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_4 -> imce_1_3

      } // endgenerate
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var33 = __builtin_IMCE_GET_CREG((short)0);
    var34 = __builtin_IMCE_GET_CREG((short)1);
    var35 = __builtin_IMCE_GET_CREG((short)2);
    var36 = __builtin_IMCE_GET_CREG((short)3);
    // generate: batch_norm

    var58 = __builtin_IMCE_MULTL(var33, var38, 15);
    var58 = __builtin_IMCE_ADD(var58, var42, 15);
    var59 = __builtin_IMCE_MULTL(var34, var39, 15);
    var59 = __builtin_IMCE_ADD(var59, var43, 15);
    var60 = __builtin_IMCE_MULTL(var35, var40, 15);
    var60 = __builtin_IMCE_ADD(var60, var44, 15);
    var61 = __builtin_IMCE_MULTL(var36, var41, 15);
    var61 = __builtin_IMCE_ADD(var61, var45, 15);
    // endgenerate: batch_norm
    // generate: multl

    var62 = __builtin_IMCE_MULTL(var46, var58, 15);
    var63 = __builtin_IMCE_MULTL(var47, var59, 15);
    var64 = __builtin_IMCE_MULTL(var48, var60, 15);
    var65 = __builtin_IMCE_MULTL(var49, var61, 15);
    // endgenerate: multl
    // generate: add

    var54 = __builtin_IMCE_ADD(var50, var62, 15);
    var55 = __builtin_IMCE_ADD(var51, var63, 15);
    var56 = __builtin_IMCE_ADD(var52, var64, 15);
    var57 = __builtin_IMCE_ADD(var53, var65, 15);
    // endgenerate: add
    __builtin_IMCE_SEND(1, var54, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_1_3 -> inode_2_0
    __builtin_IMCE_SEND(1, var55, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_1_3 -> inode_2_0
    __builtin_IMCE_SEND(1, var56, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_1_3 -> inode_2_0
    __builtin_IMCE_SEND(1, var57, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_1_3 -> inode_2_0
    // endgenerate : conv exec11_row_group0_col_group1
    // endgenerate: conv exec11_row_group0_col_group1
    // endgenerate : conv exec11_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec11_row_group0_outer_loop(iterate row offset)
    // generate: conv exec11_row_group1_outer_loop(iterate row offset)
    // generate : conv exec11_row_group1_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec11_row_group1_col_group0
    // generate : conv exec11_row_group1_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 6; i1++) { // generate : load_block
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_4 -> imce_1_3

      } // endgenerate
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var33 = __builtin_IMCE_GET_CREG((short)0);
    var34 = __builtin_IMCE_GET_CREG((short)1);
    var35 = __builtin_IMCE_GET_CREG((short)2);
    var36 = __builtin_IMCE_GET_CREG((short)3);
    // generate: batch_norm

    var58 = __builtin_IMCE_MULTL(var33, var38, 15);
    var58 = __builtin_IMCE_ADD(var58, var42, 15);
    var59 = __builtin_IMCE_MULTL(var34, var39, 15);
    var59 = __builtin_IMCE_ADD(var59, var43, 15);
    var60 = __builtin_IMCE_MULTL(var35, var40, 15);
    var60 = __builtin_IMCE_ADD(var60, var44, 15);
    var61 = __builtin_IMCE_MULTL(var36, var41, 15);
    var61 = __builtin_IMCE_ADD(var61, var45, 15);
    // endgenerate: batch_norm
    // generate: multl

    var62 = __builtin_IMCE_MULTL(var46, var58, 15);
    var63 = __builtin_IMCE_MULTL(var47, var59, 15);
    var64 = __builtin_IMCE_MULTL(var48, var60, 15);
    var65 = __builtin_IMCE_MULTL(var49, var61, 15);
    // endgenerate: multl
    // generate: add

    var54 = __builtin_IMCE_ADD(var50, var62, 15);
    var55 = __builtin_IMCE_ADD(var51, var63, 15);
    var56 = __builtin_IMCE_ADD(var52, var64, 15);
    var57 = __builtin_IMCE_ADD(var53, var65, 15);
    // endgenerate: add
    __builtin_IMCE_SEND(1, var54, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_1_3 -> inode_2_0
    __builtin_IMCE_SEND(1, var55, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_1_3 -> inode_2_0
    __builtin_IMCE_SEND(1, var56, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_1_3 -> inode_2_0
    __builtin_IMCE_SEND(1, var57, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_1_3 -> inode_2_0
    // endgenerate : conv exec11_row_group1_col_group0
    // endgenerate: conv exec11_row_group1_col_group0
    // generate: conv exec11_row_group1_col_group1
    // generate : conv exec11_row_group1_col_group1. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 2; i1++) { // generate : load_block
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_4 -> imce_1_3

      } // endgenerate
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var33 = __builtin_IMCE_GET_CREG((short)0);
    var34 = __builtin_IMCE_GET_CREG((short)1);
    var35 = __builtin_IMCE_GET_CREG((short)2);
    var36 = __builtin_IMCE_GET_CREG((short)3);
    // generate: batch_norm

    var58 = __builtin_IMCE_MULTL(var33, var38, 15);
    var58 = __builtin_IMCE_ADD(var58, var42, 15);
    var59 = __builtin_IMCE_MULTL(var34, var39, 15);
    var59 = __builtin_IMCE_ADD(var59, var43, 15);
    var60 = __builtin_IMCE_MULTL(var35, var40, 15);
    var60 = __builtin_IMCE_ADD(var60, var44, 15);
    var61 = __builtin_IMCE_MULTL(var36, var41, 15);
    var61 = __builtin_IMCE_ADD(var61, var45, 15);
    // endgenerate: batch_norm
    // generate: multl

    var62 = __builtin_IMCE_MULTL(var46, var58, 15);
    var63 = __builtin_IMCE_MULTL(var47, var59, 15);
    var64 = __builtin_IMCE_MULTL(var48, var60, 15);
    var65 = __builtin_IMCE_MULTL(var49, var61, 15);
    // endgenerate: multl
    // generate: add

    var54 = __builtin_IMCE_ADD(var50, var62, 15);
    var55 = __builtin_IMCE_ADD(var51, var63, 15);
    var56 = __builtin_IMCE_ADD(var52, var64, 15);
    var57 = __builtin_IMCE_ADD(var53, var65, 15);
    // endgenerate: add
    __builtin_IMCE_SEND(1, var54, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_1_3 -> inode_2_0
    __builtin_IMCE_SEND(1, var55, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_1_3 -> inode_2_0
    __builtin_IMCE_SEND(1, var56, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_1_3 -> inode_2_0
    __builtin_IMCE_SEND(1, var57, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_1_3 -> inode_2_0
    // endgenerate : conv exec11_row_group1_col_group1
    // endgenerate: conv exec11_row_group1_col_group1
    // endgenerate : conv exec11_row_group1_outer_loop(iterate row offset)
    // endgenerate: conv exec11_row_group1_outer_loop(iterate row offset)
    // generate: conv exec11_tail_loop
    for (int i1 = 0; i1 < 20; i1++) { // generate : conv exec11_tail_loop
      __builtin_IMCE_RECV(0);
    } // endgenerate : conv exec11_tail_loop
    // endgenerate: conv exec11_tail_loop
    // endgenerate: conv exec11
    __builtin_IMCE_STOP();
  }
  else if (hid == 1 && wid == 4) { // imce_1_4
    // generate: TensorEdge((-94, min), (103, min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge((-94, min), (103, min)), min write
    // generate: TensorEdge((-95, max), (103, max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge((-95, max), (103, max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 16; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      var68 = __builtin_IMCE_RECV(2); // TensorEdge((85, odata), (103, data)), imce_3_4 -> imce_1_4
      var69 = __builtin_IMCE_RECV(2); // TensorEdge((85, odata), (103, data)), imce_3_4 -> imce_1_4
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var68, 0, 15, 0);
      __builtin_IMCE_MM_QUANT(var69, 0, 15, 1);
      var70 = __builtin_IMCE_GET_QREG(0);
      var71 = __builtin_IMCE_GET_QREG(1);
      var72 = __builtin_IMCE_GET_QREG(2);
      var73 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      __builtin_IMCE_SEND(2, var70, 0, 0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_4 -> imce_1_3
      __builtin_IMCE_SEND(2, var71, 0, 0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_4 -> imce_1_3
      __builtin_IMCE_SEND(2, var72, 0, 0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_4 -> imce_1_3
      __builtin_IMCE_SEND(2, var73, 0, 0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_4 -> imce_1_3
      // endgenerate: min_max_quantize_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 2 && wid == 1) { // imce_2_1
    // generate: TensorEdge(((97, -66), config), ((97, 77), config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge(((97, -66), config), ((97, 77), config)), config write
    // generate: TensorEdge(((97, -67), fused_scale), ((97, 79), fused_scale)), fused_scale write

    var75 = __builtin_IMCE_RECV(1);
    var76 = __builtin_IMCE_RECV(1);
    var77 = __builtin_IMCE_RECV(1);
    var78 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((97, -67), fused_scale), ((97, 79), fused_scale)), fused_scale write
    // generate: TensorEdge(((97, -68), fused_bias), ((97, 79), fused_bias)), fused_bias write

    var79 = __builtin_IMCE_RECV(1);
    var80 = __builtin_IMCE_RECV(1);
    var81 = __builtin_IMCE_RECV(1);
    var82 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((97, -68), fused_bias), ((97, 79), fused_bias)), fused_bias write
    // generate: conv exec10
    // generate: conv exec10_row_group0_outer_loop(iterate row offset)
    // generate : conv exec10_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec10_row_group0_col_group0
    // generate : conv exec10_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 3; i1++) { // generate : load_block
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((91, odata), ((97, 77), data), 2), imce_2_4 -> imce_2_3, imce_2_2, imce_2_1
      } // endgenerate
    } // endgenerate : load_block
    for (int i2 = 0; i2 < 3; i2++) { // generate
      __builtin_IMCE_LOAD_LB(0); // TensorEdge((91, odata), ((97, 77), data), 2), imce_2_4 -> imce_2_3, imce_2_2, imce_2_1
    } // endgenerate

    __builtin_IMCE_STANDBY(12, 1);
    __builtin_IMCE_LOAD_LB(0); // TensorEdge((91, odata), ((97, 77), data), 2), imce_2_4 -> imce_2_3, imce_2_2, imce_2_1
    // endgenerate: load_block
    __builtin_IMCE_STEP();

    var25 = __builtin_IMCE_GET_CREG((short)0);
    var26 = __builtin_IMCE_GET_CREG((short)1);
    var27 = __builtin_IMCE_GET_CREG((short)2);
    var28 = __builtin_IMCE_GET_CREG((short)3);
    var29 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1
    var30 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1
    var31 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1
    var32 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1
    // generate: add

    var87 = __builtin_IMCE_ADD(var29, var25, 15);
    var88 = __builtin_IMCE_ADD(var30, var26, 15);
    var89 = __builtin_IMCE_ADD(var31, var27, 15);
    var90 = __builtin_IMCE_ADD(var32, var28, 15);
    // endgenerate: add
    // generate: batch_norm

    var83 = __builtin_IMCE_MULTL(var87, var75, 15);
    var83 = __builtin_IMCE_ADD(var83, var79, 15);
    var84 = __builtin_IMCE_MULTL(var88, var76, 15);
    var84 = __builtin_IMCE_ADD(var84, var80, 15);
    var85 = __builtin_IMCE_MULTL(var89, var77, 15);
    var85 = __builtin_IMCE_ADD(var85, var81, 15);
    var86 = __builtin_IMCE_MULTL(var90, var78, 15);
    var86 = __builtin_IMCE_ADD(var86, var82, 15);
    // endgenerate: batch_norm

    __builtin_IMCE_STEP();

    short16 var225;
    short16 var226;
    short16 var227;
    short16 var228;
    short16 var229;
    short16 var230;
    short16 var231;
    short16 var232;
    short16 var283;
    short16 var284;
    short16 var285;
    short16 var286;
    short16 var287;
    short16 var288;
    short16 var289;
    short16 var290;

    var225 = __builtin_IMCE_GET_CREG((short)0);
    var226 = __builtin_IMCE_GET_CREG((short)1);
    var227 = __builtin_IMCE_GET_CREG((short)2);
    var228 = __builtin_IMCE_GET_CREG((short)3);
    var229 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1
    var230 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1
    var231 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1
    var232 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1

    var287 = __builtin_IMCE_ADD(var229, var225, 15);
    var288 = __builtin_IMCE_ADD(var230, var226, 15);
    var289 = __builtin_IMCE_ADD(var231, var227, 15);
    var290 = __builtin_IMCE_ADD(var232, var228, 15);

    var283 = __builtin_IMCE_MULTL(var287, var75, 15);
    var283 = __builtin_IMCE_ADD(var283, var79, 15);
    var284 = __builtin_IMCE_MULTL(var288, var76, 15);
    var284 = __builtin_IMCE_ADD(var284, var80, 15);
    var285 = __builtin_IMCE_MULTL(var289, var77, 15);
    var285 = __builtin_IMCE_ADD(var285, var81, 15);
    var286 = __builtin_IMCE_MULTL(var290, var78, 15);
    var286 = __builtin_IMCE_ADD(var286, var82, 15);

    __builtin_IMCE_STEP();

    short16 var325;
    short16 var326;
    short16 var327;
    short16 var328;
    short16 var329;
    short16 var330;
    short16 var331;
    short16 var332;
    short16 var383;
    short16 var384;
    short16 var385;
    short16 var386;
    short16 var387;
    short16 var388;
    short16 var389;
    short16 var390;

    var325 = __builtin_IMCE_GET_CREG((short)0);
    var326 = __builtin_IMCE_GET_CREG((short)1);
    var327 = __builtin_IMCE_GET_CREG((short)2);
    var328 = __builtin_IMCE_GET_CREG((short)3);
    var329 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1
    var330 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1
    var331 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1
    var332 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1

    var387 = __builtin_IMCE_ADD(var329, var325, 15);
    var388 = __builtin_IMCE_ADD(var330, var326, 15);
    var389 = __builtin_IMCE_ADD(var331, var327, 15);
    var390 = __builtin_IMCE_ADD(var332, var328, 15);

    var383 = __builtin_IMCE_MULTL(var387, var75, 15);
    var383 = __builtin_IMCE_ADD(var383, var79, 15);
    var384 = __builtin_IMCE_MULTL(var388, var76, 15);
    var384 = __builtin_IMCE_ADD(var384, var80, 15);
    var385 = __builtin_IMCE_MULTL(var389, var77, 15);
    var385 = __builtin_IMCE_ADD(var385, var81, 15);
    var386 = __builtin_IMCE_MULTL(var390, var78, 15);
    var386 = __builtin_IMCE_ADD(var386, var82, 15);

    __builtin_IMCE_NOP();

    short16 var425;
    short16 var426;
    short16 var427;
    short16 var428;
    short16 var429;
    short16 var430;
    short16 var431;
    short16 var432;
    short16 var483;
    short16 var484;
    short16 var485;
    short16 var486;
    short16 var487;
    short16 var488;
    short16 var489;
    short16 var490;

    var425 = __builtin_IMCE_GET_CREG((short)0);
    var426 = __builtin_IMCE_GET_CREG((short)1);
    var427 = __builtin_IMCE_GET_CREG((short)2);
    var428 = __builtin_IMCE_GET_CREG((short)3);
    var429 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1
    var430 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1
    var431 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1
    var432 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1

    var487 = __builtin_IMCE_ADD(var429, var425, 15);
    var488 = __builtin_IMCE_ADD(var430, var426, 15);
    var489 = __builtin_IMCE_ADD(var431, var427, 15);
    var490 = __builtin_IMCE_ADD(var432, var428, 15);

    var483 = __builtin_IMCE_MULTL(var487, var75, 15);
    var483 = __builtin_IMCE_ADD(var483, var79, 15);
    var484 = __builtin_IMCE_MULTL(var488, var76, 15);
    var484 = __builtin_IMCE_ADD(var484, var80, 15);
    var485 = __builtin_IMCE_MULTL(var489, var77, 15);
    var485 = __builtin_IMCE_ADD(var485, var81, 15);
    var486 = __builtin_IMCE_MULTL(var490, var78, 15);
    var486 = __builtin_IMCE_ADD(var486, var82, 15);

    // endgenerate: batch_norm
    __builtin_IMCE_SEND(1, var83, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_2_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var84, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_2_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var85, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_2_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var86, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_2_1 -> inode_3_0

    __builtin_IMCE_SEND(1, var283, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_2_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var284, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_2_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var285, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_2_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var286, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_2_1 -> inode_3_0

    __builtin_IMCE_SEND(1, var383, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_2_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var384, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_2_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var385, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_2_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var386, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_2_1 -> inode_3_0

    __builtin_IMCE_SEND(1, var484, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_2_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var483, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_2_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var485, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_2_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var486, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_2_1 -> inode_3_0
    // endgenerate: conv exec10_row_group1_col_group0
    // endgenerate : conv exec10_row_group1_outer_loop(iterate row offset)
    // endgenerate: conv exec10_row_group1_outer_loop(iterate row offset)
    // endgenerate: conv exec10
    __builtin_IMCE_STOP();
  }
  else if (hid == 2 && wid == 2) { // imce_2_2
    // generate: TensorEdge(((96, -84), config), ((96, 92), config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge(((96, -84), config), ((96, 92), config)), config write
    // generate: conv exec9
    // generate: conv exec9_row_group0_outer_loop(iterate row offset)
    // generate : conv exec9_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec9_row_group0_col_group0
    // generate : conv exec9_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 4; i1++) { // generate : load_block
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((91, odata), ((96, 92), data), 0), imce_2_4 -> imce_2_3, imce_2_2

      } // endgenerate
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var17 = __builtin_IMCE_GET_CREG((short)0);
    var18 = __builtin_IMCE_GET_CREG((short)1);
    var19 = __builtin_IMCE_GET_CREG((short)2);
    var20 = __builtin_IMCE_GET_CREG((short)3);
    var21 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_2_2
    var22 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_2_2
    var23 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_2_2
    var24 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_2_2
    // generate: add

    var92 = __builtin_IMCE_ADD(var17, var21, 15);
    var93 = __builtin_IMCE_ADD(var18, var22, 15);
    var94 = __builtin_IMCE_ADD(var19, var23, 15);
    var95 = __builtin_IMCE_ADD(var20, var24, 15);
    // endgenerate: add
    __builtin_IMCE_SETFLAG(1);
    __builtin_IMCE_SEND(2, var92, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1
    __builtin_IMCE_SEND(2, var93, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1
    __builtin_IMCE_SEND(2, var94, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1
    __builtin_IMCE_SEND(2, var95, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1
    __builtin_IMCE_SETFLAG(0);
    // endgenerate : conv exec9_row_group0_col_group0
    // endgenerate: conv exec9_row_group0_col_group0
    // generate: conv exec9_row_group0_col_group1
    // generate : conv exec9_row_group0_col_group1. loop count == 1

    // generate: load_block
    // loop ignored with loop count == 0 : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var17 = __builtin_IMCE_GET_CREG((short)0);
    var18 = __builtin_IMCE_GET_CREG((short)1);
    var19 = __builtin_IMCE_GET_CREG((short)2);
    var20 = __builtin_IMCE_GET_CREG((short)3);
    var21 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_2_2
    var22 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_2_2
    var23 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_2_2
    var24 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_2_2
    // generate: add

    var92 = __builtin_IMCE_ADD(var17, var21, 15);
    var93 = __builtin_IMCE_ADD(var18, var22, 15);
    var94 = __builtin_IMCE_ADD(var19, var23, 15);
    var95 = __builtin_IMCE_ADD(var20, var24, 15);
    // endgenerate: add
    __builtin_IMCE_SEND(2, var92, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1
    __builtin_IMCE_SEND(2, var93, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1
    __builtin_IMCE_SEND(2, var94, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1
    __builtin_IMCE_SEND(2, var95, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1
    // endgenerate : conv exec9_row_group0_col_group1
    // endgenerate: conv exec9_row_group0_col_group1
    // endgenerate : conv exec9_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec9_row_group0_outer_loop(iterate row offset)
    // generate: conv exec9_row_group1_outer_loop(iterate row offset)
    // generate : conv exec9_row_group1_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec9_row_group1_col_group0
    for (int i1 = 0; i1 < 2; i1++) { // generate : conv exec9_row_group1_col_group0

      // generate: load_block
      // loop ignored with loop count == 0 : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var17 = __builtin_IMCE_GET_CREG((short)0);
      var18 = __builtin_IMCE_GET_CREG((short)1);
      var19 = __builtin_IMCE_GET_CREG((short)2);
      var20 = __builtin_IMCE_GET_CREG((short)3);
      var21 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_2_2
      var22 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_2_2
      var23 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_2_2
      var24 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_2_2
      // generate: add

      var92 = __builtin_IMCE_ADD(var17, var21, 15);
      var93 = __builtin_IMCE_ADD(var18, var22, 15);
      var94 = __builtin_IMCE_ADD(var19, var23, 15);
      var95 = __builtin_IMCE_ADD(var20, var24, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(2, var92, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1
      __builtin_IMCE_SEND(2, var93, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1
      __builtin_IMCE_SEND(2, var94, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1
      __builtin_IMCE_SEND(2, var95, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_2_2 -> imce_2_1
    } // endgenerate : conv exec9_row_group1_col_group0
    // endgenerate: conv exec9_row_group1_col_group0
    // endgenerate : conv exec9_row_group1_outer_loop(iterate row offset)
    // endgenerate: conv exec9_row_group1_outer_loop(iterate row offset)
    // endgenerate: conv exec9
    __builtin_IMCE_STOP();
  }
  else if (hid == 2 && wid == 3) { // imce_2_3
    // generate: TensorEdge((-86, config), (95, config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge((-86, config), (95, config)), config write
    // generate: conv exec8
    // generate: conv exec8_row_group0_outer_loop(iterate row offset)
    // generate : conv exec8_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec8_row_group0_col_group0
    // generate : conv exec8_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 4; i1++) { // generate : load_block
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_SETFLAG(1);
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((91, odata), (95, data), 1), imce_2_4 -> imce_2_3
        __builtin_IMCE_SETFLAG(0);

      } // endgenerate
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var13 = __builtin_IMCE_GET_CREG((short)0);
    var14 = __builtin_IMCE_GET_CREG((short)1);
    var15 = __builtin_IMCE_GET_CREG((short)2);
    var16 = __builtin_IMCE_GET_CREG((short)3);
    __builtin_IMCE_SEND(5, var13, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_2_2
    __builtin_IMCE_SEND(5, var14, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_2_2
    __builtin_IMCE_SEND(5, var15, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_2_2
    __builtin_IMCE_SEND(5, var16, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_2_2
    // endgenerate : conv exec8_row_group0_col_group0
    // endgenerate: conv exec8_row_group0_col_group0
    // generate: conv exec8_row_group0_col_group1
    // generate : conv exec8_row_group0_col_group1. loop count == 1

    // generate: load_block
    // loop ignored with loop count == 0 : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var13 = __builtin_IMCE_GET_CREG((short)0);
    var14 = __builtin_IMCE_GET_CREG((short)1);
    var15 = __builtin_IMCE_GET_CREG((short)2);
    var16 = __builtin_IMCE_GET_CREG((short)3);
    __builtin_IMCE_SEND(5, var13, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_2_2
    __builtin_IMCE_SEND(5, var14, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_2_2
    __builtin_IMCE_SEND(5, var15, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_2_2
    __builtin_IMCE_SEND(5, var16, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_2_2
    // endgenerate : conv exec8_row_group0_col_group1
    // endgenerate: conv exec8_row_group0_col_group1
    // endgenerate : conv exec8_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec8_row_group0_outer_loop(iterate row offset)
    // generate: conv exec8_row_group1_outer_loop(iterate row offset)
    // generate : conv exec8_row_group1_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec8_row_group1_col_group0
    for (int i1 = 0; i1 < 2; i1++) { // generate : conv exec8_row_group1_col_group0

      // generate: load_block
      // loop ignored with loop count == 0 : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var13 = __builtin_IMCE_GET_CREG((short)0);
      var14 = __builtin_IMCE_GET_CREG((short)1);
      var15 = __builtin_IMCE_GET_CREG((short)2);
      var16 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(5, var13, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_2_2
      __builtin_IMCE_SEND(5, var14, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_2_2
      __builtin_IMCE_SEND(5, var15, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_2_2
      __builtin_IMCE_SEND(5, var16, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_2_2
    } // endgenerate : conv exec8_row_group1_col_group0
    // endgenerate: conv exec8_row_group1_col_group0
    // endgenerate : conv exec8_row_group1_outer_loop(iterate row offset)
    // endgenerate: conv exec8_row_group1_outer_loop(iterate row offset)
    // endgenerate: conv exec8
    __builtin_IMCE_STOP();
  }
  else if (hid == 2 && wid == 4) { // imce_2_4
    // generate: TensorEdge((-79, min), (90, min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge((-79, min), (90, min)), min write
    // generate: TensorEdge((-80, max), (90, max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge((-80, max), (90, max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 4; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      var103 = __builtin_IMCE_RECV(2); // TensorEdge(((89, 83), odata), (90, data)), imce_3_1 -> imce_2_4
      var104 = __builtin_IMCE_RECV(2); // TensorEdge(((89, 83), odata), (90, data)), imce_3_1 -> imce_2_4
      var105 = __builtin_IMCE_RECV(2); // TensorEdge(((89, 83), odata), (90, data)), imce_3_1 -> imce_2_4
      var106 = __builtin_IMCE_RECV(2); // TensorEdge(((89, 83), odata), (90, data)), imce_3_1 -> imce_2_4
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var103, 0, 15, 0);
      __builtin_IMCE_MM_QUANT(var104, 0, 15, 1);
      __builtin_IMCE_MM_QUANT(var105, 0, 15, 2);
      __builtin_IMCE_MM_QUANT(var106, 0, 15, 3);
      var107 = __builtin_IMCE_GET_QREG(0);
      var108 = __builtin_IMCE_GET_QREG(1);
      var109 = __builtin_IMCE_GET_QREG(2);
      var110 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      __builtin_IMCE_STANDBY(13, 1);
      __builtin_IMCE_SEND(1, var107, 0, 0); // TensorEdge((90, odata), (91, data)), imce_2_4 -> imce_2_3, imce_2_2, imce_2_1
      __builtin_IMCE_STANDBY(13, 1);
      __builtin_IMCE_SEND(1, var108, 0, 0); // TensorEdge((90, odata), (91, data)), imce_2_4 -> imce_2_3, imce_2_2, imce_2_1
      __builtin_IMCE_STANDBY(13, 1);
      __builtin_IMCE_SEND(1, var109, 0, 0); // TensorEdge((90, odata), (91, data)), imce_2_4 -> imce_2_3, imce_2_2, imce_2_1
      __builtin_IMCE_STANDBY(13, 1);
      __builtin_IMCE_SEND(1, var110, 0, 0); // TensorEdge((90, odata), (91, data)), imce_2_4 -> imce_2_3, imce_2_2, imce_2_1
      // endgenerate: min_max_quantize_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 1) { // imce_3_1
    // generate: TensorEdge(((89, -72), config), ((89, 81), config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge(((89, -72), config), ((89, 81), config)), config write
    // generate: TensorEdge(((89, -73), fused_scale), ((89, 83), fused_scale)), fused_scale write

    var112 = __builtin_IMCE_RECV(1);
    var113 = __builtin_IMCE_RECV(1);
    var114 = __builtin_IMCE_RECV(1);
    var115 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((89, -73), fused_scale), ((89, 83), fused_scale)), fused_scale write
    // generate: TensorEdge(((89, -74), fused_bias), ((89, 83), fused_bias)), fused_bias write

    var116 = __builtin_IMCE_RECV(1);
    var117 = __builtin_IMCE_RECV(1);
    var118 = __builtin_IMCE_RECV(1);
    var119 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((89, -74), fused_bias), ((89, 83), fused_bias)), fused_bias write
    // generate: conv exec7
    // generate: conv exec7_row_group0_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 2; i1++) { // generate : conv exec7_row_group0_outer_loop(iterate row offset)
      // generate: conv exec7_row_group0_col_group0
      // generate : conv exec7_row_group0_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 6; i2++) { // generate : load_block
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((87, odata), ((89, 81), data), 0), imce_3_3 -> imce_3_2, imce_3_1

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      var9 = __builtin_IMCE_RECV(2); // TensorEdge((88, odata), ((89, 82), rhs)), imce_3_2 -> imce_3_1
      var10 = __builtin_IMCE_RECV(2); // TensorEdge((88, odata), ((89, 82), rhs)), imce_3_2 -> imce_3_1
      var11 = __builtin_IMCE_RECV(2); // TensorEdge((88, odata), ((89, 82), rhs)), imce_3_2 -> imce_3_1
      var12 = __builtin_IMCE_RECV(2); // TensorEdge((88, odata), ((89, 82), rhs)), imce_3_2 -> imce_3_1
      // generate: add

      var124 = __builtin_IMCE_ADD(var5, var9, 15);
      var125 = __builtin_IMCE_ADD(var6, var10, 15);
      var126 = __builtin_IMCE_ADD(var7, var11, 15);
      var127 = __builtin_IMCE_ADD(var8, var12, 15);
      // endgenerate: add
      // generate: batch_norm

      var120 = __builtin_IMCE_MULTL(var124, var112, 15);
      var120 = __builtin_IMCE_ADD(var120, var116, 15);
      var121 = __builtin_IMCE_MULTL(var125, var113, 15);
      var121 = __builtin_IMCE_ADD(var121, var117, 15);
      var122 = __builtin_IMCE_MULTL(var126, var114, 15);
      var122 = __builtin_IMCE_ADD(var122, var118, 15);
      var123 = __builtin_IMCE_MULTL(var127, var115, 15);
      var123 = __builtin_IMCE_ADD(var123, var119, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(2, var120, 2, 0); // TensorEdge(((89, 83), odata), (90, data)), imce_3_1 -> imce_2_4
      __builtin_IMCE_SEND(2, var121, 2, 0); // TensorEdge(((89, 83), odata), (90, data)), imce_3_1 -> imce_2_4
      __builtin_IMCE_SEND(2, var122, 2, 0); // TensorEdge(((89, 83), odata), (90, data)), imce_3_1 -> imce_2_4
      __builtin_IMCE_SEND(2, var123, 2, 0); // TensorEdge(((89, 83), odata), (90, data)), imce_3_1 -> imce_2_4
      // endgenerate : conv exec7_row_group0_col_group0
      // endgenerate: conv exec7_row_group0_col_group0
      // generate: conv exec7_row_group0_col_group1
      // generate : conv exec7_row_group0_col_group1. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 2; i2++) { // generate : load_block
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((87, odata), ((89, 81), data), 0), imce_3_3 -> imce_3_2, imce_3_1

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      var9 = __builtin_IMCE_RECV(2); // TensorEdge((88, odata), ((89, 82), rhs)), imce_3_2 -> imce_3_1
      var10 = __builtin_IMCE_RECV(2); // TensorEdge((88, odata), ((89, 82), rhs)), imce_3_2 -> imce_3_1
      var11 = __builtin_IMCE_RECV(2); // TensorEdge((88, odata), ((89, 82), rhs)), imce_3_2 -> imce_3_1
      var12 = __builtin_IMCE_RECV(2); // TensorEdge((88, odata), ((89, 82), rhs)), imce_3_2 -> imce_3_1
      // generate: add

      var124 = __builtin_IMCE_ADD(var5, var9, 15);
      var125 = __builtin_IMCE_ADD(var6, var10, 15);
      var126 = __builtin_IMCE_ADD(var7, var11, 15);
      var127 = __builtin_IMCE_ADD(var8, var12, 15);
      // endgenerate: add
      // generate: batch_norm

      var120 = __builtin_IMCE_MULTL(var124, var112, 15);
      var120 = __builtin_IMCE_ADD(var120, var116, 15);
      var121 = __builtin_IMCE_MULTL(var125, var113, 15);
      var121 = __builtin_IMCE_ADD(var121, var117, 15);
      var122 = __builtin_IMCE_MULTL(var126, var114, 15);
      var122 = __builtin_IMCE_ADD(var122, var118, 15);
      var123 = __builtin_IMCE_MULTL(var127, var115, 15);
      var123 = __builtin_IMCE_ADD(var123, var119, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(2, var120, 2, 0); // TensorEdge(((89, 83), odata), (90, data)), imce_3_1 -> imce_2_4
      __builtin_IMCE_SEND(2, var121, 2, 0); // TensorEdge(((89, 83), odata), (90, data)), imce_3_1 -> imce_2_4
      __builtin_IMCE_SEND(2, var122, 2, 0); // TensorEdge(((89, 83), odata), (90, data)), imce_3_1 -> imce_2_4
      __builtin_IMCE_SEND(2, var123, 2, 0); // TensorEdge(((89, 83), odata), (90, data)), imce_3_1 -> imce_2_4
      // endgenerate : conv exec7_row_group0_col_group1
      // endgenerate: conv exec7_row_group0_col_group1
    } // endgenerate : conv exec7_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec7_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec7
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 2) { // imce_3_2
    // generate: TensorEdge((-78, config), (88, config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge((-78, config), (88, config)), config write
    // generate: conv exec6
    // generate: conv exec6_row_group0_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 2; i1++) { // generate : conv exec6_row_group0_outer_loop(iterate row offset)
      // generate: conv exec6_row_group0_col_group0
      // generate : conv exec6_row_group0_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 6; i2++) { // generate : load_block
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_SETFLAG(1);
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((87, odata), (88, data), 1), imce_3_3 -> imce_3_2
          __builtin_IMCE_SETFLAG(0);
        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var1 = __builtin_IMCE_GET_CREG((short)0);
      var2 = __builtin_IMCE_GET_CREG((short)1);
      var3 = __builtin_IMCE_GET_CREG((short)2);
      var4 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(2, var1, 2, 0); // TensorEdge((88, odata), ((89, 82), rhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var2, 2, 0); // TensorEdge((88, odata), ((89, 82), rhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var3, 2, 0); // TensorEdge((88, odata), ((89, 82), rhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var4, 2, 0); // TensorEdge((88, odata), ((89, 82), rhs)), imce_3_2 -> imce_3_1
      // endgenerate : conv exec6_row_group0_col_group0
      // endgenerate: conv exec6_row_group0_col_group0
      // generate: conv exec6_row_group0_col_group1
      // generate : conv exec6_row_group0_col_group1. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 2; i2++) { // generate : load_block
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_SETFLAG(1);
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((87, odata), (88, data), 1), imce_3_3 -> imce_3_2
          __builtin_IMCE_SETFLAG(0);
        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var1 = __builtin_IMCE_GET_CREG((short)0);
      var2 = __builtin_IMCE_GET_CREG((short)1);
      var3 = __builtin_IMCE_GET_CREG((short)2);
      var4 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(2, var1, 2, 0); // TensorEdge((88, odata), ((89, 82), rhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var2, 2, 0); // TensorEdge((88, odata), ((89, 82), rhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var3, 2, 0); // TensorEdge((88, odata), ((89, 82), rhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var4, 2, 0); // TensorEdge((88, odata), ((89, 82), rhs)), imce_3_2 -> imce_3_1
      // endgenerate : conv exec6_row_group0_col_group1
      // endgenerate: conv exec6_row_group0_col_group1
    } // endgenerate : conv exec6_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec6_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec6
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 3) { // imce_3_3
    // generate: TensorEdge((-75, min), (86, min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge((-75, min), (86, min)), min write
    // generate: TensorEdge((-76, max), (86, max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge((-76, max), (86, max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 16; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      var135 = __builtin_IMCE_RECV(2); // TensorEdge((85, odata), (86, data)), imce_3_4 -> imce_3_3
      var136 = __builtin_IMCE_RECV(2); // TensorEdge((85, odata), (86, data)), imce_3_4 -> imce_3_3
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var135, 0, 15, 0);
      __builtin_IMCE_MM_QUANT(var136, 0, 15, 1);
      var137 = __builtin_IMCE_GET_QREG(0);
      var138 = __builtin_IMCE_GET_QREG(1);
      var139 = __builtin_IMCE_GET_QREG(2);
      var140 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      __builtin_IMCE_STANDBY(17, 1);
      __builtin_IMCE_SEND(1, var137, 0, 0); // TensorEdge((86, odata), (87, data)), imce_3_3 -> imce_3_2, imce_3_1
      __builtin_IMCE_STANDBY(17, 1);
      __builtin_IMCE_SEND(1, var138, 0, 0); // TensorEdge((86, odata), (87, data)), imce_3_3 -> imce_3_2, imce_3_1
      __builtin_IMCE_STANDBY(17, 1);
      __builtin_IMCE_SEND(1, var139, 0, 0); // TensorEdge((86, odata), (87, data)), imce_3_3 -> imce_3_2, imce_3_1
      __builtin_IMCE_STANDBY(17, 1);
      __builtin_IMCE_SEND(1, var140, 0, 0); // TensorEdge((86, odata), (87, data)), imce_3_3 -> imce_3_2, imce_3_1
      // endgenerate: min_max_quantize_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 4) { // imce_3_4
    // generate: call_created_loop
    for (int i1 = 0; i1 < 32; i1++) { // generate : call_created_loop
      // generate: add standalone

      var141 = __builtin_IMCE_RECV(2); // TensorEdge((-61, odata), (85, lhs)), inode_0_0 -> imce_3_4
      var142 = __builtin_IMCE_RECV(3); // TensorEdge((-62, odata), (85, rhs)), inode_1_0 -> imce_3_4
      // generate: add

      var143 = __builtin_IMCE_ADD(var141, var142, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(1, var143, 2, 0); // TensorEdge((85, odata), (103, data)),TensorEdge((85, odata), (86, data)), imce_3_4 -> imce_3_3
      // endgenerate: add standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
}
