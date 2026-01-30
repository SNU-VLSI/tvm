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
  short16 var37; // (TensorEdge((-61, odata), (85, lhs)), 0)
  short16 var38; // (TensorEdge((-62, odata), (85, rhs)), 0)
  short16 var39; // (AddBlock(gid: 85), 0)
  short16 var40; // (TensorEdge((-75, min), (86, min)), 0)
  short16 var41; // (TensorEdge((-76, max), (86, max)), 0)
  short16 var42; // (TensorEdge((85, odata), (86, data)), 0)
  short16 var43; // (TensorEdge((85, odata), (86, data)), 1)
  short16 var44; // (MinmaxQuantBlock(gid: 86), 0)
  short16 var45; // (MinmaxQuantBlock(gid: 86), 1)
  short16 var46; // (MinmaxQuantBlock(gid: 86), 2)
  short16 var47; // (MinmaxQuantBlock(gid: 86), 3)
  short16 var48; // (TensorEdge((-94, min), (103, min)), 0)
  short16 var49; // (TensorEdge((-95, max), (103, max)), 0)
  short16 var50; // (TensorEdge((85, odata), (103, data)), 0)
  short16 var51; // (TensorEdge((85, odata), (103, data)), 1)
  short16 var52; // (MinmaxQuantBlock(gid: 103), 0)
  short16 var53; // (MinmaxQuantBlock(gid: 103), 1)
  short16 var54; // (MinmaxQuantBlock(gid: 103), 2)
  short16 var55; // (MinmaxQuantBlock(gid: 103), 3)
  short16 var56; // (TensorEdge(((89, -72), config), ((89, 81), config)), 0)
  short16 var57; // (TensorEdge(((89, -73), fused_scale), ((89, 83), fused_scale)), 0)
  short16 var58; // (TensorEdge(((89, -73), fused_scale), ((89, 83), fused_scale)), 1)
  short16 var59; // (TensorEdge(((89, -73), fused_scale), ((89, 83), fused_scale)), 2)
  short16 var60; // (TensorEdge(((89, -73), fused_scale), ((89, 83), fused_scale)), 3)
  short16 var61; // (TensorEdge(((89, -74), fused_bias), ((89, 83), fused_bias)), 0)
  short16 var62; // (TensorEdge(((89, -74), fused_bias), ((89, 83), fused_bias)), 1)
  short16 var63; // (TensorEdge(((89, -74), fused_bias), ((89, 83), fused_bias)), 2)
  short16 var64; // (TensorEdge(((89, -74), fused_bias), ((89, 83), fused_bias)), 3)
  short16 var65; // (BatchNormBlock(gid: 83), 0)
  short16 var66; // (BatchNormBlock(gid: 83), 1)
  short16 var67; // (BatchNormBlock(gid: 83), 2)
  short16 var68; // (BatchNormBlock(gid: 83), 3)
  short16 var69; // (AddBlock(gid: 82), 0)
  short16 var70; // (AddBlock(gid: 82), 1)
  short16 var71; // (AddBlock(gid: 82), 2)
  short16 var72; // (AddBlock(gid: 82), 3)
  short16 var73; // (TensorEdge((-78, config), (88, config)), 0)
  short16 var74; // (TensorEdge((87, odata), (88, data), 1), 0)
  short16 var75; // (TensorEdge((87, odata), (88, data), 1), 1)
  short16 var76; // (TensorEdge((87, odata), (88, data), 1), 2)
  short16 var77; // (TensorEdge((87, odata), (88, data), 1), 3)
  short16 var78; // (TensorEdge(((104, -90), config), ((104, 98), config)), 0)
  short16 var79; // (TensorEdge(((104, -91), fused_scale), ((104, 99), fused_scale)), 0)
  short16 var80; // (TensorEdge(((104, -91), fused_scale), ((104, 99), fused_scale)), 1)
  short16 var81; // (TensorEdge(((104, -91), fused_scale), ((104, 99), fused_scale)), 2)
  short16 var82; // (TensorEdge(((104, -91), fused_scale), ((104, 99), fused_scale)), 3)
  short16 var83; // (TensorEdge(((104, -92), fused_bias), ((104, 99), fused_bias)), 0)
  short16 var84; // (TensorEdge(((104, -92), fused_bias), ((104, 99), fused_bias)), 1)
  short16 var85; // (TensorEdge(((104, -92), fused_bias), ((104, 99), fused_bias)), 2)
  short16 var86; // (TensorEdge(((104, -92), fused_bias), ((104, 99), fused_bias)), 3)
  short16 var87; // (TensorEdge(((104, -88), odata), ((104, 100), lhs)), 0)
  short16 var88; // (TensorEdge(((104, -88), odata), ((104, 100), lhs)), 1)
  short16 var89; // (TensorEdge(((104, -88), odata), ((104, 100), lhs)), 2)
  short16 var90; // (TensorEdge(((104, -88), odata), ((104, 100), lhs)), 3)
  short16 var91; // (TensorEdge(((104, -93), odata), ((104, 101), rhs)), 0)
  short16 var92; // (TensorEdge(((104, -93), odata), ((104, 101), rhs)), 1)
  short16 var93; // (TensorEdge(((104, -93), odata), ((104, 101), rhs)), 2)
  short16 var94; // (TensorEdge(((104, -93), odata), ((104, 101), rhs)), 3)
  short16 var95; // (AddBlock(gid: 101), 0)
  short16 var96; // (AddBlock(gid: 101), 1)
  short16 var97; // (AddBlock(gid: 101), 2)
  short16 var98; // (AddBlock(gid: 101), 3)
  short16 var99; // (BatchNormBlock(gid: 99), 0)
  short16 var100; // (BatchNormBlock(gid: 99), 1)
  short16 var101; // (BatchNormBlock(gid: 99), 2)
  short16 var102; // (BatchNormBlock(gid: 99), 3)
  short16 var103; // (MultlBlock(gid: 100), 0)
  short16 var104; // (MultlBlock(gid: 100), 1)
  short16 var105; // (MultlBlock(gid: 100), 2)
  short16 var106; // (MultlBlock(gid: 100), 3)
  short16 var107; // (TensorEdge((-79, min), (90, min)), 0)
  short16 var108; // (TensorEdge((-80, max), (90, max)), 0)
  short16 var109; // (TensorEdge(((89, 83), odata), (90, data)), 0)
  short16 var110; // (TensorEdge(((89, 83), odata), (90, data)), 1)
  short16 var111; // (TensorEdge(((89, 83), odata), (90, data)), 2)
  short16 var112; // (TensorEdge(((89, 83), odata), (90, data)), 3)
  short16 var113; // (MinmaxQuantBlock(gid: 90), 0)
  short16 var114; // (MinmaxQuantBlock(gid: 90), 1)
  short16 var115; // (MinmaxQuantBlock(gid: 90), 2)
  short16 var116; // (MinmaxQuantBlock(gid: 90), 3)
  short16 var117; // (TensorEdge((-86, config), (95, config)), 0)
  short16 var118; // (TensorEdge((91, odata), (95, data), 1), 0)
  short16 var119; // (TensorEdge((91, odata), (95, data), 1), 1)
  short16 var120; // (TensorEdge((91, odata), (95, data), 1), 2)
  short16 var121; // (TensorEdge((91, odata), (95, data), 1), 3)
  short16 var122; // (TensorEdge(((97, -66), config), ((97, 77), config)), 0)
  short16 var123; // (TensorEdge(((97, -67), fused_scale), ((97, 79), fused_scale)), 0)
  short16 var124; // (TensorEdge(((97, -67), fused_scale), ((97, 79), fused_scale)), 1)
  short16 var125; // (TensorEdge(((97, -67), fused_scale), ((97, 79), fused_scale)), 2)
  short16 var126; // (TensorEdge(((97, -67), fused_scale), ((97, 79), fused_scale)), 3)
  short16 var127; // (TensorEdge(((97, -68), fused_bias), ((97, 79), fused_bias)), 0)
  short16 var128; // (TensorEdge(((97, -68), fused_bias), ((97, 79), fused_bias)), 1)
  short16 var129; // (TensorEdge(((97, -68), fused_bias), ((97, 79), fused_bias)), 2)
  short16 var130; // (TensorEdge(((97, -68), fused_bias), ((97, 79), fused_bias)), 3)
  short16 var131; // (BatchNormBlock(gid: 79), 0)
  short16 var132; // (BatchNormBlock(gid: 79), 1)
  short16 var133; // (BatchNormBlock(gid: 79), 2)
  short16 var134; // (BatchNormBlock(gid: 79), 3)
  short16 var135; // (AddBlock(gid: 78), 0)
  short16 var136; // (AddBlock(gid: 78), 1)
  short16 var137; // (AddBlock(gid: 78), 2)
  short16 var138; // (AddBlock(gid: 78), 3)
  short16 var139; // (TensorEdge(((96, -84), config), ((96, 92), config)), 0)
  short16 var140; // (AddBlock(gid: 93), 0)
  short16 var141; // (AddBlock(gid: 93), 1)
  short16 var142; // (AddBlock(gid: 93), 2)
  short16 var143; // (AddBlock(gid: 93), 3)
  if (hid == 0 && wid == 1) { // imce_0_1
    // generate: call_created_loop
    for (int i1 = 0; i1 < 256; i1++) { // generate : call_created_loop
      // generate: add standalone

      __builtin_IMCE_SETFLAG(1);
      __builtin_IMCE_STANDBY(0, 1);
      __builtin_IMCE_STANDBY(5, 1);
      __builtin_IMCE_SETFLAG(0);
      var37 = __builtin_IMCE_RECV(2); // TensorEdge((-61, odata), (85, lhs)), inode_0_0 -> imce_0_1
      var38 = __builtin_IMCE_RECV(3); // TensorEdge((-62, odata), (85, rhs)), inode_1_0 -> imce_0_1

      __builtin_IMCE_SETFLAG(1);
      __builtin_IMCE_STANDBY(0, 1);
      __builtin_IMCE_STANDBY(5, 1);
      __builtin_IMCE_SETFLAG(0);
      short16 var237, var238, var239;
      var237 = __builtin_IMCE_RECV(2); // TensorEdge((-61, odata), (85, lhs)), inode_0_0 -> imce_0_1
      var238 = __builtin_IMCE_RECV(3); // TensorEdge((-62, odata), (85, rhs)), inode_1_0 -> imce_0_1
      // generate: add

      var39 = __builtin_IMCE_ADD(var37, var38, 15);
      var239 = __builtin_IMCE_ADD(var237, var238, 15);
      // endgenerate: add

      __builtin_IMCE_STANDBY(2, 2);
      __builtin_IMCE_STANDBY(6, 2);
      __builtin_IMCE_SETFLAG(2);
      __builtin_IMCE_STANDBY(2, 0);
      __builtin_IMCE_STANDBY(6, 0);
      __builtin_IMCE_SETFLAG(0);
      __builtin_IMCE_SEND(1, var39, 2, 0); // TensorEdge((85, odata), (103, data)),TensorEdge((85, odata), (86, data)), imce_0_1 -> imce_0_2
      __builtin_IMCE_SEND(1, var239, 2, 0); // TensorEdge((85, odata), (103, data)),TensorEdge((85, odata), (86, data)), imce_0_1 -> imce_0_2
      // endgenerate: add standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 0 && wid == 2) { // imce_0_2
    // generate: TensorEdge((-75, min), (86, min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge((-75, min), (86, min)), min write
    // generate: TensorEdge((-76, max), (86, max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge((-76, max), (86, max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 256; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      __builtin_IMCE_SETFLAG(2);
      __builtin_IMCE_STANDBY(1, 2);
      __builtin_IMCE_SETFLAG(0);
      var42 = __builtin_IMCE_RECV(2); // TensorEdge((85, odata), (86, data)), imce_0_1 -> imce_0_2
      var43 = __builtin_IMCE_RECV(2); // TensorEdge((85, odata), (86, data)), imce_0_1 -> imce_0_2
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var42, 0, 15, 0);
      __builtin_IMCE_MM_QUANT(var43, 0, 15, 1);
      var44 = __builtin_IMCE_GET_QREG(0);
      var45 = __builtin_IMCE_GET_QREG(1);
      var46 = __builtin_IMCE_GET_QREG(2);
      var47 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize

      __builtin_IMCE_STANDBY(7, 1);
      __builtin_IMCE_SEND(1, var44, 0, 0); // TensorEdge((86, odata), (87, data)), imce_0_2 -> imce_1_2
      __builtin_IMCE_SEND(1, var45, 0, 0); // TensorEdge((86, odata), (87, data)), imce_0_2 -> imce_1_2
      __builtin_IMCE_SEND(1, var46, 0, 0); // TensorEdge((86, odata), (87, data)), imce_0_2 -> imce_1_2
      __builtin_IMCE_SEND(1, var47, 0, 0); // TensorEdge((86, odata), (87, data)), imce_0_2 -> imce_1_2
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
    // generate: TensorEdge((-94, min), (103, min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge((-94, min), (103, min)), min write
    // generate: TensorEdge((-95, max), (103, max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge((-95, max), (103, max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 256; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      __builtin_IMCE_SETFLAG(2);
      __builtin_IMCE_STANDBY(1, 2);
      __builtin_IMCE_SETFLAG(0);

      var50 = __builtin_IMCE_RECV(2); // TensorEdge((85, odata), (103, data)), imce_0_1 -> imce_1_1
      var51 = __builtin_IMCE_RECV(2); // TensorEdge((85, odata), (103, data)), imce_0_1 -> imce_1_1
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var50, 0, 15, 0);
      __builtin_IMCE_MM_QUANT(var51, 0, 15, 1);
      var52 = __builtin_IMCE_GET_QREG(0);
      var53 = __builtin_IMCE_GET_QREG(1);
      var54 = __builtin_IMCE_GET_QREG(2);
      var55 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      __builtin_IMCE_STANDBY(11, 1);
      __builtin_IMCE_SEND(9, var52, 0, 0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(9, var53, 0, 0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(9, var54, 0, 0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(9, var55, 0, 0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1
      // endgenerate: min_max_quantize_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 1 && wid == 2) { // imce_1_2
    // generate: TensorEdge(((89, -72), config), ((89, 81), config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge(((89, -72), config), ((89, 81), config)), config write
    // generate: TensorEdge(((89, -73), fused_scale), ((89, 83), fused_scale)), fused_scale write

    var57 = __builtin_IMCE_RECV(1);
    var58 = __builtin_IMCE_RECV(1);
    var59 = __builtin_IMCE_RECV(1);
    var60 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((89, -73), fused_scale), ((89, 83), fused_scale)), fused_scale write
    // generate: TensorEdge(((89, -74), fused_bias), ((89, 83), fused_bias)), fused_bias write

    var61 = __builtin_IMCE_RECV(1);
    var62 = __builtin_IMCE_RECV(1);
    var63 = __builtin_IMCE_RECV(1);
    var64 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((89, -74), fused_bias), ((89, 83), fused_bias)), fused_bias write
    // generate: conv exec7
    // generate: conv exec7_row_group0_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 8; i1++) { // generate : conv exec7_row_group0_outer_loop(iterate row offset)
      // generate: conv exec7_row_group0_col_group0
      // generate : conv exec7_row_group0_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 18; i2++) { // generate : load_block
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((87, odata), ((89, 81), data), 0), imce_0_2 -> imce_1_2

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
      var9 = __builtin_IMCE_RECV(2); // TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
      // endgenerate: TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
      // generate: TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
      var10 = __builtin_IMCE_RECV(2); // TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
      // endgenerate: TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
      // generate: TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
      var11 = __builtin_IMCE_RECV(2); // TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
      // endgenerate: TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
      // generate: TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
      var12 = __builtin_IMCE_RECV(2); // TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
      // endgenerate: TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
      // generate: add

      var69 = __builtin_IMCE_ADD(var5, var9, 15);
      var70 = __builtin_IMCE_ADD(var6, var10, 15);
      var71 = __builtin_IMCE_ADD(var7, var11, 15);
      var72 = __builtin_IMCE_ADD(var8, var12, 15);
      // endgenerate: add
      // generate: batch_norm

      var65 = __builtin_IMCE_MULTL(var69, var57, 15);
      var65 = __builtin_IMCE_ADD(var65, var61, 15);
      var66 = __builtin_IMCE_MULTL(var70, var58, 15);
      var66 = __builtin_IMCE_ADD(var66, var62, 15);
      var67 = __builtin_IMCE_MULTL(var71, var59, 15);
      var67 = __builtin_IMCE_ADD(var67, var63, 15);
      var68 = __builtin_IMCE_MULTL(var72, var60, 15);
      var68 = __builtin_IMCE_ADD(var68, var64, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var65, 2, 0); // TensorEdge(((89, 83), odata), (90, data)), imce_1_2 -> imce_2_2
      __builtin_IMCE_SEND(1, var66, 2, 0); // TensorEdge(((89, 83), odata), (90, data)), imce_1_2 -> imce_2_2
      __builtin_IMCE_SEND(1, var67, 2, 0); // TensorEdge(((89, 83), odata), (90, data)), imce_1_2 -> imce_2_2
      __builtin_IMCE_SEND(1, var68, 2, 0); // TensorEdge(((89, 83), odata), (90, data)), imce_1_2 -> imce_2_2
      // endgenerate : conv exec7_row_group0_col_group0
      // endgenerate: conv exec7_row_group0_col_group0
      // generate: conv exec7_row_group0_col_group1
      for (int i2 = 0; i2 < 7; i2++) { // generate : conv exec7_row_group0_col_group1

        // generate: load_block
        for (int i3 = 0; i3 < 2; i3++) { // generate : load_block
          for (int i4 = 0; i4 < 4; i4++) { // generate
            __builtin_IMCE_LOAD_LB(0); // TensorEdge((87, odata), ((89, 81), data), 0), imce_0_2 -> imce_1_2

          } // endgenerate
        } // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var5 = __builtin_IMCE_GET_CREG((short)0);
        var6 = __builtin_IMCE_GET_CREG((short)1);
        var7 = __builtin_IMCE_GET_CREG((short)2);
        var8 = __builtin_IMCE_GET_CREG((short)3);
        // generate: TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
        var9 = __builtin_IMCE_RECV(2); // TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
        // endgenerate: TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
        // generate: TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
        var10 = __builtin_IMCE_RECV(2); // TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
        // endgenerate: TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
        // generate: TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
        var11 = __builtin_IMCE_RECV(2); // TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
        // endgenerate: TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
        // generate: TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
        var12 = __builtin_IMCE_RECV(2); // TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
        // endgenerate: TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
        // generate: add

        var69 = __builtin_IMCE_ADD(var5, var9, 15);
        var70 = __builtin_IMCE_ADD(var6, var10, 15);
        var71 = __builtin_IMCE_ADD(var7, var11, 15);
        var72 = __builtin_IMCE_ADD(var8, var12, 15);
        // endgenerate: add
        // generate: batch_norm

        var65 = __builtin_IMCE_MULTL(var69, var57, 15);
        var65 = __builtin_IMCE_ADD(var65, var61, 15);
        var66 = __builtin_IMCE_MULTL(var70, var58, 15);
        var66 = __builtin_IMCE_ADD(var66, var62, 15);
        var67 = __builtin_IMCE_MULTL(var71, var59, 15);
        var67 = __builtin_IMCE_ADD(var67, var63, 15);
        var68 = __builtin_IMCE_MULTL(var72, var60, 15);
        var68 = __builtin_IMCE_ADD(var68, var64, 15);
        // endgenerate: batch_norm
        __builtin_IMCE_SEND(1, var65, 2, 0); // TensorEdge(((89, 83), odata), (90, data)), imce_1_2 -> imce_2_2
        __builtin_IMCE_SEND(1, var66, 2, 0); // TensorEdge(((89, 83), odata), (90, data)), imce_1_2 -> imce_2_2
        __builtin_IMCE_SEND(1, var67, 2, 0); // TensorEdge(((89, 83), odata), (90, data)), imce_1_2 -> imce_2_2
        __builtin_IMCE_SEND(1, var68, 2, 0); // TensorEdge(((89, 83), odata), (90, data)), imce_1_2 -> imce_2_2
      } // endgenerate : conv exec7_row_group0_col_group1
      // endgenerate: conv exec7_row_group0_col_group1
    } // endgenerate : conv exec7_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec7_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec7
    __builtin_IMCE_STOP();
  }
  else if (hid == 1 && wid == 3) { // imce_1_3
    // generate: TensorEdge((-78, config), (88, config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge((-78, config), (88, config)), config write
    // generate: conv exec6
    // generate: conv exec6_row_group0_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 8; i1++) { // generate : conv exec6_row_group0_outer_loop(iterate row offset)
      // generate: conv exec6_row_group0_col_group0
      // generate : conv exec6_row_group0_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 18; i2++) { // generate : load_block

        __builtin_IMCE_SETFLAG(1);
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((87, odata), (88, data), 1), imce_0_2 -> imce_1_3

        } // endgenerate
        __builtin_IMCE_SETFLAG(0);
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var1 = __builtin_IMCE_GET_CREG((short)0);
      var2 = __builtin_IMCE_GET_CREG((short)1);
      var3 = __builtin_IMCE_GET_CREG((short)2);
      var4 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(2, var1, 2, 0); // TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
      __builtin_IMCE_SEND(2, var2, 2, 0); // TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
      __builtin_IMCE_SEND(2, var3, 2, 0); // TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
      __builtin_IMCE_SEND(2, var4, 2, 0); // TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
      // endgenerate : conv exec6_row_group0_col_group0
      // endgenerate: conv exec6_row_group0_col_group0
      // generate: conv exec6_row_group0_col_group1
      for (int i2 = 0; i2 < 7; i2++) { // generate : conv exec6_row_group0_col_group1

        // generate: load_block
        for (int i3 = 0; i3 < 2; i3++) { // generate : load_block
          __builtin_IMCE_SETFLAG(1);
          for (int i4 = 0; i4 < 4; i4++) { // generate
            __builtin_IMCE_LOAD_LB(0); // TensorEdge((87, odata), (88, data), 1), imce_0_2 -> imce_1_3

          } // endgenerate
          __builtin_IMCE_SETFLAG(0);
        } // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var1 = __builtin_IMCE_GET_CREG((short)0);
        var2 = __builtin_IMCE_GET_CREG((short)1);
        var3 = __builtin_IMCE_GET_CREG((short)2);
        var4 = __builtin_IMCE_GET_CREG((short)3);
        __builtin_IMCE_SEND(2, var1, 2, 0); // TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
        __builtin_IMCE_SEND(2, var2, 2, 0); // TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
        __builtin_IMCE_SEND(2, var3, 2, 0); // TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
        __builtin_IMCE_SEND(2, var4, 2, 0); // TensorEdge((88, odata), ((89, 82), rhs)), imce_1_3 -> imce_1_2
      } // endgenerate : conv exec6_row_group0_col_group1
      // endgenerate: conv exec6_row_group0_col_group1
    } // endgenerate : conv exec6_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec6_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec6
    __builtin_IMCE_STOP();
  }
  else if (hid == 1 && wid == 4) { // imce_1_4
  }
  else if (hid == 2 && wid == 1) { // imce_2_1
    // generate: TensorEdge(((104, -90), config), ((104, 98), config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge(((104, -90), config), ((104, 98), config)), config write
    // generate: TensorEdge(((104, -91), fused_scale), ((104, 99), fused_scale)), fused_scale write

    var79 = __builtin_IMCE_RECV(1);
    var80 = __builtin_IMCE_RECV(1);
    var81 = __builtin_IMCE_RECV(1);
    var82 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((104, -91), fused_scale), ((104, 99), fused_scale)), fused_scale write
    // generate: TensorEdge(((104, -92), fused_bias), ((104, 99), fused_bias)), fused_bias write

    var83 = __builtin_IMCE_RECV(1);
    var84 = __builtin_IMCE_RECV(1);
    var85 = __builtin_IMCE_RECV(1);
    var86 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((104, -92), fused_bias), ((104, 99), fused_bias)), fused_bias write
    // generate: mult const

    var87 = __builtin_IMCE_RECV(3);
    var88 = __builtin_IMCE_RECV(3);
    var89 = __builtin_IMCE_RECV(3);
    var90 = __builtin_IMCE_RECV(3);
    // endgenerate: mult const
    // generate: add const

    var91 = __builtin_IMCE_RECV(2);
    var92 = __builtin_IMCE_RECV(2);
    var93 = __builtin_IMCE_RECV(2);
    var94 = __builtin_IMCE_RECV(2);
    // endgenerate: add const
    // generate: conv exec11
    // generate: conv exec11_row_group0_outer_loop(iterate row offset)
    // generate : conv exec11_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec11_row_group0_col_group0
    // generate : conv exec11_row_group0_col_group0. loop count == 1

    // generate: load_block
    // generate : load_block. loop count == 1
    __builtin_IMCE_SETFLAG(1);
    for (int i1 = 0; i1 < 4; i1++) { // generate
      __builtin_IMCE_LOAD_LB(0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1

    } // endgenerate
    __builtin_IMCE_SETFLAG(0);
    // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var33 = __builtin_IMCE_GET_CREG((short)0);
    var34 = __builtin_IMCE_GET_CREG((short)1);
    var35 = __builtin_IMCE_GET_CREG((short)2);
    var36 = __builtin_IMCE_GET_CREG((short)3);
    // generate: batch_norm

    var99 = __builtin_IMCE_MULTL(var33, var79, 15);
    var99 = __builtin_IMCE_ADD(var99, var83, 15);
    var100 = __builtin_IMCE_MULTL(var34, var80, 15);
    var100 = __builtin_IMCE_ADD(var100, var84, 15);
    var101 = __builtin_IMCE_MULTL(var35, var81, 15);
    var101 = __builtin_IMCE_ADD(var101, var85, 15);
    var102 = __builtin_IMCE_MULTL(var36, var82, 15);
    var102 = __builtin_IMCE_ADD(var102, var86, 15);
    // endgenerate: batch_norm
    // generate: multl

    var103 = __builtin_IMCE_MULTL(var87, var99, 15);
    var104 = __builtin_IMCE_MULTL(var88, var100, 15);
    var105 = __builtin_IMCE_MULTL(var89, var101, 15);
    var106 = __builtin_IMCE_MULTL(var90, var102, 15);
    // endgenerate: multl
    // generate: add

    var95 = __builtin_IMCE_ADD(var91, var103, 15);
    var96 = __builtin_IMCE_ADD(var92, var104, 15);
    var97 = __builtin_IMCE_ADD(var93, var105, 15);
    var98 = __builtin_IMCE_ADD(var94, var106, 15);
    // endgenerate: add
    __builtin_IMCE_SEND(1, var95, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_2_1 -> inode_2_0
    __builtin_IMCE_SEND(1, var96, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_2_1 -> inode_2_0
    __builtin_IMCE_SEND(1, var97, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_2_1 -> inode_2_0
    __builtin_IMCE_SEND(1, var98, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_2_1 -> inode_2_0
    // endgenerate : conv exec11_row_group0_col_group0
    // endgenerate: conv exec11_row_group0_col_group0
    // generate: conv exec11_row_group0_col_group1
    for (int i1 = 0; i1 < 7; i1++) { // generate : conv exec11_row_group0_col_group1

      // generate: load_block
      for (int i2 = 0; i2 < 2; i2++) { // generate : load_block
        __builtin_IMCE_SETFLAG(1);
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1

        } // endgenerate
        __builtin_IMCE_SETFLAG(0);
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var33 = __builtin_IMCE_GET_CREG((short)0);
      var34 = __builtin_IMCE_GET_CREG((short)1);
      var35 = __builtin_IMCE_GET_CREG((short)2);
      var36 = __builtin_IMCE_GET_CREG((short)3);
      // generate: batch_norm

      var99 = __builtin_IMCE_MULTL(var33, var79, 15);
      var99 = __builtin_IMCE_ADD(var99, var83, 15);
      var100 = __builtin_IMCE_MULTL(var34, var80, 15);
      var100 = __builtin_IMCE_ADD(var100, var84, 15);
      var101 = __builtin_IMCE_MULTL(var35, var81, 15);
      var101 = __builtin_IMCE_ADD(var101, var85, 15);
      var102 = __builtin_IMCE_MULTL(var36, var82, 15);
      var102 = __builtin_IMCE_ADD(var102, var86, 15);
      // endgenerate: batch_norm
      // generate: multl

      var103 = __builtin_IMCE_MULTL(var87, var99, 15);
      var104 = __builtin_IMCE_MULTL(var88, var100, 15);
      var105 = __builtin_IMCE_MULTL(var89, var101, 15);
      var106 = __builtin_IMCE_MULTL(var90, var102, 15);
      // endgenerate: multl
      // generate: add

      var95 = __builtin_IMCE_ADD(var91, var103, 15);
      var96 = __builtin_IMCE_ADD(var92, var104, 15);
      var97 = __builtin_IMCE_ADD(var93, var105, 15);
      var98 = __builtin_IMCE_ADD(var94, var106, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(1, var95, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_2_1 -> inode_2_0
      __builtin_IMCE_SEND(1, var96, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_2_1 -> inode_2_0
      __builtin_IMCE_SEND(1, var97, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_2_1 -> inode_2_0
      __builtin_IMCE_SEND(1, var98, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_2_1 -> inode_2_0
    } // endgenerate : conv exec11_row_group0_col_group1
    // endgenerate: conv exec11_row_group0_col_group1
    // endgenerate : conv exec11_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec11_row_group0_outer_loop(iterate row offset)
    // generate: conv exec11_row_group1_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 7; i1++) { // generate : conv exec11_row_group1_outer_loop(iterate row offset)
      // generate: conv exec11_row_group1_col_group0
      // generate : conv exec11_row_group1_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 18; i2++) { // generate : load_block
        __builtin_IMCE_SETFLAG(1);
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1

        } // endgenerate
        __builtin_IMCE_SETFLAG(0);
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var33 = __builtin_IMCE_GET_CREG((short)0);
      var34 = __builtin_IMCE_GET_CREG((short)1);
      var35 = __builtin_IMCE_GET_CREG((short)2);
      var36 = __builtin_IMCE_GET_CREG((short)3);
      // generate: batch_norm

      var99 = __builtin_IMCE_MULTL(var33, var79, 15);
      var99 = __builtin_IMCE_ADD(var99, var83, 15);
      var100 = __builtin_IMCE_MULTL(var34, var80, 15);
      var100 = __builtin_IMCE_ADD(var100, var84, 15);
      var101 = __builtin_IMCE_MULTL(var35, var81, 15);
      var101 = __builtin_IMCE_ADD(var101, var85, 15);
      var102 = __builtin_IMCE_MULTL(var36, var82, 15);
      var102 = __builtin_IMCE_ADD(var102, var86, 15);
      // endgenerate: batch_norm
      // generate: multl

      var103 = __builtin_IMCE_MULTL(var87, var99, 15);
      var104 = __builtin_IMCE_MULTL(var88, var100, 15);
      var105 = __builtin_IMCE_MULTL(var89, var101, 15);
      var106 = __builtin_IMCE_MULTL(var90, var102, 15);
      // endgenerate: multl
      // generate: add

      var95 = __builtin_IMCE_ADD(var91, var103, 15);
      var96 = __builtin_IMCE_ADD(var92, var104, 15);
      var97 = __builtin_IMCE_ADD(var93, var105, 15);
      var98 = __builtin_IMCE_ADD(var94, var106, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(1, var95, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_2_1 -> inode_2_0
      __builtin_IMCE_SEND(1, var96, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_2_1 -> inode_2_0
      __builtin_IMCE_SEND(1, var97, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_2_1 -> inode_2_0
      __builtin_IMCE_SEND(1, var98, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_2_1 -> inode_2_0
      // endgenerate : conv exec11_row_group1_col_group0
      // endgenerate: conv exec11_row_group1_col_group0
      // generate: conv exec11_row_group1_col_group1
      for (int i2 = 0; i2 < 7; i2++) { // generate : conv exec11_row_group1_col_group1

        // generate: load_block
        for (int i3 = 0; i3 < 2; i3++) { // generate : load_block
          __builtin_IMCE_SETFLAG(1);
          for (int i4 = 0; i4 < 4; i4++) { // generate
            __builtin_IMCE_LOAD_LB(0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1

          } // endgenerate
          __builtin_IMCE_SETFLAG(0);
        } // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var33 = __builtin_IMCE_GET_CREG((short)0);
        var34 = __builtin_IMCE_GET_CREG((short)1);
        var35 = __builtin_IMCE_GET_CREG((short)2);
        var36 = __builtin_IMCE_GET_CREG((short)3);
        // generate: batch_norm

        var99 = __builtin_IMCE_MULTL(var33, var79, 15);
        var99 = __builtin_IMCE_ADD(var99, var83, 15);
        var100 = __builtin_IMCE_MULTL(var34, var80, 15);
        var100 = __builtin_IMCE_ADD(var100, var84, 15);
        var101 = __builtin_IMCE_MULTL(var35, var81, 15);
        var101 = __builtin_IMCE_ADD(var101, var85, 15);
        var102 = __builtin_IMCE_MULTL(var36, var82, 15);
        var102 = __builtin_IMCE_ADD(var102, var86, 15);
        // endgenerate: batch_norm
        // generate: multl

        var103 = __builtin_IMCE_MULTL(var87, var99, 15);
        var104 = __builtin_IMCE_MULTL(var88, var100, 15);
        var105 = __builtin_IMCE_MULTL(var89, var101, 15);
        var106 = __builtin_IMCE_MULTL(var90, var102, 15);
        // endgenerate: multl
        // generate: add

        var95 = __builtin_IMCE_ADD(var91, var103, 15);
        var96 = __builtin_IMCE_ADD(var92, var104, 15);
        var97 = __builtin_IMCE_ADD(var93, var105, 15);
        var98 = __builtin_IMCE_ADD(var94, var106, 15);
        // endgenerate: add
        __builtin_IMCE_SEND(1, var95, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_2_1 -> inode_2_0
        __builtin_IMCE_SEND(1, var96, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_2_1 -> inode_2_0
        __builtin_IMCE_SEND(1, var97, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_2_1 -> inode_2_0
        __builtin_IMCE_SEND(1, var98, 2, 0); // TensorEdge(((104, 101), odata), (105, func_out1), 1), imce_2_1 -> inode_2_0
      } // endgenerate : conv exec11_row_group1_col_group1
      // endgenerate: conv exec11_row_group1_col_group1
    } // endgenerate : conv exec11_row_group1_outer_loop(iterate row offset)
    // endgenerate: conv exec11_row_group1_outer_loop(iterate row offset)
    // generate: conv exec11_tail_loop
    for (int i1 = 0; i1 < 17; i1++) { // generate : conv exec11_tail_loop
      __builtin_IMCE_SETFLAG(1);
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_RECV(0);
      }
      __builtin_IMCE_SETFLAG(0);
    } // endgenerate : conv exec11_tail_loop
    // endgenerate: conv exec11_tail_loop
    // endgenerate: conv exec11
    __builtin_IMCE_STOP();
  }
  else if (hid == 2 && wid == 2) { // imce_2_2
    // generate: TensorEdge((-79, min), (90, min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge((-79, min), (90, min)), min write
    // generate: TensorEdge((-80, max), (90, max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge((-80, max), (90, max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 64; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      var109 = __builtin_IMCE_RECV(2); // TensorEdge(((89, 83), odata), (90, data)), imce_1_2 -> imce_2_2
      var110 = __builtin_IMCE_RECV(2); // TensorEdge(((89, 83), odata), (90, data)), imce_1_2 -> imce_2_2
      var111 = __builtin_IMCE_RECV(2); // TensorEdge(((89, 83), odata), (90, data)), imce_1_2 -> imce_2_2
      var112 = __builtin_IMCE_RECV(2); // TensorEdge(((89, 83), odata), (90, data)), imce_1_2 -> imce_2_2
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var109, 0, 15, 0);
      __builtin_IMCE_MM_QUANT(var110, 0, 15, 1);
      __builtin_IMCE_MM_QUANT(var111, 0, 15, 2);
      __builtin_IMCE_MM_QUANT(var112, 0, 15, 3);
      var113 = __builtin_IMCE_GET_QREG(0);
      var114 = __builtin_IMCE_GET_QREG(1);
      var115 = __builtin_IMCE_GET_QREG(2);
      var116 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      __builtin_IMCE_SEND(1, var113, 0, 0); // TensorEdge((90, odata), (91, data)), imce_2_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var114, 0, 0); // TensorEdge((90, odata), (91, data)), imce_2_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var115, 0, 0); // TensorEdge((90, odata), (91, data)), imce_2_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var116, 0, 0); // TensorEdge((90, odata), (91, data)), imce_2_2 -> imce_3_1
      // endgenerate: min_max_quantize_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
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
    for (int i1 = 0; i1 < 10; i1++) { // generate : load_block
      __builtin_IMCE_SETFLAG(1);
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((91, odata), (95, data), 1), imce_2_2 -> imce_2_3

      } // endgenerate
      __builtin_IMCE_SETFLAG(0);
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var13 = __builtin_IMCE_GET_CREG((short)0);
    var14 = __builtin_IMCE_GET_CREG((short)1);
    var15 = __builtin_IMCE_GET_CREG((short)2);
    var16 = __builtin_IMCE_GET_CREG((short)3);
    __builtin_IMCE_SEND(2, var13, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    __builtin_IMCE_SEND(2, var14, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    __builtin_IMCE_SEND(2, var15, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    __builtin_IMCE_SEND(2, var16, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    // endgenerate : conv exec8_row_group0_col_group0
    // endgenerate: conv exec8_row_group0_col_group0
    // generate: conv exec8_row_group0_col_group1
    for (int i1 = 0; i1 < 6; i1++) { // generate : conv exec8_row_group0_col_group1

      // generate: load_block
      // generate : load_block. loop count == 1
      __builtin_IMCE_SETFLAG(1);
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((91, odata), (95, data), 1), imce_2_2 -> imce_2_3

      } // endgenerate
      __builtin_IMCE_SETFLAG(0);
      // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var13 = __builtin_IMCE_GET_CREG((short)0);
      var14 = __builtin_IMCE_GET_CREG((short)1);
      var15 = __builtin_IMCE_GET_CREG((short)2);
      var16 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(2, var13, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      __builtin_IMCE_SEND(2, var14, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      __builtin_IMCE_SEND(2, var15, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      __builtin_IMCE_SEND(2, var16, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    } // endgenerate : conv exec8_row_group0_col_group1
    // endgenerate: conv exec8_row_group0_col_group1
    // generate: conv exec8_row_group0_col_group2
    // generate : conv exec8_row_group0_col_group2. loop count == 1

    // generate: load_block
    // loop ignored with loop count == 0 : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var13 = __builtin_IMCE_GET_CREG((short)0);
    var14 = __builtin_IMCE_GET_CREG((short)1);
    var15 = __builtin_IMCE_GET_CREG((short)2);
    var16 = __builtin_IMCE_GET_CREG((short)3);
    __builtin_IMCE_SEND(2, var13, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    __builtin_IMCE_SEND(2, var14, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    __builtin_IMCE_SEND(2, var15, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    __builtin_IMCE_SEND(2, var16, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    // endgenerate : conv exec8_row_group0_col_group2
    // endgenerate: conv exec8_row_group0_col_group2
    // endgenerate : conv exec8_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec8_row_group0_outer_loop(iterate row offset)
    // generate: conv exec8_row_group1_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 6; i1++) { // generate : conv exec8_row_group1_outer_loop(iterate row offset)
      // generate: conv exec8_row_group1_col_group0
      // generate : conv exec8_row_group1_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 2; i2++) { // generate : load_block
        __builtin_IMCE_SETFLAG(1);
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((91, odata), (95, data), 1), imce_2_2 -> imce_2_3

        } // endgenerate
        __builtin_IMCE_SETFLAG(0);
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var13 = __builtin_IMCE_GET_CREG((short)0);
      var14 = __builtin_IMCE_GET_CREG((short)1);
      var15 = __builtin_IMCE_GET_CREG((short)2);
      var16 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(2, var13, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      __builtin_IMCE_SEND(2, var14, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      __builtin_IMCE_SEND(2, var15, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      __builtin_IMCE_SEND(2, var16, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // endgenerate : conv exec8_row_group1_col_group0
      // endgenerate: conv exec8_row_group1_col_group0
      // generate: conv exec8_row_group1_col_group1
      for (int i2 = 0; i2 < 6; i2++) { // generate : conv exec8_row_group1_col_group1

        // generate: load_block
        // generate : load_block. loop count == 1
        __builtin_IMCE_SETFLAG(1);
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((91, odata), (95, data), 1), imce_2_2 -> imce_2_3

        } // endgenerate
        __builtin_IMCE_SETFLAG(0);
        // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var13 = __builtin_IMCE_GET_CREG((short)0);
        var14 = __builtin_IMCE_GET_CREG((short)1);
        var15 = __builtin_IMCE_GET_CREG((short)2);
        var16 = __builtin_IMCE_GET_CREG((short)3);
        __builtin_IMCE_SEND(2, var13, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
        __builtin_IMCE_SEND(2, var14, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
        __builtin_IMCE_SEND(2, var15, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
        __builtin_IMCE_SEND(2, var16, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      } // endgenerate : conv exec8_row_group1_col_group1
      // endgenerate: conv exec8_row_group1_col_group1
      // generate: conv exec8_row_group1_col_group2
      // generate : conv exec8_row_group1_col_group2. loop count == 1

      // generate: load_block
      // loop ignored with loop count == 0 : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var13 = __builtin_IMCE_GET_CREG((short)0);
      var14 = __builtin_IMCE_GET_CREG((short)1);
      var15 = __builtin_IMCE_GET_CREG((short)2);
      var16 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(2, var13, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      __builtin_IMCE_SEND(2, var14, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      __builtin_IMCE_SEND(2, var15, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      __builtin_IMCE_SEND(2, var16, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // endgenerate : conv exec8_row_group1_col_group2
      // endgenerate: conv exec8_row_group1_col_group2
    } // endgenerate : conv exec8_row_group1_outer_loop(iterate row offset)
    // endgenerate: conv exec8_row_group1_outer_loop(iterate row offset)
    // generate: conv exec8_row_group2_outer_loop(iterate row offset)
    // generate : conv exec8_row_group2_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec8_row_group2_col_group0
    for (int i1 = 0; i1 < 8; i1++) { // generate : conv exec8_row_group2_col_group0

      // generate: load_block
      // loop ignored with loop count == 0 : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var13 = __builtin_IMCE_GET_CREG((short)0);
      var14 = __builtin_IMCE_GET_CREG((short)1);
      var15 = __builtin_IMCE_GET_CREG((short)2);
      var16 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(2, var13, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      __builtin_IMCE_SEND(2, var14, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      __builtin_IMCE_SEND(2, var15, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      __builtin_IMCE_SEND(2, var16, 2, 0); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    } // endgenerate : conv exec8_row_group2_col_group0
    // endgenerate: conv exec8_row_group2_col_group0
    // endgenerate : conv exec8_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec8_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec8
    __builtin_IMCE_STOP();
  }
  else if (hid == 2 && wid == 4) { // imce_2_4
  }
  else if (hid == 3 && wid == 1) { // imce_3_1
    // generate: TensorEdge(((97, -66), config), ((97, 77), config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge(((97, -66), config), ((97, 77), config)), config write
    // generate: TensorEdge(((97, -67), fused_scale), ((97, 79), fused_scale)), fused_scale write

    var123 = __builtin_IMCE_RECV(1);
    var124 = __builtin_IMCE_RECV(1);
    var125 = __builtin_IMCE_RECV(1);
    var126 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((97, -67), fused_scale), ((97, 79), fused_scale)), fused_scale write
    // generate: TensorEdge(((97, -68), fused_bias), ((97, 79), fused_bias)), fused_bias write

    var127 = __builtin_IMCE_RECV(1);
    var128 = __builtin_IMCE_RECV(1);
    var129 = __builtin_IMCE_RECV(1);
    var130 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((97, -68), fused_bias), ((97, 79), fused_bias)), fused_bias write
    // generate: conv exec10
    // generate: conv exec10_row_group0_outer_loop(iterate row offset)
    // generate : conv exec10_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec10_row_group0_col_group0
    // generate : conv exec10_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 10; i1++) { // generate : load_block
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((91, odata), ((97, 77), data), 2), imce_2_2 -> imce_3_1

      } // endgenerate
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var25 = __builtin_IMCE_GET_CREG((short)0);
    var26 = __builtin_IMCE_GET_CREG((short)1);
    var27 = __builtin_IMCE_GET_CREG((short)2);
    var28 = __builtin_IMCE_GET_CREG((short)3);
    // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    var29 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    var30 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    var31 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    var32 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    // generate: add

    var135 = __builtin_IMCE_ADD(var29, var25, 15);
    var136 = __builtin_IMCE_ADD(var30, var26, 15);
    var137 = __builtin_IMCE_ADD(var31, var27, 15);
    var138 = __builtin_IMCE_ADD(var32, var28, 15);
    // endgenerate: add
    // generate: batch_norm

    var131 = __builtin_IMCE_MULTL(var135, var123, 15);
    var131 = __builtin_IMCE_ADD(var131, var127, 15);
    var132 = __builtin_IMCE_MULTL(var136, var124, 15);
    var132 = __builtin_IMCE_ADD(var132, var128, 15);
    var133 = __builtin_IMCE_MULTL(var137, var125, 15);
    var133 = __builtin_IMCE_ADD(var133, var129, 15);
    var134 = __builtin_IMCE_MULTL(var138, var126, 15);
    var134 = __builtin_IMCE_ADD(var134, var130, 15);
    // endgenerate: batch_norm
    __builtin_IMCE_SEND(1, var131, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var132, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var133, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var134, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
    // endgenerate : conv exec10_row_group0_col_group0
    // endgenerate: conv exec10_row_group0_col_group0
    // generate: conv exec10_row_group0_col_group1
    for (int i1 = 0; i1 < 6; i1++) { // generate : conv exec10_row_group0_col_group1

      // generate: load_block
      // generate : load_block. loop count == 1
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((91, odata), ((97, 77), data), 2), imce_2_2 -> imce_3_1

      } // endgenerate
      // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var25 = __builtin_IMCE_GET_CREG((short)0);
      var26 = __builtin_IMCE_GET_CREG((short)1);
      var27 = __builtin_IMCE_GET_CREG((short)2);
      var28 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      var29 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      var30 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      var31 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      var32 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // generate: add

      var135 = __builtin_IMCE_ADD(var29, var25, 15);
      var136 = __builtin_IMCE_ADD(var30, var26, 15);
      var137 = __builtin_IMCE_ADD(var31, var27, 15);
      var138 = __builtin_IMCE_ADD(var32, var28, 15);
      // endgenerate: add
      // generate: batch_norm

      var131 = __builtin_IMCE_MULTL(var135, var123, 15);
      var131 = __builtin_IMCE_ADD(var131, var127, 15);
      var132 = __builtin_IMCE_MULTL(var136, var124, 15);
      var132 = __builtin_IMCE_ADD(var132, var128, 15);
      var133 = __builtin_IMCE_MULTL(var137, var125, 15);
      var133 = __builtin_IMCE_ADD(var133, var129, 15);
      var134 = __builtin_IMCE_MULTL(var138, var126, 15);
      var134 = __builtin_IMCE_ADD(var134, var130, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var131, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var132, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var133, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var134, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
    } // endgenerate : conv exec10_row_group0_col_group1
    // endgenerate: conv exec10_row_group0_col_group1
    // generate: conv exec10_row_group0_col_group2
    // generate : conv exec10_row_group0_col_group2. loop count == 1

    // generate: load_block
    // loop ignored with loop count == 0 : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var25 = __builtin_IMCE_GET_CREG((short)0);
    var26 = __builtin_IMCE_GET_CREG((short)1);
    var27 = __builtin_IMCE_GET_CREG((short)2);
    var28 = __builtin_IMCE_GET_CREG((short)3);
    // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    var29 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    var30 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    var31 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    var32 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    // generate: add

    var135 = __builtin_IMCE_ADD(var29, var25, 15);
    var136 = __builtin_IMCE_ADD(var30, var26, 15);
    var137 = __builtin_IMCE_ADD(var31, var27, 15);
    var138 = __builtin_IMCE_ADD(var32, var28, 15);
    // endgenerate: add
    // generate: batch_norm

    var131 = __builtin_IMCE_MULTL(var135, var123, 15);
    var131 = __builtin_IMCE_ADD(var131, var127, 15);
    var132 = __builtin_IMCE_MULTL(var136, var124, 15);
    var132 = __builtin_IMCE_ADD(var132, var128, 15);
    var133 = __builtin_IMCE_MULTL(var137, var125, 15);
    var133 = __builtin_IMCE_ADD(var133, var129, 15);
    var134 = __builtin_IMCE_MULTL(var138, var126, 15);
    var134 = __builtin_IMCE_ADD(var134, var130, 15);
    // endgenerate: batch_norm
    __builtin_IMCE_SEND(1, var131, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var132, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var133, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
    __builtin_IMCE_SEND(1, var134, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
    // endgenerate : conv exec10_row_group0_col_group2
    // endgenerate: conv exec10_row_group0_col_group2
    // endgenerate : conv exec10_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec10_row_group0_outer_loop(iterate row offset)
    // generate: conv exec10_row_group1_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 6; i1++) { // generate : conv exec10_row_group1_outer_loop(iterate row offset)
      // generate: conv exec10_row_group1_col_group0
      // generate : conv exec10_row_group1_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 2; i2++) { // generate : load_block
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((91, odata), ((97, 77), data), 2), imce_2_2 -> imce_3_1

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var25 = __builtin_IMCE_GET_CREG((short)0);
      var26 = __builtin_IMCE_GET_CREG((short)1);
      var27 = __builtin_IMCE_GET_CREG((short)2);
      var28 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      var29 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      var30 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      var31 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      var32 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // generate: add

      var135 = __builtin_IMCE_ADD(var29, var25, 15);
      var136 = __builtin_IMCE_ADD(var30, var26, 15);
      var137 = __builtin_IMCE_ADD(var31, var27, 15);
      var138 = __builtin_IMCE_ADD(var32, var28, 15);
      // endgenerate: add
      // generate: batch_norm

      var131 = __builtin_IMCE_MULTL(var135, var123, 15);
      var131 = __builtin_IMCE_ADD(var131, var127, 15);
      var132 = __builtin_IMCE_MULTL(var136, var124, 15);
      var132 = __builtin_IMCE_ADD(var132, var128, 15);
      var133 = __builtin_IMCE_MULTL(var137, var125, 15);
      var133 = __builtin_IMCE_ADD(var133, var129, 15);
      var134 = __builtin_IMCE_MULTL(var138, var126, 15);
      var134 = __builtin_IMCE_ADD(var134, var130, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var131, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var132, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var133, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var134, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
      // endgenerate : conv exec10_row_group1_col_group0
      // endgenerate: conv exec10_row_group1_col_group0
      // generate: conv exec10_row_group1_col_group1
      for (int i2 = 0; i2 < 6; i2++) { // generate : conv exec10_row_group1_col_group1

        // generate: load_block
        // generate : load_block. loop count == 1
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((91, odata), ((97, 77), data), 2), imce_2_2 -> imce_3_1

        } // endgenerate
        // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var25 = __builtin_IMCE_GET_CREG((short)0);
        var26 = __builtin_IMCE_GET_CREG((short)1);
        var27 = __builtin_IMCE_GET_CREG((short)2);
        var28 = __builtin_IMCE_GET_CREG((short)3);
        // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
        var29 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
        // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
        // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
        var30 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
        // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
        // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
        var31 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
        // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
        // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
        var32 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
        // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
        // generate: add

        var135 = __builtin_IMCE_ADD(var29, var25, 15);
        var136 = __builtin_IMCE_ADD(var30, var26, 15);
        var137 = __builtin_IMCE_ADD(var31, var27, 15);
        var138 = __builtin_IMCE_ADD(var32, var28, 15);
        // endgenerate: add
        // generate: batch_norm

        var131 = __builtin_IMCE_MULTL(var135, var123, 15);
        var131 = __builtin_IMCE_ADD(var131, var127, 15);
        var132 = __builtin_IMCE_MULTL(var136, var124, 15);
        var132 = __builtin_IMCE_ADD(var132, var128, 15);
        var133 = __builtin_IMCE_MULTL(var137, var125, 15);
        var133 = __builtin_IMCE_ADD(var133, var129, 15);
        var134 = __builtin_IMCE_MULTL(var138, var126, 15);
        var134 = __builtin_IMCE_ADD(var134, var130, 15);
        // endgenerate: batch_norm
        __builtin_IMCE_SEND(1, var131, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
        __builtin_IMCE_SEND(1, var132, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
        __builtin_IMCE_SEND(1, var133, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
        __builtin_IMCE_SEND(1, var134, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
      } // endgenerate : conv exec10_row_group1_col_group1
      // endgenerate: conv exec10_row_group1_col_group1
      // generate: conv exec10_row_group1_col_group2
      // generate : conv exec10_row_group1_col_group2. loop count == 1

      // generate: load_block
      // loop ignored with loop count == 0 : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var25 = __builtin_IMCE_GET_CREG((short)0);
      var26 = __builtin_IMCE_GET_CREG((short)1);
      var27 = __builtin_IMCE_GET_CREG((short)2);
      var28 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      var29 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      var30 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      var31 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      var32 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // generate: add

      var135 = __builtin_IMCE_ADD(var29, var25, 15);
      var136 = __builtin_IMCE_ADD(var30, var26, 15);
      var137 = __builtin_IMCE_ADD(var31, var27, 15);
      var138 = __builtin_IMCE_ADD(var32, var28, 15);
      // endgenerate: add
      // generate: batch_norm

      var131 = __builtin_IMCE_MULTL(var135, var123, 15);
      var131 = __builtin_IMCE_ADD(var131, var127, 15);
      var132 = __builtin_IMCE_MULTL(var136, var124, 15);
      var132 = __builtin_IMCE_ADD(var132, var128, 15);
      var133 = __builtin_IMCE_MULTL(var137, var125, 15);
      var133 = __builtin_IMCE_ADD(var133, var129, 15);
      var134 = __builtin_IMCE_MULTL(var138, var126, 15);
      var134 = __builtin_IMCE_ADD(var134, var130, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var131, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var132, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var133, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var134, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
      // endgenerate : conv exec10_row_group1_col_group2
      // endgenerate: conv exec10_row_group1_col_group2
    } // endgenerate : conv exec10_row_group1_outer_loop(iterate row offset)
    // endgenerate: conv exec10_row_group1_outer_loop(iterate row offset)
    // generate: conv exec10_row_group2_outer_loop(iterate row offset)
    // generate : conv exec10_row_group2_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec10_row_group2_col_group0
    for (int i1 = 0; i1 < 8; i1++) { // generate : conv exec10_row_group2_col_group0

      // generate: load_block
      // loop ignored with loop count == 0 : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var25 = __builtin_IMCE_GET_CREG((short)0);
      var26 = __builtin_IMCE_GET_CREG((short)1);
      var27 = __builtin_IMCE_GET_CREG((short)2);
      var28 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      var29 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      var30 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      var31 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // generate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      var32 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // endgenerate: TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // generate: add

      var135 = __builtin_IMCE_ADD(var29, var25, 15);
      var136 = __builtin_IMCE_ADD(var30, var26, 15);
      var137 = __builtin_IMCE_ADD(var31, var27, 15);
      var138 = __builtin_IMCE_ADD(var32, var28, 15);
      // endgenerate: add
      // generate: batch_norm

      var131 = __builtin_IMCE_MULTL(var135, var123, 15);
      var131 = __builtin_IMCE_ADD(var131, var127, 15);
      var132 = __builtin_IMCE_MULTL(var136, var124, 15);
      var132 = __builtin_IMCE_ADD(var132, var128, 15);
      var133 = __builtin_IMCE_MULTL(var137, var125, 15);
      var133 = __builtin_IMCE_ADD(var133, var129, 15);
      var134 = __builtin_IMCE_MULTL(var138, var126, 15);
      var134 = __builtin_IMCE_ADD(var134, var130, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var131, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var132, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var133, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var134, 2, 0); // TensorEdge(((97, 79), odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
    } // endgenerate : conv exec10_row_group2_col_group0
    // endgenerate: conv exec10_row_group2_col_group0
    // endgenerate : conv exec10_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec10_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec10
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 2) { // imce_3_2
    // generate: TensorEdge(((96, -84), config), ((96, 92), config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge(((96, -84), config), ((96, 92), config)), config write
    // generate: conv exec9
    // generate: conv exec9_row_group0_outer_loop(iterate row offset)
    // generate : conv exec9_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec9_row_group0_col_group0
    // generate : conv exec9_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 10; i1++) { // generate : load_block
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((91, odata), ((96, 92), data), 0), imce_2_2 -> imce_3_2

      } // endgenerate
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var17 = __builtin_IMCE_GET_CREG((short)0);
    var18 = __builtin_IMCE_GET_CREG((short)1);
    var19 = __builtin_IMCE_GET_CREG((short)2);
    var20 = __builtin_IMCE_GET_CREG((short)3);
    // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    var21 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    var22 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    var23 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    var24 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    // generate: add

    var140 = __builtin_IMCE_ADD(var17, var21, 15);
    var141 = __builtin_IMCE_ADD(var18, var22, 15);
    var142 = __builtin_IMCE_ADD(var19, var23, 15);
    var143 = __builtin_IMCE_ADD(var20, var24, 15);
    // endgenerate: add
    __builtin_IMCE_SEND(2, var140, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(2, var141, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(2, var142, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(2, var143, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    // endgenerate : conv exec9_row_group0_col_group0
    // endgenerate: conv exec9_row_group0_col_group0
    // generate: conv exec9_row_group0_col_group1
    for (int i1 = 0; i1 < 6; i1++) { // generate : conv exec9_row_group0_col_group1

      // generate: load_block
      // generate : load_block. loop count == 1
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((91, odata), ((96, 92), data), 0), imce_2_2 -> imce_3_2

      } // endgenerate
      // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var17 = __builtin_IMCE_GET_CREG((short)0);
      var18 = __builtin_IMCE_GET_CREG((short)1);
      var19 = __builtin_IMCE_GET_CREG((short)2);
      var20 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      var21 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      var22 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      var23 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      var24 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // generate: add

      var140 = __builtin_IMCE_ADD(var17, var21, 15);
      var141 = __builtin_IMCE_ADD(var18, var22, 15);
      var142 = __builtin_IMCE_ADD(var19, var23, 15);
      var143 = __builtin_IMCE_ADD(var20, var24, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(2, var140, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var141, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var142, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var143, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    } // endgenerate : conv exec9_row_group0_col_group1
    // endgenerate: conv exec9_row_group0_col_group1
    // generate: conv exec9_row_group0_col_group2
    // generate : conv exec9_row_group0_col_group2. loop count == 1

    // generate: load_block
    // loop ignored with loop count == 0 : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var17 = __builtin_IMCE_GET_CREG((short)0);
    var18 = __builtin_IMCE_GET_CREG((short)1);
    var19 = __builtin_IMCE_GET_CREG((short)2);
    var20 = __builtin_IMCE_GET_CREG((short)3);
    // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    var21 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    var22 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    var23 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    var24 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
    // generate: add

    var140 = __builtin_IMCE_ADD(var17, var21, 15);
    var141 = __builtin_IMCE_ADD(var18, var22, 15);
    var142 = __builtin_IMCE_ADD(var19, var23, 15);
    var143 = __builtin_IMCE_ADD(var20, var24, 15);
    // endgenerate: add
    __builtin_IMCE_SEND(2, var140, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(2, var141, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(2, var142, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(2, var143, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    // endgenerate : conv exec9_row_group0_col_group2
    // endgenerate: conv exec9_row_group0_col_group2
    // endgenerate : conv exec9_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec9_row_group0_outer_loop(iterate row offset)
    // generate: conv exec9_row_group1_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 6; i1++) { // generate : conv exec9_row_group1_outer_loop(iterate row offset)
      // generate: conv exec9_row_group1_col_group0
      // generate : conv exec9_row_group1_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 2; i2++) { // generate : load_block
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((91, odata), ((96, 92), data), 0), imce_2_2 -> imce_3_2

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var17 = __builtin_IMCE_GET_CREG((short)0);
      var18 = __builtin_IMCE_GET_CREG((short)1);
      var19 = __builtin_IMCE_GET_CREG((short)2);
      var20 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      var21 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      var22 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      var23 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      var24 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // generate: add

      var140 = __builtin_IMCE_ADD(var17, var21, 15);
      var141 = __builtin_IMCE_ADD(var18, var22, 15);
      var142 = __builtin_IMCE_ADD(var19, var23, 15);
      var143 = __builtin_IMCE_ADD(var20, var24, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(2, var140, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var141, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var142, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var143, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // endgenerate : conv exec9_row_group1_col_group0
      // endgenerate: conv exec9_row_group1_col_group0
      // generate: conv exec9_row_group1_col_group1
      for (int i2 = 0; i2 < 6; i2++) { // generate : conv exec9_row_group1_col_group1

        // generate: load_block
        // generate : load_block. loop count == 1
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((91, odata), ((96, 92), data), 0), imce_2_2 -> imce_3_2

        } // endgenerate
        // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var17 = __builtin_IMCE_GET_CREG((short)0);
        var18 = __builtin_IMCE_GET_CREG((short)1);
        var19 = __builtin_IMCE_GET_CREG((short)2);
        var20 = __builtin_IMCE_GET_CREG((short)3);
        // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
        var21 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
        // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
        // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
        var22 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
        // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
        // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
        var23 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
        // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
        // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
        var24 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
        // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
        // generate: add

        var140 = __builtin_IMCE_ADD(var17, var21, 15);
        var141 = __builtin_IMCE_ADD(var18, var22, 15);
        var142 = __builtin_IMCE_ADD(var19, var23, 15);
        var143 = __builtin_IMCE_ADD(var20, var24, 15);
        // endgenerate: add
        __builtin_IMCE_SEND(2, var140, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
        __builtin_IMCE_SEND(2, var141, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
        __builtin_IMCE_SEND(2, var142, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
        __builtin_IMCE_SEND(2, var143, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      } // endgenerate : conv exec9_row_group1_col_group1
      // endgenerate: conv exec9_row_group1_col_group1
      // generate: conv exec9_row_group1_col_group2
      // generate : conv exec9_row_group1_col_group2. loop count == 1

      // generate: load_block
      // loop ignored with loop count == 0 : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var17 = __builtin_IMCE_GET_CREG((short)0);
      var18 = __builtin_IMCE_GET_CREG((short)1);
      var19 = __builtin_IMCE_GET_CREG((short)2);
      var20 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      var21 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      var22 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      var23 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      var24 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // generate: add

      var140 = __builtin_IMCE_ADD(var17, var21, 15);
      var141 = __builtin_IMCE_ADD(var18, var22, 15);
      var142 = __builtin_IMCE_ADD(var19, var23, 15);
      var143 = __builtin_IMCE_ADD(var20, var24, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(2, var140, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var141, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var142, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var143, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      // endgenerate : conv exec9_row_group1_col_group2
      // endgenerate: conv exec9_row_group1_col_group2
    } // endgenerate : conv exec9_row_group1_outer_loop(iterate row offset)
    // endgenerate: conv exec9_row_group1_outer_loop(iterate row offset)
    // generate: conv exec9_row_group2_outer_loop(iterate row offset)
    // generate : conv exec9_row_group2_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec9_row_group2_col_group0
    for (int i1 = 0; i1 < 8; i1++) { // generate : conv exec9_row_group2_col_group0

      // generate: load_block
      // loop ignored with loop count == 0 : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var17 = __builtin_IMCE_GET_CREG((short)0);
      var18 = __builtin_IMCE_GET_CREG((short)1);
      var19 = __builtin_IMCE_GET_CREG((short)2);
      var20 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      var21 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      var22 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      var23 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // generate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      var24 = __builtin_IMCE_RECV(2); // TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // endgenerate: TensorEdge((95, odata), ((96, 93), rhs)), imce_2_3 -> imce_3_2
      // generate: add

      var140 = __builtin_IMCE_ADD(var17, var21, 15);
      var141 = __builtin_IMCE_ADD(var18, var22, 15);
      var142 = __builtin_IMCE_ADD(var19, var23, 15);
      var143 = __builtin_IMCE_ADD(var20, var24, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(2, var140, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var141, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var142, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(2, var143, 2, 0); // TensorEdge(((96, 93), odata), ((97, 78), lhs)), imce_3_2 -> imce_3_1
    } // endgenerate : conv exec9_row_group2_col_group0
    // endgenerate: conv exec9_row_group2_col_group0
    // endgenerate : conv exec9_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec9_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec9
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 3) { // imce_3_3
  }
  else if (hid == 3 && wid == 4) { // imce_3_4
  }
}
