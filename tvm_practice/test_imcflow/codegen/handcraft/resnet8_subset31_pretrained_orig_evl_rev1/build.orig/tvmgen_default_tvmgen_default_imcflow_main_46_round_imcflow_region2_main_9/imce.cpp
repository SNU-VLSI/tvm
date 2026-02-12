#include "../common_decl.h"
void tvmgen_default_tvmgen_default_imcflow_main_46_round_imcflow_region2_main_9() {
  int hid = __builtin_IMCE_GET_CORE_HID();
  int wid = __builtin_IMCE_GET_CORE_WID();
  short16 var1; // (ConvBlock(gid: 62), 0)
  short16 var2; // (ConvBlock(gid: 62), 1)
  short16 var3; // (ConvBlock(gid: 62), 2)
  short16 var4; // (ConvBlock(gid: 62), 3)
  short16 var5; // (ConvBlock(gid: 65), 0)
  short16 var6; // (ConvBlock(gid: 65), 1)
  short16 var7; // (ConvBlock(gid: 65), 2)
  short16 var8; // (ConvBlock(gid: 65), 3)
  short16 var9; // (ConvBlock(gid: 54), 0)
  short16 var10; // (ConvBlock(gid: 54), 1)
  short16 var11; // (ConvBlock(gid: 54), 2)
  short16 var12; // (ConvBlock(gid: 54), 3)
  short16 var13; // (TensorEdge((65, odata), ((66, 55), rhs)), 0)
  short16 var14; // (TensorEdge((65, odata), ((66, 55), rhs)), 1)
  short16 var15; // (TensorEdge((65, odata), ((66, 55), rhs)), 2)
  short16 var16; // (TensorEdge((65, odata), ((66, 55), rhs)), 3)
  short16 var17; // (ConvBlock(gid: 73), 0)
  short16 var18; // (ConvBlock(gid: 73), 1)
  short16 var19; // (ConvBlock(gid: 73), 2)
  short16 var20; // (ConvBlock(gid: 73), 3)
  short16 var21; // (TensorEdge(((74, -53), fused_scale), ((74, 68), fused_scale)), 0)
  short16 var22; // (TensorEdge(((74, -53), fused_scale), ((74, 68), fused_scale)), 1)
  short16 var23; // (TensorEdge(((74, -54), fused_bias), ((74, 68), fused_bias)), 0)
  short16 var24; // (TensorEdge(((74, -54), fused_bias), ((74, 68), fused_bias)), 1)
  short16 var25; // (TensorEdge(((74, -52), scale), ((74, 69), rhs)), 0)
  short16 var26; // (TensorEdge(((74, -52), scale), ((74, 69), rhs)), 1)
  short16 var27; // (TensorEdge((-59, scale), ((74, 70), rhs)), 0)
  short16 var28; // (TensorEdge((-59, scale), ((74, 70), rhs)), 1)
  short16 var29; // (TensorEdge((73, odata), ((74, 68), data)), 0)
  short16 var30; // (TensorEdge((73, odata), ((74, 68), data)), 1)
  short16 var31; // (AddBlock(gid: 70), 0)
  short16 var32; // (AddBlock(gid: 70), 1)
  short16 var33; // (BatchNormBlock(gid: 68), 0)
  short16 var34; // (BatchNormBlock(gid: 68), 0, 'mult_result')
  short16 var35; // (BatchNormBlock(gid: 68), 1)
  short16 var36; // (BatchNormBlock(gid: 68), 1, 'mult_result')
  short16 var37; // ((BatchNormBlock(gid: 68), 0), 'L')
  short16 var38; // ((BatchNormBlock(gid: 68), 0), 'H')
  short16 var39; // ((BatchNormBlock(gid: 68), 0), 'neg1')
  short16 var40; // ((BatchNormBlock(gid: 68), 0), 'const_7fff')
  short16 var41; // ((BatchNormBlock(gid: 68), 0), 'H_sign')
  short16 var42; // ((BatchNormBlock(gid: 68), 0), 'L_sign')
  short16 var43; // ((BatchNormBlock(gid: 68), 0), 'mismatch')
  short16 var44; // ((BatchNormBlock(gid: 68), 0), 'saturate_val')
  short16 var45; // ((BatchNormBlock(gid: 68), 0), 'not_mismatch')
  short16 var46; // ((BatchNormBlock(gid: 68), 0), 'part1')
  short16 var47; // ((BatchNormBlock(gid: 68), 0), 'part2')
  short16 var48; // ((BatchNormBlock(gid: 68), 1), 'L')
  short16 var49; // ((BatchNormBlock(gid: 68), 1), 'H')
  short16 var50; // ((BatchNormBlock(gid: 68), 1), 'neg1')
  short16 var51; // ((BatchNormBlock(gid: 68), 1), 'const_7fff')
  short16 var52; // ((BatchNormBlock(gid: 68), 1), 'H_sign')
  short16 var53; // ((BatchNormBlock(gid: 68), 1), 'L_sign')
  short16 var54; // ((BatchNormBlock(gid: 68), 1), 'mismatch')
  short16 var55; // ((BatchNormBlock(gid: 68), 1), 'saturate_val')
  short16 var56; // ((BatchNormBlock(gid: 68), 1), 'not_mismatch')
  short16 var57; // ((BatchNormBlock(gid: 68), 1), 'part1')
  short16 var58; // ((BatchNormBlock(gid: 68), 1), 'part2')
  short16 var59; // (MultlBlock(gid: 69), 0)
  short16 var60; // (MultlBlock(gid: 69), 1)
  short16 var61; // ((MultlBlock(gid: 69), 0), 'L')
  short16 var62; // ((MultlBlock(gid: 69), 0), 'H')
  short16 var63; // ((MultlBlock(gid: 69), 0), 'neg1')
  short16 var64; // ((MultlBlock(gid: 69), 0), 'const_7fff')
  short16 var65; // ((MultlBlock(gid: 69), 0), 'H_sign')
  short16 var66; // ((MultlBlock(gid: 69), 0), 'L_sign')
  short16 var67; // ((MultlBlock(gid: 69), 0), 'mismatch')
  short16 var68; // ((MultlBlock(gid: 69), 0), 'saturate_val')
  short16 var69; // ((MultlBlock(gid: 69), 0), 'not_mismatch')
  short16 var70; // ((MultlBlock(gid: 69), 0), 'part1')
  short16 var71; // ((MultlBlock(gid: 69), 0), 'part2')
  short16 var72; // ((MultlBlock(gid: 69), 1), 'L')
  short16 var73; // ((MultlBlock(gid: 69), 1), 'H')
  short16 var74; // ((MultlBlock(gid: 69), 1), 'neg1')
  short16 var75; // ((MultlBlock(gid: 69), 1), 'const_7fff')
  short16 var76; // ((MultlBlock(gid: 69), 1), 'H_sign')
  short16 var77; // ((MultlBlock(gid: 69), 1), 'L_sign')
  short16 var78; // ((MultlBlock(gid: 69), 1), 'mismatch')
  short16 var79; // ((MultlBlock(gid: 69), 1), 'saturate_val')
  short16 var80; // ((MultlBlock(gid: 69), 1), 'not_mismatch')
  short16 var81; // ((MultlBlock(gid: 69), 1), 'part1')
  short16 var82; // ((MultlBlock(gid: 69), 1), 'part2')
  short16 var83; // (TensorEdge((-58, config), (73, config)), 0)
  short16 var84; // (TensorEdge((72, odata), (73, data)), 0)
  short16 var85; // (TensorEdge((72, odata), (73, data)), 1)
  short16 var86; // (TensorEdge((72, odata), (73, data)), 2)
  short16 var87; // (TensorEdge((72, odata), (73, data)), 3)
  short16 var88; // (TensorEdge((-55, min), (72, min)), 0)
  short16 var89; // (TensorEdge((-56, max), (72, max)), 0)
  short16 var90; // (TensorEdge((60, odata), (72, data)), 0)
  short16 var91; // (MinmaxQuantBlock(gid: 72), 0)
  short16 var92; // (MinmaxQuantBlock(gid: 72), 1)
  short16 var93; // (MinmaxQuantBlock(gid: 72), 2)
  short16 var94; // (MinmaxQuantBlock(gid: 72), 3)
  short16 var95; // (TensorEdge((-45, config), (62, config)), 0)
  short16 var96; // (TensorEdge((61, odata), (62, data)), 0)
  short16 var97; // (TensorEdge((61, odata), (62, data)), 1)
  short16 var98; // (TensorEdge((61, odata), (62, data)), 2)
  short16 var99; // (TensorEdge((61, odata), (62, data)), 3)
  short16 var100; // (TensorEdge((-42, min), (61, min)), 0)
  short16 var101; // (TensorEdge((-43, max), (61, max)), 0)
  short16 var102; // (TensorEdge((60, odata), (61, data)), 0)
  short16 var103; // (MinmaxQuantBlock(gid: 61), 0)
  short16 var104; // (MinmaxQuantBlock(gid: 61), 1)
  short16 var105; // (MinmaxQuantBlock(gid: 61), 2)
  short16 var106; // (MinmaxQuantBlock(gid: 61), 3)
  short16 var107; // (TensorEdge((-31, odata), (60, lhs)), 0)
  short16 var108; // (TensorEdge((-32, odata), (60, rhs)), 0)
  short16 var109; // (AddBlock(gid: 60), 0)
  short16 var110; // (TensorEdge(((63, -38), fused_scale), ((63, 57), fused_scale)), 0)
  short16 var111; // (TensorEdge(((63, -38), fused_scale), ((63, 57), fused_scale)), 1)
  short16 var112; // (TensorEdge(((63, -39), fused_bias), ((63, 57), fused_bias)), 0)
  short16 var113; // (TensorEdge(((63, -39), fused_bias), ((63, 57), fused_bias)), 1)
  short16 var114; // (TensorEdge(((63, -40), min), ((63, 58), min)), 0)
  short16 var115; // (TensorEdge(((63, -41), max), ((63, 58), max)), 0)
  short16 var116; // (TensorEdge((62, odata), ((63, 57), data)), 0)
  short16 var117; // (TensorEdge((62, odata), ((63, 57), data)), 1)
  short16 var118; // (TensorEdge((62, odata), ((63, 57), data)), 2)
  short16 var119; // (TensorEdge((62, odata), ((63, 57), data)), 3)
  short16 var120; // (MinmaxQuantBlock(gid: 58), 0)
  short16 var121; // (MinmaxQuantBlock(gid: 58), 1)
  short16 var122; // (MinmaxQuantBlock(gid: 58), 2)
  short16 var123; // (MinmaxQuantBlock(gid: 58), 3)
  short16 var124; // (BatchNormBlock(gid: 57), 0)
  short16 var125; // (BatchNormBlock(gid: 57), 0, 'mult_result')
  short16 var126; // (BatchNormBlock(gid: 57), 1)
  short16 var127; // (BatchNormBlock(gid: 57), 1, 'mult_result')
  short16 var128; // ((BatchNormBlock(gid: 57), 0), 'L')
  short16 var129; // ((BatchNormBlock(gid: 57), 0), 'H')
  short16 var130; // ((BatchNormBlock(gid: 57), 0), 'neg1')
  short16 var131; // ((BatchNormBlock(gid: 57), 0), 'const_7fff')
  short16 var132; // ((BatchNormBlock(gid: 57), 0), 'H_sign')
  short16 var133; // ((BatchNormBlock(gid: 57), 0), 'L_sign')
  short16 var134; // ((BatchNormBlock(gid: 57), 0), 'mismatch')
  short16 var135; // ((BatchNormBlock(gid: 57), 0), 'saturate_val')
  short16 var136; // ((BatchNormBlock(gid: 57), 0), 'not_mismatch')
  short16 var137; // ((BatchNormBlock(gid: 57), 0), 'part1')
  short16 var138; // ((BatchNormBlock(gid: 57), 0), 'part2')
  short16 var139; // ((BatchNormBlock(gid: 57), 1), 'L')
  short16 var140; // ((BatchNormBlock(gid: 57), 1), 'H')
  short16 var141; // ((BatchNormBlock(gid: 57), 1), 'neg1')
  short16 var142; // ((BatchNormBlock(gid: 57), 1), 'const_7fff')
  short16 var143; // ((BatchNormBlock(gid: 57), 1), 'H_sign')
  short16 var144; // ((BatchNormBlock(gid: 57), 1), 'L_sign')
  short16 var145; // ((BatchNormBlock(gid: 57), 1), 'mismatch')
  short16 var146; // ((BatchNormBlock(gid: 57), 1), 'saturate_val')
  short16 var147; // ((BatchNormBlock(gid: 57), 1), 'not_mismatch')
  short16 var148; // ((BatchNormBlock(gid: 57), 1), 'part1')
  short16 var149; // ((BatchNormBlock(gid: 57), 1), 'part2')
  short16 var150; // (TensorEdge((-47, config), (65, config)), 0)
  short16 var151; // (TensorEdge((64, odata), (65, data), 1), 0)
  short16 var152; // (TensorEdge((64, odata), (65, data), 1), 1)
  short16 var153; // (TensorEdge((64, odata), (65, data), 1), 2)
  short16 var154; // (TensorEdge((64, odata), (65, data), 1), 3)
  short16 var155; // (TensorEdge((-48, fused_scale), (67, fused_scale)), 0)
  short16 var156; // (TensorEdge((-48, fused_scale), (67, fused_scale)), 1)
  short16 var157; // (TensorEdge((-49, fused_bias), (67, fused_bias)), 0)
  short16 var158; // (TensorEdge((-49, fused_bias), (67, fused_bias)), 1)
  short16 var159; // (TensorEdge(((66, 55), odata), (67, data)), 0)
  short16 var160; // (TensorEdge(((66, 55), odata), (67, data)), 1)
  short16 var161; // (BatchNormBlock(gid: 67), 0)
  short16 var162; // (BatchNormBlock(gid: 67), 1)
  short16 var163; // (BatchNormBlock(gid: 67), 0, 'mult_result')
  short16 var164; // (BatchNormBlock(gid: 67), 1, 'mult_result')
  short16 var165; // ((BatchNormBlock(gid: 67), 0), 'L')
  short16 var166; // ((BatchNormBlock(gid: 67), 0), 'H')
  short16 var167; // ((BatchNormBlock(gid: 67), 0), 'neg1')
  short16 var168; // ((BatchNormBlock(gid: 67), 0), 'const_7fff')
  short16 var169; // ((BatchNormBlock(gid: 67), 0), 'H_sign')
  short16 var170; // ((BatchNormBlock(gid: 67), 0), 'L_sign')
  short16 var171; // ((BatchNormBlock(gid: 67), 0), 'mismatch')
  short16 var172; // ((BatchNormBlock(gid: 67), 0), 'saturate_val')
  short16 var173; // ((BatchNormBlock(gid: 67), 0), 'not_mismatch')
  short16 var174; // ((BatchNormBlock(gid: 67), 0), 'part1')
  short16 var175; // ((BatchNormBlock(gid: 67), 0), 'part2')
  short16 var176; // ((BatchNormBlock(gid: 67), 1), 'L')
  short16 var177; // ((BatchNormBlock(gid: 67), 1), 'H')
  short16 var178; // ((BatchNormBlock(gid: 67), 1), 'neg1')
  short16 var179; // ((BatchNormBlock(gid: 67), 1), 'const_7fff')
  short16 var180; // ((BatchNormBlock(gid: 67), 1), 'H_sign')
  short16 var181; // ((BatchNormBlock(gid: 67), 1), 'L_sign')
  short16 var182; // ((BatchNormBlock(gid: 67), 1), 'mismatch')
  short16 var183; // ((BatchNormBlock(gid: 67), 1), 'saturate_val')
  short16 var184; // ((BatchNormBlock(gid: 67), 1), 'not_mismatch')
  short16 var185; // ((BatchNormBlock(gid: 67), 1), 'part1')
  short16 var186; // ((BatchNormBlock(gid: 67), 1), 'part2')
  short16 var187; // (TensorEdge(((66, -36), config), ((66, 54), config)), 0)
  short16 var188; // (AddBlock(gid: 55), 0)
  short16 var189; // (AddBlock(gid: 55), 1)
  short16 var190; // (AddBlock(gid: 55), 2)
  short16 var191; // (AddBlock(gid: 55), 3)
  if (hid == 0 && wid == 1) { // imce_0_1
    // generate: TensorEdge(((74, -53), fused_scale), ((74, 68), fused_scale)), fused_scale write

    var21 = __builtin_IMCE_RECV(1);
    var22 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((74, -53), fused_scale), ((74, 68), fused_scale)), fused_scale write
    // generate: TensorEdge(((74, -54), fused_bias), ((74, 68), fused_bias)), fused_bias write

    var23 = __builtin_IMCE_RECV(1);
    var24 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((74, -54), fused_bias), ((74, 68), fused_bias)), fused_bias write
    // generate: mult const

    var25 = __builtin_IMCE_RECV(1);
    var26 = __builtin_IMCE_RECV(1);
    // endgenerate: mult const
    // generate: add const

    var27 = __builtin_IMCE_RECV(1);
    var28 = __builtin_IMCE_RECV(1);
    // endgenerate: add const
    // generate: call_created_loop
    for (int i1 = 0; i1 < 512; i1++) { // generate : call_created_loop
      // generate: imcflow.vecops_wrapper

      var29 = __builtin_IMCE_RECV(2); // TensorEdge((73, odata), ((74, 68), data)), imce_0_2 -> imce_0_1
      var30 = __builtin_IMCE_RECV(2); // TensorEdge((73, odata), ((74, 68), data)), imce_0_2 -> imce_0_1
      // generate: imcflow.vecops_block
      // generate: batch_norm


      var37 = __builtin_IMCE_MULTL(var29, var21, 15);
      var38 = __builtin_IMCE_MULTH(var29, var21, 15);
      var39 = __builtin_IMCE_SUBI(0, 1);
      var40 = __builtin_IMCE_SRLI(var39, 1);
      var41 = __builtin_IMCE_SRAI(var38, 15);
      var42 = __builtin_IMCE_SRAI(var37, 15);
      var43 = __builtin_IMCE_XOR(var41, var42, 15);
      var44 = __builtin_IMCE_XOR(var41, var40, 15);
      var45 = __builtin_IMCE_XOR(var43, var39, 15);
      var46 = __builtin_IMCE_AND(var43, var44, 15);
      var47 = __builtin_IMCE_AND(var45, var37, 15);
      var34 = __builtin_IMCE_OR(var46, var47, 15);
      var33 = __builtin_IMCE_ADD(var34, var23, 15);

      var48 = __builtin_IMCE_MULTL(var30, var22, 15);
      var49 = __builtin_IMCE_MULTH(var30, var22, 15);
      var50 = __builtin_IMCE_SUBI(0, 1);
      var51 = __builtin_IMCE_SRLI(var50, 1);
      var52 = __builtin_IMCE_SRAI(var49, 15);
      var53 = __builtin_IMCE_SRAI(var48, 15);
      var54 = __builtin_IMCE_XOR(var52, var53, 15);
      var55 = __builtin_IMCE_XOR(var52, var51, 15);
      var56 = __builtin_IMCE_XOR(var54, var50, 15);
      var57 = __builtin_IMCE_AND(var54, var55, 15);
      var58 = __builtin_IMCE_AND(var56, var48, 15);
      var36 = __builtin_IMCE_OR(var57, var58, 15);
      var35 = __builtin_IMCE_ADD(var36, var24, 15);
      // endgenerate: batch_norm
      // generate: multl


      var61 = __builtin_IMCE_MULTL(var25, var33, 15);
      var62 = __builtin_IMCE_MULTH(var25, var33, 15);
      var63 = __builtin_IMCE_SUBI(0, 1);
      var64 = __builtin_IMCE_SRLI(var63, 1);
      var65 = __builtin_IMCE_SRAI(var62, 15);
      var66 = __builtin_IMCE_SRAI(var61, 15);
      var67 = __builtin_IMCE_XOR(var65, var66, 15);
      var68 = __builtin_IMCE_XOR(var65, var64, 15);
      var69 = __builtin_IMCE_XOR(var67, var63, 15);
      var70 = __builtin_IMCE_AND(var67, var68, 15);
      var71 = __builtin_IMCE_AND(var69, var61, 15);
      var59 = __builtin_IMCE_OR(var70, var71, 15);

      var72 = __builtin_IMCE_MULTL(var26, var35, 15);
      var73 = __builtin_IMCE_MULTH(var26, var35, 15);
      var74 = __builtin_IMCE_SUBI(0, 1);
      var75 = __builtin_IMCE_SRLI(var74, 1);
      var76 = __builtin_IMCE_SRAI(var73, 15);
      var77 = __builtin_IMCE_SRAI(var72, 15);
      var78 = __builtin_IMCE_XOR(var76, var77, 15);
      var79 = __builtin_IMCE_XOR(var76, var75, 15);
      var80 = __builtin_IMCE_XOR(var78, var74, 15);
      var81 = __builtin_IMCE_AND(var78, var79, 15);
      var82 = __builtin_IMCE_AND(var80, var72, 15);
      var60 = __builtin_IMCE_OR(var81, var82, 15);
      // endgenerate: multl
      // generate: add

      var31 = __builtin_IMCE_ADD(var27, var59, 15);
      var32 = __builtin_IMCE_ADD(var28, var60, 15);
      // endgenerate: add
      // endgenerate: imcflow.vecops_block
      __builtin_IMCE_SEND(1, var31, 2, 0); // TensorEdge(((74, 70), odata), (75, func_out1), 1), imce_0_1 -> inode_2_0
      __builtin_IMCE_SEND(1, var32, 2, 0); // TensorEdge(((74, 70), odata), (75, func_out1), 1), imce_0_1 -> inode_2_0
      // endgenerate: imcflow.vecops_wrapper
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 0 && wid == 2) { // imce_0_2
    // generate: TensorEdge((-58, config), (73, config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge((-58, config), (73, config)), config write
    // generate: conv exec5
    // generate: conv exec5_row_group0_outer_loop(iterate row offset)
    // generate : conv exec5_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec5_row_group0_col_group0
    // generate : conv exec5_row_group0_col_group0. loop count == 1

    // generate: load_block
    // generate : load_block. loop count == 1
    for (int i1 = 0; i1 < 4; i1++) { // generate
      __builtin_IMCE_LOAD_LB(0); // TensorEdge((72, odata), (73, data)), imce_0_3 -> imce_0_2

    } // endgenerate
    // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var17 = __builtin_IMCE_GET_CREG((short)0);
    var18 = __builtin_IMCE_GET_CREG((short)1);
    var19 = __builtin_IMCE_GET_CREG((short)2);
    var20 = __builtin_IMCE_GET_CREG((short)3);
    __builtin_IMCE_SEND(2, var17, 2, 0); // TensorEdge((73, odata), ((74, 68), data)), imce_0_2 -> imce_0_1
    __builtin_IMCE_SEND(2, var18, 2, 0); // TensorEdge((73, odata), ((74, 68), data)), imce_0_2 -> imce_0_1
    __builtin_IMCE_SEND(2, var19, 2, 0); // TensorEdge((73, odata), ((74, 68), data)), imce_0_2 -> imce_0_1
    __builtin_IMCE_SEND(2, var20, 2, 0); // TensorEdge((73, odata), ((74, 68), data)), imce_0_2 -> imce_0_1
    // endgenerate : conv exec5_row_group0_col_group0
    // endgenerate: conv exec5_row_group0_col_group0
    // generate: conv exec5_row_group0_col_group1
    for (int i1 = 0; i1 < 15; i1++) { // generate : conv exec5_row_group0_col_group1

      // generate: load_block
      for (int i2 = 0; i2 < 2; i2++) { // generate : load_block
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((72, odata), (73, data)), imce_0_3 -> imce_0_2

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var17 = __builtin_IMCE_GET_CREG((short)0);
      var18 = __builtin_IMCE_GET_CREG((short)1);
      var19 = __builtin_IMCE_GET_CREG((short)2);
      var20 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(2, var17, 2, 0); // TensorEdge((73, odata), ((74, 68), data)), imce_0_2 -> imce_0_1
      __builtin_IMCE_SEND(2, var18, 2, 0); // TensorEdge((73, odata), ((74, 68), data)), imce_0_2 -> imce_0_1
      __builtin_IMCE_SEND(2, var19, 2, 0); // TensorEdge((73, odata), ((74, 68), data)), imce_0_2 -> imce_0_1
      __builtin_IMCE_SEND(2, var20, 2, 0); // TensorEdge((73, odata), ((74, 68), data)), imce_0_2 -> imce_0_1
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
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((72, odata), (73, data)), imce_0_3 -> imce_0_2

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var17 = __builtin_IMCE_GET_CREG((short)0);
      var18 = __builtin_IMCE_GET_CREG((short)1);
      var19 = __builtin_IMCE_GET_CREG((short)2);
      var20 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(2, var17, 2, 0); // TensorEdge((73, odata), ((74, 68), data)), imce_0_2 -> imce_0_1
      __builtin_IMCE_SEND(2, var18, 2, 0); // TensorEdge((73, odata), ((74, 68), data)), imce_0_2 -> imce_0_1
      __builtin_IMCE_SEND(2, var19, 2, 0); // TensorEdge((73, odata), ((74, 68), data)), imce_0_2 -> imce_0_1
      __builtin_IMCE_SEND(2, var20, 2, 0); // TensorEdge((73, odata), ((74, 68), data)), imce_0_2 -> imce_0_1
      // endgenerate : conv exec5_row_group1_col_group0
      // endgenerate: conv exec5_row_group1_col_group0
      // generate: conv exec5_row_group1_col_group1
      for (int i2 = 0; i2 < 15; i2++) { // generate : conv exec5_row_group1_col_group1

        // generate: load_block
        for (int i3 = 0; i3 < 2; i3++) { // generate : load_block
          for (int i4 = 0; i4 < 4; i4++) { // generate
            __builtin_IMCE_LOAD_LB(0); // TensorEdge((72, odata), (73, data)), imce_0_3 -> imce_0_2

          } // endgenerate
        } // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var17 = __builtin_IMCE_GET_CREG((short)0);
        var18 = __builtin_IMCE_GET_CREG((short)1);
        var19 = __builtin_IMCE_GET_CREG((short)2);
        var20 = __builtin_IMCE_GET_CREG((short)3);
        __builtin_IMCE_SEND(2, var17, 2, 0); // TensorEdge((73, odata), ((74, 68), data)), imce_0_2 -> imce_0_1
        __builtin_IMCE_SEND(2, var18, 2, 0); // TensorEdge((73, odata), ((74, 68), data)), imce_0_2 -> imce_0_1
        __builtin_IMCE_SEND(2, var19, 2, 0); // TensorEdge((73, odata), ((74, 68), data)), imce_0_2 -> imce_0_1
        __builtin_IMCE_SEND(2, var20, 2, 0); // TensorEdge((73, odata), ((74, 68), data)), imce_0_2 -> imce_0_1
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
  else if (hid == 0 && wid == 3) { // imce_0_3
    // generate: TensorEdge((-55, min), (72, min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge((-55, min), (72, min)), min write
    // generate: TensorEdge((-56, max), (72, max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge((-56, max), (72, max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 1024; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      var90 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), (72, data)), imce_1_3 -> imce_0_3
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var90, 0, 15, 0);
      var91 = __builtin_IMCE_GET_QREG(0);
      var92 = __builtin_IMCE_GET_QREG(1);
      var93 = __builtin_IMCE_GET_QREG(2);
      var94 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      __builtin_IMCE_SEND(3, var91, 0, 0); // TensorEdge((72, odata), (73, data)), imce_0_3 -> imce_0_2
      __builtin_IMCE_SEND(3, var92, 0, 0); // TensorEdge((72, odata), (73, data)), imce_0_3 -> imce_0_2
      __builtin_IMCE_SEND(3, var93, 0, 0); // TensorEdge((72, odata), (73, data)), imce_0_3 -> imce_0_2
      __builtin_IMCE_SEND(3, var94, 0, 0); // TensorEdge((72, odata), (73, data)), imce_0_3 -> imce_0_2
      // endgenerate: min_max_quantize_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 0 && wid == 4) { // imce_0_4
  }
  else if (hid == 1 && wid == 1) { // imce_1_1
    // generate: TensorEdge((-45, config), (62, config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge((-45, config), (62, config)), config write
    // generate: conv exec2
    // generate: conv exec2_row_group0_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 16; i1++) { // generate : conv exec2_row_group0_outer_loop(iterate row offset)
      // generate: conv exec2_row_group0_col_group0
      // generate : conv exec2_row_group0_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 34; i2++) { // generate : load_block
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((61, odata), (62, data)), imce_1_2 -> imce_1_1

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var1 = __builtin_IMCE_GET_CREG((short)0);
      var2 = __builtin_IMCE_GET_CREG((short)1);
      var3 = __builtin_IMCE_GET_CREG((short)2);
      var4 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(1, var1, 2, 0); // TensorEdge((62, odata), ((63, 57), data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(1, var2, 2, 0); // TensorEdge((62, odata), ((63, 57), data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(1, var3, 2, 0); // TensorEdge((62, odata), ((63, 57), data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(1, var4, 2, 0); // TensorEdge((62, odata), ((63, 57), data)), imce_1_1 -> imce_2_1
      // endgenerate : conv exec2_row_group0_col_group0
      // endgenerate: conv exec2_row_group0_col_group0
      // generate: conv exec2_row_group0_col_group1
      for (int i2 = 0; i2 < 15; i2++) { // generate : conv exec2_row_group0_col_group1

        // generate: load_block
        for (int i3 = 0; i3 < 2; i3++) { // generate : load_block
          for (int i4 = 0; i4 < 4; i4++) { // generate
            __builtin_IMCE_LOAD_LB(0); // TensorEdge((61, odata), (62, data)), imce_1_2 -> imce_1_1

          } // endgenerate
        } // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var1 = __builtin_IMCE_GET_CREG((short)0);
        var2 = __builtin_IMCE_GET_CREG((short)1);
        var3 = __builtin_IMCE_GET_CREG((short)2);
        var4 = __builtin_IMCE_GET_CREG((short)3);
        __builtin_IMCE_SEND(1, var1, 2, 0); // TensorEdge((62, odata), ((63, 57), data)), imce_1_1 -> imce_2_1
        __builtin_IMCE_SEND(1, var2, 2, 0); // TensorEdge((62, odata), ((63, 57), data)), imce_1_1 -> imce_2_1
        __builtin_IMCE_SEND(1, var3, 2, 0); // TensorEdge((62, odata), ((63, 57), data)), imce_1_1 -> imce_2_1
        __builtin_IMCE_SEND(1, var4, 2, 0); // TensorEdge((62, odata), ((63, 57), data)), imce_1_1 -> imce_2_1
      } // endgenerate : conv exec2_row_group0_col_group1
      // endgenerate: conv exec2_row_group0_col_group1
    } // endgenerate : conv exec2_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec2_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec2
    __builtin_IMCE_STOP();
  }
  else if (hid == 1 && wid == 2) { // imce_1_2
    // generate: TensorEdge((-42, min), (61, min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge((-42, min), (61, min)), min write
    // generate: TensorEdge((-43, max), (61, max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge((-43, max), (61, max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 1024; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      var102 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), (61, data)), imce_1_3 -> imce_1_2
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var102, 0, 15, 0);
      var103 = __builtin_IMCE_GET_QREG(0);
      var104 = __builtin_IMCE_GET_QREG(1);
      var105 = __builtin_IMCE_GET_QREG(2);
      var106 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      __builtin_IMCE_SEND(1, var103, 0, 0); // TensorEdge((61, odata), (62, data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var104, 0, 0); // TensorEdge((61, odata), (62, data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var105, 0, 0); // TensorEdge((61, odata), (62, data)), imce_1_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var106, 0, 0); // TensorEdge((61, odata), (62, data)), imce_1_2 -> imce_1_1
      // endgenerate: min_max_quantize_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 1 && wid == 3) { // imce_1_3
    // generate: call_created_loop
    for (int i1 = 0; i1 < 1024; i1++) { // generate : call_created_loop
      // generate: add standalone

      var107 = __builtin_IMCE_RECV(2); // TensorEdge((-31, odata), (60, lhs)), inode_0_0 -> imce_1_3
      var108 = __builtin_IMCE_RECV(3); // TensorEdge((-32, odata), (60, rhs)), inode_1_0 -> imce_1_3
      // generate: add

      var109 = __builtin_IMCE_ADD(var107, var108, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(1, var109, 2, 0); // TensorEdge((60, odata), (61, data)),TensorEdge((60, odata), (72, data)), imce_1_3 -> imce_1_2
      // endgenerate: add standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 1 && wid == 4) { // imce_1_4
  }
  else if (hid == 2 && wid == 1) { // imce_2_1
    // generate: TensorEdge(((63, -38), fused_scale), ((63, 57), fused_scale)), fused_scale write

    var110 = __builtin_IMCE_RECV(1);
    var111 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((63, -38), fused_scale), ((63, 57), fused_scale)), fused_scale write
    // generate: TensorEdge(((63, -39), fused_bias), ((63, 57), fused_bias)), fused_bias write

    var112 = __builtin_IMCE_RECV(1);
    var113 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((63, -39), fused_bias), ((63, 57), fused_bias)), fused_bias write
    // generate: TensorEdge(((63, -40), min), ((63, 58), min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge(((63, -40), min), ((63, 58), min)), min write
    // generate: TensorEdge(((63, -41), max), ((63, 58), max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge(((63, -41), max), ((63, 58), max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 256; i1++) { // generate : call_created_loop
      // generate: imcflow.preop-minmax_wrapper

      var116 = __builtin_IMCE_RECV(2); // TensorEdge((62, odata), ((63, 57), data)), imce_1_1 -> imce_2_1
      var117 = __builtin_IMCE_RECV(2); // TensorEdge((62, odata), ((63, 57), data)), imce_1_1 -> imce_2_1
      var118 = __builtin_IMCE_RECV(2); // TensorEdge((62, odata), ((63, 57), data)), imce_1_1 -> imce_2_1
      var119 = __builtin_IMCE_RECV(2); // TensorEdge((62, odata), ((63, 57), data)), imce_1_1 -> imce_2_1
      // generate: imcflow.preop-minmax_block
      // generate: batch_norm


      var128 = __builtin_IMCE_MULTL(var116, var110, 15);
      var129 = __builtin_IMCE_MULTH(var116, var110, 15);
      var130 = __builtin_IMCE_SUBI(0, 1);
      var131 = __builtin_IMCE_SRLI(var130, 1);
      var132 = __builtin_IMCE_SRAI(var129, 15);
      var133 = __builtin_IMCE_SRAI(var128, 15);
      var134 = __builtin_IMCE_XOR(var132, var133, 15);
      var135 = __builtin_IMCE_XOR(var132, var131, 15);
      var136 = __builtin_IMCE_XOR(var134, var130, 15);
      var137 = __builtin_IMCE_AND(var134, var135, 15);
      var138 = __builtin_IMCE_AND(var136, var128, 15);
      var125 = __builtin_IMCE_OR(var137, var138, 15);
      var124 = __builtin_IMCE_ADD(var125, var112, 15);

      var139 = __builtin_IMCE_MULTL(var117, var111, 15);
      var140 = __builtin_IMCE_MULTH(var117, var111, 15);
      var141 = __builtin_IMCE_SUBI(0, 1);
      var142 = __builtin_IMCE_SRLI(var141, 1);
      var143 = __builtin_IMCE_SRAI(var140, 15);
      var144 = __builtin_IMCE_SRAI(var139, 15);
      var145 = __builtin_IMCE_XOR(var143, var144, 15);
      var146 = __builtin_IMCE_XOR(var143, var142, 15);
      var147 = __builtin_IMCE_XOR(var145, var141, 15);
      var148 = __builtin_IMCE_AND(var145, var146, 15);
      var149 = __builtin_IMCE_AND(var147, var139, 15);
      var127 = __builtin_IMCE_OR(var148, var149, 15);
      var126 = __builtin_IMCE_ADD(var127, var113, 15);
      // endgenerate: batch_norm
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var124, 0, 15, 0);
      __builtin_IMCE_MM_QUANT(var126, 0, 15, 1);
      var120 = __builtin_IMCE_GET_QREG(0);
      var121 = __builtin_IMCE_GET_QREG(1);
      var122 = __builtin_IMCE_GET_QREG(2);
      var123 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      // endgenerate: imcflow.preop-minmax_block
      __builtin_IMCE_SEND(1, var120, 0, 0); // TensorEdge(((63, 58), odata), (64, data)), imce_2_1 -> imce_3_2
      __builtin_IMCE_SEND(1, var121, 0, 0); // TensorEdge(((63, 58), odata), (64, data)), imce_2_1 -> imce_3_2
      __builtin_IMCE_SEND(1, var122, 0, 0); // TensorEdge(((63, 58), odata), (64, data)), imce_2_1 -> imce_3_2
      __builtin_IMCE_SEND(1, var123, 0, 0); // TensorEdge(((63, 58), odata), (64, data)), imce_2_1 -> imce_3_2
      // endgenerate: imcflow.preop-minmax_wrapper
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 2 && wid == 2) { // imce_2_2
    // generate: TensorEdge((-47, config), (65, config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge((-47, config), (65, config)), config write
    // generate: conv exec3
    // generate: conv exec3_row_group0_outer_loop(iterate row offset)
    // generate : conv exec3_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec3_row_group0_col_group0
    // generate : conv exec3_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 18; i1++) { // generate : load_block
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((64, odata), (65, data), 1), imce_2_1 -> imce_2_2

      } // endgenerate
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var5 = __builtin_IMCE_GET_CREG((short)0);
    var6 = __builtin_IMCE_GET_CREG((short)1);
    var7 = __builtin_IMCE_GET_CREG((short)2);
    var8 = __builtin_IMCE_GET_CREG((short)3);
    __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    // endgenerate : conv exec3_row_group0_col_group0
    // endgenerate: conv exec3_row_group0_col_group0
    // generate: conv exec3_row_group0_col_group1
    for (int i1 = 0; i1 < 14; i1++) { // generate : conv exec3_row_group0_col_group1

      // generate: load_block
      // generate : load_block. loop count == 1
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((64, odata), (65, data), 1), imce_2_1 -> imce_2_2

      } // endgenerate
      // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
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
    __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
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
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((64, odata), (65, data), 1), imce_2_1 -> imce_2_2

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // endgenerate : conv exec3_row_group1_col_group0
      // endgenerate: conv exec3_row_group1_col_group0
      // generate: conv exec3_row_group1_col_group1
      for (int i2 = 0; i2 < 14; i2++) { // generate : conv exec3_row_group1_col_group1

        // generate: load_block
        // generate : load_block. loop count == 1
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((64, odata), (65, data), 1), imce_2_1 -> imce_2_2

        } // endgenerate
        // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var5 = __builtin_IMCE_GET_CREG((short)0);
        var6 = __builtin_IMCE_GET_CREG((short)1);
        var7 = __builtin_IMCE_GET_CREG((short)2);
        var8 = __builtin_IMCE_GET_CREG((short)3);
        __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
        __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
        __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
        __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
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
      __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
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
      __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    } // endgenerate : conv exec3_row_group2_col_group0
    // endgenerate: conv exec3_row_group2_col_group0
    // endgenerate : conv exec3_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec3_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec3
    __builtin_IMCE_STOP();
  }
  else if (hid == 2 && wid == 3) { // imce_2_3
  }
  else if (hid == 2 && wid == 4) { // imce_2_4
  }
  else if (hid == 3 && wid == 1) { // imce_3_1
    // generate: TensorEdge((-48, fused_scale), (67, fused_scale)), fused_scale write

    var155 = __builtin_IMCE_RECV(1);
    var156 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge((-48, fused_scale), (67, fused_scale)), fused_scale write
    // generate: TensorEdge((-49, fused_bias), (67, fused_bias)), fused_bias write

    var157 = __builtin_IMCE_RECV(1);
    var158 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge((-49, fused_bias), (67, fused_bias)), fused_bias write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 512; i1++) { // generate : call_created_loop
      // generate: bn_standalone

      var159 = __builtin_IMCE_RECV(2); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
      var160 = __builtin_IMCE_RECV(2); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
      // generate: batch_norm


      var165 = __builtin_IMCE_MULTL(var159, var155, 15);
      var166 = __builtin_IMCE_MULTH(var159, var155, 15);
      var167 = __builtin_IMCE_SUBI(0, 1);
      var168 = __builtin_IMCE_SRLI(var167, 1);
      var169 = __builtin_IMCE_SRAI(var166, 15);
      var170 = __builtin_IMCE_SRAI(var165, 15);
      var171 = __builtin_IMCE_XOR(var169, var170, 15);
      var172 = __builtin_IMCE_XOR(var169, var168, 15);
      var173 = __builtin_IMCE_XOR(var171, var167, 15);
      var174 = __builtin_IMCE_AND(var171, var172, 15);
      var175 = __builtin_IMCE_AND(var173, var165, 15);
      var163 = __builtin_IMCE_OR(var174, var175, 15);
      var161 = __builtin_IMCE_ADD(var163, var157, 15);

      var176 = __builtin_IMCE_MULTL(var160, var156, 15);
      var177 = __builtin_IMCE_MULTH(var160, var156, 15);
      var178 = __builtin_IMCE_SUBI(0, 1);
      var179 = __builtin_IMCE_SRLI(var178, 1);
      var180 = __builtin_IMCE_SRAI(var177, 15);
      var181 = __builtin_IMCE_SRAI(var176, 15);
      var182 = __builtin_IMCE_XOR(var180, var181, 15);
      var183 = __builtin_IMCE_XOR(var180, var179, 15);
      var184 = __builtin_IMCE_XOR(var182, var178, 15);
      var185 = __builtin_IMCE_AND(var182, var183, 15);
      var186 = __builtin_IMCE_AND(var184, var176, 15);
      var164 = __builtin_IMCE_OR(var185, var186, 15);
      var162 = __builtin_IMCE_ADD(var164, var158, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var161, 2, 0); // TensorEdge((67, odata), (75, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var162, 2, 0); // TensorEdge((67, odata), (75, func_out0), 0), imce_3_1 -> inode_3_0
      // endgenerate: bn_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 2) { // imce_3_2
    // generate: TensorEdge(((66, -36), config), ((66, 54), config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge(((66, -36), config), ((66, 54), config)), config write
    // generate: conv exec4
    // generate: conv exec4_row_group0_outer_loop(iterate row offset)
    // generate : conv exec4_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec4_row_group0_col_group0
    // generate : conv exec4_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 18; i1++) { // generate : load_block
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((64, odata), ((66, 54), data), 0), imce_2_1 -> imce_3_2

      } // endgenerate
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var9 = __builtin_IMCE_GET_CREG((short)0);
    var10 = __builtin_IMCE_GET_CREG((short)1);
    var11 = __builtin_IMCE_GET_CREG((short)2);
    var12 = __builtin_IMCE_GET_CREG((short)3);
    // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    var13 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    var14 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    var15 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    var16 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    // generate: add

    var188 = __builtin_IMCE_ADD(var9, var13, 15);
    var189 = __builtin_IMCE_ADD(var10, var14, 15);
    var190 = __builtin_IMCE_ADD(var11, var15, 15);
    var191 = __builtin_IMCE_ADD(var12, var16, 15);
    // endgenerate: add
    __builtin_IMCE_SEND(1, var188, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(1, var189, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(1, var190, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(1, var191, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
    // endgenerate : conv exec4_row_group0_col_group0
    // endgenerate: conv exec4_row_group0_col_group0
    // generate: conv exec4_row_group0_col_group1
    for (int i1 = 0; i1 < 14; i1++) { // generate : conv exec4_row_group0_col_group1

      // generate: load_block
      // generate : load_block. loop count == 1
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((64, odata), ((66, 54), data), 0), imce_2_1 -> imce_3_2

      } // endgenerate
      // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var9 = __builtin_IMCE_GET_CREG((short)0);
      var10 = __builtin_IMCE_GET_CREG((short)1);
      var11 = __builtin_IMCE_GET_CREG((short)2);
      var12 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      var13 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      var14 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      var15 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      var16 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // generate: add

      var188 = __builtin_IMCE_ADD(var9, var13, 15);
      var189 = __builtin_IMCE_ADD(var10, var14, 15);
      var190 = __builtin_IMCE_ADD(var11, var15, 15);
      var191 = __builtin_IMCE_ADD(var12, var16, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(1, var188, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var189, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var190, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var191, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
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
    // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    var13 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    var14 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    var15 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    var16 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
    // generate: add

    var188 = __builtin_IMCE_ADD(var9, var13, 15);
    var189 = __builtin_IMCE_ADD(var10, var14, 15);
    var190 = __builtin_IMCE_ADD(var11, var15, 15);
    var191 = __builtin_IMCE_ADD(var12, var16, 15);
    // endgenerate: add
    __builtin_IMCE_SEND(1, var188, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(1, var189, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(1, var190, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(1, var191, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
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
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((64, odata), ((66, 54), data), 0), imce_2_1 -> imce_3_2

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var9 = __builtin_IMCE_GET_CREG((short)0);
      var10 = __builtin_IMCE_GET_CREG((short)1);
      var11 = __builtin_IMCE_GET_CREG((short)2);
      var12 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      var13 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      var14 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      var15 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      var16 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // generate: add

      var188 = __builtin_IMCE_ADD(var9, var13, 15);
      var189 = __builtin_IMCE_ADD(var10, var14, 15);
      var190 = __builtin_IMCE_ADD(var11, var15, 15);
      var191 = __builtin_IMCE_ADD(var12, var16, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(1, var188, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var189, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var190, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var191, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
      // endgenerate : conv exec4_row_group1_col_group0
      // endgenerate: conv exec4_row_group1_col_group0
      // generate: conv exec4_row_group1_col_group1
      for (int i2 = 0; i2 < 14; i2++) { // generate : conv exec4_row_group1_col_group1

        // generate: load_block
        // generate : load_block. loop count == 1
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((64, odata), ((66, 54), data), 0), imce_2_1 -> imce_3_2

        } // endgenerate
        // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var9 = __builtin_IMCE_GET_CREG((short)0);
        var10 = __builtin_IMCE_GET_CREG((short)1);
        var11 = __builtin_IMCE_GET_CREG((short)2);
        var12 = __builtin_IMCE_GET_CREG((short)3);
        // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
        var13 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
        // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
        // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
        var14 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
        // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
        // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
        var15 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
        // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
        // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
        var16 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
        // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
        // generate: add

        var188 = __builtin_IMCE_ADD(var9, var13, 15);
        var189 = __builtin_IMCE_ADD(var10, var14, 15);
        var190 = __builtin_IMCE_ADD(var11, var15, 15);
        var191 = __builtin_IMCE_ADD(var12, var16, 15);
        // endgenerate: add
        __builtin_IMCE_SEND(1, var188, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
        __builtin_IMCE_SEND(1, var189, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
        __builtin_IMCE_SEND(1, var190, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
        __builtin_IMCE_SEND(1, var191, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
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
      // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      var13 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      var14 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      var15 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      var16 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // generate: add

      var188 = __builtin_IMCE_ADD(var9, var13, 15);
      var189 = __builtin_IMCE_ADD(var10, var14, 15);
      var190 = __builtin_IMCE_ADD(var11, var15, 15);
      var191 = __builtin_IMCE_ADD(var12, var16, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(1, var188, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var189, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var190, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var191, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
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
      // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      var13 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      var14 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      var15 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // generate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      var16 = __builtin_IMCE_RECV(2); // TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge((65, odata), ((66, 55), rhs)), imce_2_2 -> imce_3_2
      // generate: add

      var188 = __builtin_IMCE_ADD(var9, var13, 15);
      var189 = __builtin_IMCE_ADD(var10, var14, 15);
      var190 = __builtin_IMCE_ADD(var11, var15, 15);
      var191 = __builtin_IMCE_ADD(var12, var16, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(1, var188, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var189, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var190, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var191, 2, 0); // TensorEdge(((66, 55), odata), (67, data)), imce_3_2 -> imce_3_1
    } // endgenerate : conv exec4_row_group2_col_group0
    // endgenerate: conv exec4_row_group2_col_group0
    // endgenerate : conv exec4_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec4_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec4
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 3) { // imce_3_3
  }
  else if (hid == 3 && wid == 4) { // imce_3_4
  }
}
