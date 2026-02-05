#include "../common_decl.h"
void tvmgen_default_tvmgen_default_imcflow_main_46_round_imcflow_region2_main_21() {
  int hid = __builtin_IMCE_GET_CORE_HID();
  int wid = __builtin_IMCE_GET_CORE_WID();
  short16 var1; // (ConvBlock(gid: 57), 0)
  short16 var2; // (ConvBlock(gid: 57), 1)
  short16 var3; // (ConvBlock(gid: 57), 2)
  short16 var4; // (ConvBlock(gid: 57), 3)
  short16 var5; // (ConvBlock(gid: 60), 0)
  short16 var6; // (ConvBlock(gid: 60), 1)
  short16 var7; // (ConvBlock(gid: 60), 2)
  short16 var8; // (ConvBlock(gid: 60), 3)
  short16 var9; // (ConvBlock(gid: 47), 0)
  short16 var10; // (ConvBlock(gid: 47), 1)
  short16 var11; // (ConvBlock(gid: 47), 2)
  short16 var12; // (ConvBlock(gid: 47), 3)
  short16 var13; // (TensorEdge((60, odata), ((61, 48), rhs)), 0)
  short16 var14; // (TensorEdge((60, odata), ((61, 48), rhs)), 1)
  short16 var15; // (TensorEdge((60, odata), ((61, 48), rhs)), 2)
  short16 var16; // (TensorEdge((60, odata), ((61, 48), rhs)), 3)
  short16 var17; // (ConvBlock(gid: 66), 0)
  short16 var18; // (ConvBlock(gid: 66), 1)
  short16 var19; // (ConvBlock(gid: 66), 2)
  short16 var20; // (ConvBlock(gid: 66), 3)
  short16 var21; // (TensorEdge((-46, config), (57, config)), 0)
  short16 var22; // (TensorEdge((56, odata), (57, data)), 0)
  short16 var23; // (TensorEdge((56, odata), (57, data)), 1)
  short16 var24; // (TensorEdge((56, odata), (57, data)), 2)
  short16 var25; // (TensorEdge((56, odata), (57, data)), 3)
  short16 var26; // (TensorEdge((-43, min), (56, min)), 0)
  short16 var27; // (TensorEdge((-44, max), (56, max)), 0)
  short16 var28; // (TensorEdge(((55, 53), odata), (56, data)), 0)
  short16 var29; // (MinmaxQuantBlock(gid: 56), 0)
  short16 var30; // (MinmaxQuantBlock(gid: 56), 1)
  short16 var31; // (MinmaxQuantBlock(gid: 56), 2)
  short16 var32; // (MinmaxQuantBlock(gid: 56), 3)
  short16 var33; // (TensorEdge((-54, min), (65, min)), 0)
  short16 var34; // (TensorEdge((-55, max), (65, max)), 0)
  short16 var35; // (TensorEdge(((55, 53), odata), (65, data)), 0)
  short16 var36; // (MinmaxQuantBlock(gid: 65), 0)
  short16 var37; // (MinmaxQuantBlock(gid: 65), 1)
  short16 var38; // (MinmaxQuantBlock(gid: 65), 2)
  short16 var39; // (MinmaxQuantBlock(gid: 65), 3)
  short16 var40; // (TensorEdge(((58, -37), fused_scale), ((58, 50), fused_scale)), 0)
  short16 var41; // (TensorEdge(((58, -37), fused_scale), ((58, 50), fused_scale)), 1)
  short16 var42; // (TensorEdge(((58, -38), fused_bias), ((58, 50), fused_bias)), 0)
  short16 var43; // (TensorEdge(((58, -38), fused_bias), ((58, 50), fused_bias)), 1)
  short16 var44; // (TensorEdge(((58, -39), min), ((58, 51), min)), 0)
  short16 var45; // (TensorEdge(((58, -40), max), ((58, 51), max)), 0)
  short16 var46; // (TensorEdge((57, odata), ((58, 50), data)), 0)
  short16 var47; // (TensorEdge((57, odata), ((58, 50), data)), 1)
  short16 var48; // (TensorEdge((57, odata), ((58, 50), data)), 2)
  short16 var49; // (TensorEdge((57, odata), ((58, 50), data)), 3)
  short16 var50; // (MinmaxQuantBlock(gid: 51), 0)
  short16 var51; // (MinmaxQuantBlock(gid: 51), 1)
  short16 var52; // (MinmaxQuantBlock(gid: 51), 2)
  short16 var53; // (MinmaxQuantBlock(gid: 51), 3)
  short16 var54; // (BatchNormBlock(gid: 50), 0)
  short16 var55; // (BatchNormBlock(gid: 50), 0, 'mult_result')
  short16 var56; // (BatchNormBlock(gid: 50), 1)
  short16 var57; // (BatchNormBlock(gid: 50), 1, 'mult_result')
  short16 var58; // ((BatchNormBlock(gid: 50), 0), 'L')
  short16 var59; // ((BatchNormBlock(gid: 50), 0), 'H')
  short16 var60; // ((BatchNormBlock(gid: 50), 0), 'neg1')
  short16 var61; // ((BatchNormBlock(gid: 50), 0), 'const_7fff')
  short16 var62; // ((BatchNormBlock(gid: 50), 0), 'H_sign')
  short16 var63; // ((BatchNormBlock(gid: 50), 0), 'L_sign')
  short16 var64; // ((BatchNormBlock(gid: 50), 0), 'mismatch')
  short16 var65; // ((BatchNormBlock(gid: 50), 0), 'saturate_val')
  short16 var66; // ((BatchNormBlock(gid: 50), 0), 'not_mismatch')
  short16 var67; // ((BatchNormBlock(gid: 50), 0), 'part1')
  short16 var68; // ((BatchNormBlock(gid: 50), 0), 'part2')
  short16 var69; // ((BatchNormBlock(gid: 50), 1), 'L')
  short16 var70; // ((BatchNormBlock(gid: 50), 1), 'H')
  short16 var71; // ((BatchNormBlock(gid: 50), 1), 'neg1')
  short16 var72; // ((BatchNormBlock(gid: 50), 1), 'const_7fff')
  short16 var73; // ((BatchNormBlock(gid: 50), 1), 'H_sign')
  short16 var74; // ((BatchNormBlock(gid: 50), 1), 'L_sign')
  short16 var75; // ((BatchNormBlock(gid: 50), 1), 'mismatch')
  short16 var76; // ((BatchNormBlock(gid: 50), 1), 'saturate_val')
  short16 var77; // ((BatchNormBlock(gid: 50), 1), 'not_mismatch')
  short16 var78; // ((BatchNormBlock(gid: 50), 1), 'part1')
  short16 var79; // ((BatchNormBlock(gid: 50), 1), 'part2')
  short16 var80; // (TensorEdge((-30, odata), ((55, 53), lhs)), 0)
  short16 var81; // (TensorEdge((-31, odata), ((55, 53), rhs)), 0)
  short16 var82; // (AddBlock(gid: 53), 0)
  short16 var83; // (TensorEdge((-57, config), (66, config)), 0)
  short16 var84; // (TensorEdge((65, odata), (66, data)), 0)
  short16 var85; // (TensorEdge((65, odata), (66, data)), 1)
  short16 var86; // (TensorEdge((65, odata), (66, data)), 2)
  short16 var87; // (TensorEdge((65, odata), (66, data)), 3)
  short16 var88; // (TensorEdge(((61, -35), config), ((61, 47), config)), 0)
  short16 var89; // (AddBlock(gid: 48), 0)
  short16 var90; // (AddBlock(gid: 48), 1)
  short16 var91; // (AddBlock(gid: 48), 2)
  short16 var92; // (AddBlock(gid: 48), 3)
  short16 var93; // (TensorEdge((-48, config), (60, config)), 0)
  short16 var94; // (TensorEdge((59, odata), (60, data), 1), 0)
  short16 var95; // (TensorEdge((59, odata), (60, data), 1), 1)
  short16 var96; // (TensorEdge((59, odata), (60, data), 1), 2)
  short16 var97; // (TensorEdge((59, odata), (60, data), 1), 3)
  short16 var98; // (TensorEdge((-58, fused_scale), (67, fused_scale)), 0)
  short16 var99; // (TensorEdge((-58, fused_scale), (67, fused_scale)), 1)
  short16 var100; // (TensorEdge((-59, fused_bias), (67, fused_bias)), 0)
  short16 var101; // (TensorEdge((-59, fused_bias), (67, fused_bias)), 1)
  short16 var102; // (TensorEdge((66, odata), (67, data)), 0)
  short16 var103; // (TensorEdge((66, odata), (67, data)), 1)
  short16 var104; // (BatchNormBlock(gid: 67), 0)
  short16 var105; // (BatchNormBlock(gid: 67), 1)
  short16 var106; // (BatchNormBlock(gid: 67), 0, 'mult_result')
  short16 var107; // (BatchNormBlock(gid: 67), 1, 'mult_result')
  short16 var108; // ((BatchNormBlock(gid: 67), 0), 'L')
  short16 var109; // ((BatchNormBlock(gid: 67), 0), 'H')
  short16 var110; // ((BatchNormBlock(gid: 67), 0), 'neg1')
  short16 var111; // ((BatchNormBlock(gid: 67), 0), 'const_7fff')
  short16 var112; // ((BatchNormBlock(gid: 67), 0), 'H_sign')
  short16 var113; // ((BatchNormBlock(gid: 67), 0), 'L_sign')
  short16 var114; // ((BatchNormBlock(gid: 67), 0), 'mismatch')
  short16 var115; // ((BatchNormBlock(gid: 67), 0), 'saturate_val')
  short16 var116; // ((BatchNormBlock(gid: 67), 0), 'not_mismatch')
  short16 var117; // ((BatchNormBlock(gid: 67), 0), 'part1')
  short16 var118; // ((BatchNormBlock(gid: 67), 0), 'part2')
  short16 var119; // ((BatchNormBlock(gid: 67), 1), 'L')
  short16 var120; // ((BatchNormBlock(gid: 67), 1), 'H')
  short16 var121; // ((BatchNormBlock(gid: 67), 1), 'neg1')
  short16 var122; // ((BatchNormBlock(gid: 67), 1), 'const_7fff')
  short16 var123; // ((BatchNormBlock(gid: 67), 1), 'H_sign')
  short16 var124; // ((BatchNormBlock(gid: 67), 1), 'L_sign')
  short16 var125; // ((BatchNormBlock(gid: 67), 1), 'mismatch')
  short16 var126; // ((BatchNormBlock(gid: 67), 1), 'saturate_val')
  short16 var127; // ((BatchNormBlock(gid: 67), 1), 'not_mismatch')
  short16 var128; // ((BatchNormBlock(gid: 67), 1), 'part1')
  short16 var129; // ((BatchNormBlock(gid: 67), 1), 'part2')
  short16 var130; // (TensorEdge((-49, fused_scale), (62, fused_scale)), 0)
  short16 var131; // (TensorEdge((-49, fused_scale), (62, fused_scale)), 1)
  short16 var132; // (TensorEdge((-50, fused_bias), (62, fused_bias)), 0)
  short16 var133; // (TensorEdge((-50, fused_bias), (62, fused_bias)), 1)
  short16 var134; // (TensorEdge(((61, 48), odata), (62, data)), 0)
  short16 var135; // (TensorEdge(((61, 48), odata), (62, data)), 1)
  short16 var136; // (BatchNormBlock(gid: 62), 0)
  short16 var137; // (BatchNormBlock(gid: 62), 1)
  short16 var138; // (BatchNormBlock(gid: 62), 0, 'mult_result')
  short16 var139; // (BatchNormBlock(gid: 62), 1, 'mult_result')
  short16 var140; // ((BatchNormBlock(gid: 62), 0), 'L')
  short16 var141; // ((BatchNormBlock(gid: 62), 0), 'H')
  short16 var142; // ((BatchNormBlock(gid: 62), 0), 'neg1')
  short16 var143; // ((BatchNormBlock(gid: 62), 0), 'const_7fff')
  short16 var144; // ((BatchNormBlock(gid: 62), 0), 'H_sign')
  short16 var145; // ((BatchNormBlock(gid: 62), 0), 'L_sign')
  short16 var146; // ((BatchNormBlock(gid: 62), 0), 'mismatch')
  short16 var147; // ((BatchNormBlock(gid: 62), 0), 'saturate_val')
  short16 var148; // ((BatchNormBlock(gid: 62), 0), 'not_mismatch')
  short16 var149; // ((BatchNormBlock(gid: 62), 0), 'part1')
  short16 var150; // ((BatchNormBlock(gid: 62), 0), 'part2')
  short16 var151; // ((BatchNormBlock(gid: 62), 1), 'L')
  short16 var152; // ((BatchNormBlock(gid: 62), 1), 'H')
  short16 var153; // ((BatchNormBlock(gid: 62), 1), 'neg1')
  short16 var154; // ((BatchNormBlock(gid: 62), 1), 'const_7fff')
  short16 var155; // ((BatchNormBlock(gid: 62), 1), 'H_sign')
  short16 var156; // ((BatchNormBlock(gid: 62), 1), 'L_sign')
  short16 var157; // ((BatchNormBlock(gid: 62), 1), 'mismatch')
  short16 var158; // ((BatchNormBlock(gid: 62), 1), 'saturate_val')
  short16 var159; // ((BatchNormBlock(gid: 62), 1), 'not_mismatch')
  short16 var160; // ((BatchNormBlock(gid: 62), 1), 'part1')
  short16 var161; // ((BatchNormBlock(gid: 62), 1), 'part2')
  short16 var162; // (TensorEdge((-60, scale), ((69, 63), rhs)), 0)
  short16 var163; // (TensorEdge((-60, scale), ((69, 63), rhs)), 1)
  short16 var164; // (TensorEdge((68, odata), ((69, 63), lhs)), 0)
  short16 var165; // (AddBlock(gid: 63), 0)
  short16 var166; // (TensorEdge((-53, scale), (68, rhs)), 0)
  short16 var167; // (TensorEdge((-53, scale), (68, rhs)), 1)
  short16 var168; // (TensorEdge((67, odata), (68, lhs)), 0)
  short16 var169; // (MultlBlock(gid: 68), 0)
  short16 var170; // ((MultlBlock(gid: 68), 0), 'L')
  short16 var171; // ((MultlBlock(gid: 68), 0), 'H')
  short16 var172; // ((MultlBlock(gid: 68), 0), 'neg1')
  short16 var173; // ((MultlBlock(gid: 68), 0), 'const_7fff')
  short16 var174; // ((MultlBlock(gid: 68), 0), 'H_sign')
  short16 var175; // ((MultlBlock(gid: 68), 0), 'L_sign')
  short16 var176; // ((MultlBlock(gid: 68), 0), 'mismatch')
  short16 var177; // ((MultlBlock(gid: 68), 0), 'saturate_val')
  short16 var178; // ((MultlBlock(gid: 68), 0), 'not_mismatch')
  short16 var179; // ((MultlBlock(gid: 68), 0), 'part1')
  short16 var180; // ((MultlBlock(gid: 68), 0), 'part2')
  if (hid == 0 && wid == 1) { // imce_0_1
    // generate: TensorEdge((-46, config), (57, config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge((-46, config), (57, config)), config write
    // generate: conv exec2
    // generate: conv exec2_row_group0_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 16; i1++) { // generate : conv exec2_row_group0_outer_loop(iterate row offset)
      // generate: conv exec2_row_group0_col_group0
      // generate : conv exec2_row_group0_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 34; i2++) { // generate : load_block
        __builtin_IMCE_SETFLAG(1); 
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((56, odata), (57, data)), imce_0_2 -> imce_0_1

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
      __builtin_IMCE_SEND(1, var1, 2, 0); // TensorEdge((57, odata), ((58, 50), data)), imce_0_1 -> imce_1_1
      __builtin_IMCE_SEND(1, var2, 2, 0); // TensorEdge((57, odata), ((58, 50), data)), imce_0_1 -> imce_1_1
      __builtin_IMCE_SEND(1, var3, 2, 0); // TensorEdge((57, odata), ((58, 50), data)), imce_0_1 -> imce_1_1
      __builtin_IMCE_SEND(1, var4, 2, 0); // TensorEdge((57, odata), ((58, 50), data)), imce_0_1 -> imce_1_1
      // endgenerate : conv exec2_row_group0_col_group0
      // endgenerate: conv exec2_row_group0_col_group0
      // generate: conv exec2_row_group0_col_group1
      for (int i2 = 0; i2 < 15; i2++) { // generate : conv exec2_row_group0_col_group1

        // generate: load_block
        for (int i3 = 0; i3 < 2; i3++) { // generate : load_block
          __builtin_IMCE_SETFLAG(1); 
          for (int i4 = 0; i4 < 4; i4++) { // generate
            __builtin_IMCE_LOAD_LB(0); // TensorEdge((56, odata), (57, data)), imce_0_2 -> imce_0_1

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
        __builtin_IMCE_SEND(1, var1, 2, 0); // TensorEdge((57, odata), ((58, 50), data)), imce_0_1 -> imce_1_1
        __builtin_IMCE_SEND(1, var2, 2, 0); // TensorEdge((57, odata), ((58, 50), data)), imce_0_1 -> imce_1_1
        __builtin_IMCE_SEND(1, var3, 2, 0); // TensorEdge((57, odata), ((58, 50), data)), imce_0_1 -> imce_1_1
        __builtin_IMCE_SEND(1, var4, 2, 0); // TensorEdge((57, odata), ((58, 50), data)), imce_0_1 -> imce_1_1
      } // endgenerate : conv exec2_row_group0_col_group1
      // endgenerate: conv exec2_row_group0_col_group1
    } // endgenerate : conv exec2_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec2_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec2
    __builtin_IMCE_STOP();
  }
  else if (hid == 0 && wid == 2) { // imce_0_2
    // generate: TensorEdge((-43, min), (56, min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge((-43, min), (56, min)), min write
    // generate: TensorEdge((-44, max), (56, max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge((-44, max), (56, max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 1024; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      __builtin_IMCE_SETFLAG(2);
      __builtin_IMCE_STANDBY(7, 2); 
      __builtin_IMCE_SETFLAG(0); 
      var28 = __builtin_IMCE_RECV(2); // TensorEdge(((55, 53), odata), (56, data)), imce_1_2 -> imce_0_2
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var28, 0, 15, 0);
      var29 = __builtin_IMCE_GET_QREG(0);
      var30 = __builtin_IMCE_GET_QREG(1);
      var31 = __builtin_IMCE_GET_QREG(2);
      var32 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      __builtin_IMCE_STANDBY(1, 1); 
      __builtin_IMCE_SEND(1, var29, 0, 0); // TensorEdge((56, odata), (57, data)), imce_0_2 -> imce_0_1
      __builtin_IMCE_SEND(1, var30, 0, 0); // TensorEdge((56, odata), (57, data)), imce_0_2 -> imce_0_1
      __builtin_IMCE_SEND(1, var31, 0, 0); // TensorEdge((56, odata), (57, data)), imce_0_2 -> imce_0_1
      __builtin_IMCE_SEND(1, var32, 0, 0); // TensorEdge((56, odata), (57, data)), imce_0_2 -> imce_0_1
      // endgenerate: min_max_quantize_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 0 && wid == 3) { // imce_0_3
    // generate: TensorEdge((-54, min), (65, min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge((-54, min), (65, min)), min write
    // generate: TensorEdge((-55, max), (65, max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge((-55, max), (65, max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 1024; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      __builtin_IMCE_SETFLAG(2);
      __builtin_IMCE_STANDBY(7, 2); 
      __builtin_IMCE_SETFLAG(0); 
      var35 = __builtin_IMCE_RECV(2); // TensorEdge(((55, 53), odata), (65, data)), imce_1_2 -> imce_0_3
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var35, 0, 15, 0);
      var36 = __builtin_IMCE_GET_QREG(0);
      var37 = __builtin_IMCE_GET_QREG(1);
      var38 = __builtin_IMCE_GET_QREG(2);
      var39 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize

      __builtin_IMCE_STANDBY(8, 1); 
      __builtin_IMCE_SEND(2, var36, 0, 0); // TensorEdge((65, odata), (66, data)), imce_0_3 -> imce_1_3
      __builtin_IMCE_SEND(2, var37, 0, 0); // TensorEdge((65, odata), (66, data)), imce_0_3 -> imce_1_3
      __builtin_IMCE_SEND(2, var38, 0, 0); // TensorEdge((65, odata), (66, data)), imce_0_3 -> imce_1_3
      __builtin_IMCE_SEND(2, var39, 0, 0); // TensorEdge((65, odata), (66, data)), imce_0_3 -> imce_1_3
      // endgenerate: min_max_quantize_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 0 && wid == 4) { // imce_0_4
  }
  else if (hid == 1 && wid == 1) { // imce_1_1
    // generate: TensorEdge(((58, -37), fused_scale), ((58, 50), fused_scale)), fused_scale write

    var40 = __builtin_IMCE_RECV(1);
    var41 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((58, -37), fused_scale), ((58, 50), fused_scale)), fused_scale write
    // generate: TensorEdge(((58, -38), fused_bias), ((58, 50), fused_bias)), fused_bias write

    var42 = __builtin_IMCE_RECV(1);
    var43 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((58, -38), fused_bias), ((58, 50), fused_bias)), fused_bias write
    // generate: TensorEdge(((58, -39), min), ((58, 51), min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge(((58, -39), min), ((58, 51), min)), min write
    // generate: TensorEdge(((58, -40), max), ((58, 51), max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge(((58, -40), max), ((58, 51), max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 256; i1++) { // generate : call_created_loop
      // generate: imcflow.preop-minmax_wrapper

      __builtin_IMCE_SETFLAG(1);
      var46 = __builtin_IMCE_RECV(2); // TensorEdge((57, odata), ((58, 50), data)), imce_0_1 -> imce_1_1
      var47 = __builtin_IMCE_RECV(2); // TensorEdge((57, odata), ((58, 50), data)), imce_0_1 -> imce_1_1
      var48 = __builtin_IMCE_RECV(2); // TensorEdge((57, odata), ((58, 50), data)), imce_0_1 -> imce_1_1
      var49 = __builtin_IMCE_RECV(2); // TensorEdge((57, odata), ((58, 50), data)), imce_0_1 -> imce_1_1
      __builtin_IMCE_SETFLAG(0);
      // generate: imcflow.preop-minmax_block
      // generate: batch_norm


      var58 = __builtin_IMCE_MULTL(var46, var40, 15);
      var59 = __builtin_IMCE_MULTH(var46, var40, 15);
      var60 = __builtin_IMCE_SUBI(0, 1);
      var61 = __builtin_IMCE_SRLI(var60, 1);
      var62 = __builtin_IMCE_SRAI(var59, 15);
      var63 = __builtin_IMCE_SRAI(var58, 15);
      var64 = __builtin_IMCE_XOR(var62, var63, 15);
      var65 = __builtin_IMCE_XOR(var62, var61, 15);
      var66 = __builtin_IMCE_XOR(var64, var60, 15);
      var67 = __builtin_IMCE_AND(var64, var65, 15);
      var68 = __builtin_IMCE_AND(var66, var58, 15);
      var55 = __builtin_IMCE_OR(var67, var68, 15);
      var54 = __builtin_IMCE_ADD(var55, var42, 15);

      var69 = __builtin_IMCE_MULTL(var47, var41, 15);
      var70 = __builtin_IMCE_MULTH(var47, var41, 15);
      var71 = __builtin_IMCE_SUBI(0, 1);
      var72 = __builtin_IMCE_SRLI(var71, 1);
      var73 = __builtin_IMCE_SRAI(var70, 15);
      var74 = __builtin_IMCE_SRAI(var69, 15);
      var75 = __builtin_IMCE_XOR(var73, var74, 15);
      var76 = __builtin_IMCE_XOR(var73, var72, 15);
      var77 = __builtin_IMCE_XOR(var75, var71, 15);
      var78 = __builtin_IMCE_AND(var75, var76, 15);
      var79 = __builtin_IMCE_AND(var77, var69, 15);
      var57 = __builtin_IMCE_OR(var78, var79, 15);
      var56 = __builtin_IMCE_ADD(var57, var43, 15);
      // endgenerate: batch_norm
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var54, 0, 15, 0);
      __builtin_IMCE_MM_QUANT(var56, 0, 15, 1);
      var50 = __builtin_IMCE_GET_QREG(0);
      var51 = __builtin_IMCE_GET_QREG(1);
      var52 = __builtin_IMCE_GET_QREG(2);
      var53 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      // endgenerate: imcflow.preop-minmax_block
      __builtin_IMCE_STANDBY(11, 1);
      __builtin_IMCE_SEND(1, var50, 0, 0); // TensorEdge(((58, 51), odata), (59, data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(1, var51, 0, 0); // TensorEdge(((58, 51), odata), (59, data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(1, var52, 0, 0); // TensorEdge(((58, 51), odata), (59, data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(1, var53, 0, 0); // TensorEdge(((58, 51), odata), (59, data)), imce_1_1 -> imce_2_1
      // endgenerate: imcflow.preop-minmax_wrapper
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 1 && wid == 2) { // imce_1_2
    // generate: call_created_loop
    for (int i1 = 0; i1 < 1024; i1++) { // generate : call_created_loop
      // generate: imcflow.vecops_wrapper

      __builtin_IMCE_SETFLAG(1);
      __builtin_IMCE_STANDBY(0, 1); 
      __builtin_IMCE_STANDBY(5, 1); 
      __builtin_IMCE_SETFLAG(0); 
      var80 = __builtin_IMCE_RECV(2); // TensorEdge((-30, odata), ((55, 53), lhs)), inode_0_0 -> imce_1_2
      var81 = __builtin_IMCE_RECV(3); // TensorEdge((-31, odata), ((55, 53), rhs)), inode_1_0 -> imce_1_2
      // generate: imcflow.vecops_block
      // generate: add

      var82 = __builtin_IMCE_ADD(var80, var81, 15);
      // endgenerate: add
      // endgenerate: imcflow.vecops_block

      __builtin_IMCE_STANDBY(2, 2);
      __builtin_IMCE_STANDBY(3, 2);
      __builtin_IMCE_SETFLAG(2);
      __builtin_IMCE_STANDBY(2, 0);
      __builtin_IMCE_STANDBY(3, 0);
      __builtin_IMCE_SETFLAG(0);
      __builtin_IMCE_SEND(1, var82, 2, 0); // TensorEdge(((55, 53), odata), (56, data)),TensorEdge(((55, 53), odata), (65, data)), imce_1_2 -> imce_0_2
      // endgenerate: imcflow.vecops_wrapper
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 1 && wid == 3) { // imce_1_3
    // generate: TensorEdge((-57, config), (66, config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge((-57, config), (66, config)), config write
    // generate: conv exec5
    // generate: conv exec5_row_group0_outer_loop(iterate row offset)
    // generate : conv exec5_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec5_row_group0_col_group0
    // generate : conv exec5_row_group0_col_group0. loop count == 1

    // generate: load_block
    // generate : load_block. loop count == 1
    __builtin_IMCE_SETFLAG(1);
    for (int i1 = 0; i1 < 4; i1++) { // generate
      __builtin_IMCE_LOAD_LB(0); // TensorEdge((65, odata), (66, data)), imce_0_3 -> imce_1_3

    } // endgenerate
    __builtin_IMCE_SETFLAG(0);
    // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var17 = __builtin_IMCE_GET_CREG((short)0);
    var18 = __builtin_IMCE_GET_CREG((short)1);
    var19 = __builtin_IMCE_GET_CREG((short)2);
    var20 = __builtin_IMCE_GET_CREG((short)3);
    __builtin_IMCE_SEND(2, var17, 2, 0); // TensorEdge((66, odata), (67, data)), imce_1_3 -> imce_2_3
    __builtin_IMCE_SEND(2, var18, 2, 0); // TensorEdge((66, odata), (67, data)), imce_1_3 -> imce_2_3
    __builtin_IMCE_SEND(2, var19, 2, 0); // TensorEdge((66, odata), (67, data)), imce_1_3 -> imce_2_3
    __builtin_IMCE_SEND(2, var20, 2, 0); // TensorEdge((66, odata), (67, data)), imce_1_3 -> imce_2_3
    // endgenerate : conv exec5_row_group0_col_group0
    // endgenerate: conv exec5_row_group0_col_group0
    // generate: conv exec5_row_group0_col_group1
    for (int i1 = 0; i1 < 15; i1++) { // generate : conv exec5_row_group0_col_group1

      // generate: load_block
      for (int i2 = 0; i2 < 2; i2++) { // generate : load_block
        __builtin_IMCE_SETFLAG(1);
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((65, odata), (66, data)), imce_0_3 -> imce_1_3

        } // endgenerate
        __builtin_IMCE_SETFLAG(0);
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var17 = __builtin_IMCE_GET_CREG((short)0);
      var18 = __builtin_IMCE_GET_CREG((short)1);
      var19 = __builtin_IMCE_GET_CREG((short)2);
      var20 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(2, var17, 2, 0); // TensorEdge((66, odata), (67, data)), imce_1_3 -> imce_2_3
      __builtin_IMCE_SEND(2, var18, 2, 0); // TensorEdge((66, odata), (67, data)), imce_1_3 -> imce_2_3
      __builtin_IMCE_SEND(2, var19, 2, 0); // TensorEdge((66, odata), (67, data)), imce_1_3 -> imce_2_3
      __builtin_IMCE_SEND(2, var20, 2, 0); // TensorEdge((66, odata), (67, data)), imce_1_3 -> imce_2_3
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
        __builtin_IMCE_SETFLAG(1);
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((65, odata), (66, data)), imce_0_3 -> imce_1_3

        } // endgenerate
        __builtin_IMCE_SETFLAG(0);
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var17 = __builtin_IMCE_GET_CREG((short)0);
      var18 = __builtin_IMCE_GET_CREG((short)1);
      var19 = __builtin_IMCE_GET_CREG((short)2);
      var20 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(2, var17, 2, 0); // TensorEdge((66, odata), (67, data)), imce_1_3 -> imce_2_3
      __builtin_IMCE_SEND(2, var18, 2, 0); // TensorEdge((66, odata), (67, data)), imce_1_3 -> imce_2_3
      __builtin_IMCE_SEND(2, var19, 2, 0); // TensorEdge((66, odata), (67, data)), imce_1_3 -> imce_2_3
      __builtin_IMCE_SEND(2, var20, 2, 0); // TensorEdge((66, odata), (67, data)), imce_1_3 -> imce_2_3
      // endgenerate : conv exec5_row_group1_col_group0
      // endgenerate: conv exec5_row_group1_col_group0
      // generate: conv exec5_row_group1_col_group1
      for (int i2 = 0; i2 < 15; i2++) { // generate : conv exec5_row_group1_col_group1

        // generate: load_block
        for (int i3 = 0; i3 < 2; i3++) { // generate : load_block
          __builtin_IMCE_SETFLAG(1);
          for (int i4 = 0; i4 < 4; i4++) { // generate
            __builtin_IMCE_LOAD_LB(0); // TensorEdge((65, odata), (66, data)), imce_0_3 -> imce_1_3

          } // endgenerate
          __builtin_IMCE_SETFLAG(0);
        } // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var17 = __builtin_IMCE_GET_CREG((short)0);
        var18 = __builtin_IMCE_GET_CREG((short)1);
        var19 = __builtin_IMCE_GET_CREG((short)2);
        var20 = __builtin_IMCE_GET_CREG((short)3);
        __builtin_IMCE_SEND(2, var17, 2, 0); // TensorEdge((66, odata), (67, data)), imce_1_3 -> imce_2_3
        __builtin_IMCE_SEND(2, var18, 2, 0); // TensorEdge((66, odata), (67, data)), imce_1_3 -> imce_2_3
        __builtin_IMCE_SEND(2, var19, 2, 0); // TensorEdge((66, odata), (67, data)), imce_1_3 -> imce_2_3
        __builtin_IMCE_SEND(2, var20, 2, 0); // TensorEdge((66, odata), (67, data)), imce_1_3 -> imce_2_3
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
  else if (hid == 1 && wid == 4) { // imce_1_4
  }
  else if (hid == 2 && wid == 1) { // imce_2_1
    // generate: TensorEdge(((61, -35), config), ((61, 47), config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge(((61, -35), config), ((61, 47), config)), config write
    // generate: conv exec4
    // generate: conv exec4_row_group0_outer_loop(iterate row offset)
    // generate : conv exec4_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec4_row_group0_col_group0
    // generate : conv exec4_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 18; i1++) { // generate : load_block
      __builtin_IMCE_SETFLAG(1);
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((59, odata), ((61, 47), data), 0), imce_1_1 -> imce_2_1

      } // endgenerate
      __builtin_IMCE_SETFLAG(0);
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var9 = __builtin_IMCE_GET_CREG((short)0);
    var10 = __builtin_IMCE_GET_CREG((short)1);
    var11 = __builtin_IMCE_GET_CREG((short)2);
    var12 = __builtin_IMCE_GET_CREG((short)3);
    // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    var13 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    var14 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    var15 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    var16 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    // generate: add

    var89 = __builtin_IMCE_ADD(var9, var13, 15);
    var90 = __builtin_IMCE_ADD(var10, var14, 15);
    var91 = __builtin_IMCE_ADD(var11, var15, 15);
    var92 = __builtin_IMCE_ADD(var12, var16, 15);
    // endgenerate: add
    __builtin_IMCE_STANDBY(16, 1);
    __builtin_IMCE_SEND(2, var89, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
    __builtin_IMCE_SEND(2, var90, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
    __builtin_IMCE_SEND(2, var91, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
    __builtin_IMCE_SEND(2, var92, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
    // endgenerate : conv exec4_row_group0_col_group0
    // endgenerate: conv exec4_row_group0_col_group0
    // generate: conv exec4_row_group0_col_group1
    for (int i1 = 0; i1 < 14; i1++) { // generate : conv exec4_row_group0_col_group1

      // generate: load_block
      // generate : load_block. loop count == 1
      __builtin_IMCE_SETFLAG(1);
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((59, odata), ((61, 47), data), 0), imce_1_1 -> imce_2_1

      } // endgenerate
      __builtin_IMCE_SETFLAG(0);
      // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var9 = __builtin_IMCE_GET_CREG((short)0);
      var10 = __builtin_IMCE_GET_CREG((short)1);
      var11 = __builtin_IMCE_GET_CREG((short)2);
      var12 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      var13 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      var14 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      var15 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      var16 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // generate: add

      var89 = __builtin_IMCE_ADD(var9, var13, 15);
      var90 = __builtin_IMCE_ADD(var10, var14, 15);
      var91 = __builtin_IMCE_ADD(var11, var15, 15);
      var92 = __builtin_IMCE_ADD(var12, var16, 15);
      // endgenerate: add
      __builtin_IMCE_STANDBY(16, 1);
      __builtin_IMCE_SEND(2, var89, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(2, var90, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(2, var91, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(2, var92, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
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
    // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    var13 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    var14 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    var15 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    var16 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    // generate: add

    var89 = __builtin_IMCE_ADD(var9, var13, 15);
    var90 = __builtin_IMCE_ADD(var10, var14, 15);
    var91 = __builtin_IMCE_ADD(var11, var15, 15);
    var92 = __builtin_IMCE_ADD(var12, var16, 15);
    // endgenerate: add
    __builtin_IMCE_STANDBY(16, 1);
    __builtin_IMCE_SEND(2, var89, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
    __builtin_IMCE_SEND(2, var90, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
    __builtin_IMCE_SEND(2, var91, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
    __builtin_IMCE_SEND(2, var92, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
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
        __builtin_IMCE_SETFLAG(1);
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((59, odata), ((61, 47), data), 0), imce_1_1 -> imce_2_1

        } // endgenerate
        __builtin_IMCE_SETFLAG(0);
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var9 = __builtin_IMCE_GET_CREG((short)0);
      var10 = __builtin_IMCE_GET_CREG((short)1);
      var11 = __builtin_IMCE_GET_CREG((short)2);
      var12 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      var13 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      var14 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      var15 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      var16 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // generate: add

      var89 = __builtin_IMCE_ADD(var9, var13, 15);
      var90 = __builtin_IMCE_ADD(var10, var14, 15);
      var91 = __builtin_IMCE_ADD(var11, var15, 15);
      var92 = __builtin_IMCE_ADD(var12, var16, 15);
      // endgenerate: add
      __builtin_IMCE_STANDBY(16, 1);
      __builtin_IMCE_SEND(2, var89, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(2, var90, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(2, var91, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(2, var92, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
      // endgenerate : conv exec4_row_group1_col_group0
      // endgenerate: conv exec4_row_group1_col_group0
      // generate: conv exec4_row_group1_col_group1
      for (int i2 = 0; i2 < 14; i2++) { // generate : conv exec4_row_group1_col_group1

        // generate: load_block
        // generate : load_block. loop count == 1
        __builtin_IMCE_SETFLAG(1);
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((59, odata), ((61, 47), data), 0), imce_1_1 -> imce_2_1

        } // endgenerate
        __builtin_IMCE_SETFLAG(0);
        // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var9 = __builtin_IMCE_GET_CREG((short)0);
        var10 = __builtin_IMCE_GET_CREG((short)1);
        var11 = __builtin_IMCE_GET_CREG((short)2);
        var12 = __builtin_IMCE_GET_CREG((short)3);
        // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
        var13 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
        // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
        // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
        var14 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
        // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
        // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
        var15 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
        // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
        // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
        var16 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
        // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
        // generate: add

        var89 = __builtin_IMCE_ADD(var9, var13, 15);
        var90 = __builtin_IMCE_ADD(var10, var14, 15);
        var91 = __builtin_IMCE_ADD(var11, var15, 15);
        var92 = __builtin_IMCE_ADD(var12, var16, 15);
        // endgenerate: add
        __builtin_IMCE_STANDBY(16, 1);
        __builtin_IMCE_SEND(2, var89, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
        __builtin_IMCE_SEND(2, var90, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
        __builtin_IMCE_SEND(2, var91, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
        __builtin_IMCE_SEND(2, var92, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
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
      // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      var13 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      var14 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      var15 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      var16 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // generate: add

      var89 = __builtin_IMCE_ADD(var9, var13, 15);
      var90 = __builtin_IMCE_ADD(var10, var14, 15);
      var91 = __builtin_IMCE_ADD(var11, var15, 15);
      var92 = __builtin_IMCE_ADD(var12, var16, 15);
      // endgenerate: add
      __builtin_IMCE_STANDBY(16, 1);
      __builtin_IMCE_SEND(2, var89, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(2, var90, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(2, var91, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(2, var92, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
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
      // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      var13 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      var14 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      var15 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // generate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      var16 = __builtin_IMCE_RECV(2); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // endgenerate: TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // generate: add

      var89 = __builtin_IMCE_ADD(var9, var13, 15);
      var90 = __builtin_IMCE_ADD(var10, var14, 15);
      var91 = __builtin_IMCE_ADD(var11, var15, 15);
      var92 = __builtin_IMCE_ADD(var12, var16, 15);
      // endgenerate: add
      __builtin_IMCE_STANDBY(16, 1);
      __builtin_IMCE_SEND(2, var89, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(2, var90, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(2, var91, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SEND(2, var92, 2, 0); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
    } // endgenerate : conv exec4_row_group2_col_group0
    // endgenerate: conv exec4_row_group2_col_group0
    // endgenerate : conv exec4_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec4_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec4
    __builtin_IMCE_STOP();
  }
  else if (hid == 2 && wid == 2) { // imce_2_2
    // generate: TensorEdge((-48, config), (60, config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge((-48, config), (60, config)), config write
    // generate: conv exec3
    // generate: conv exec3_row_group0_outer_loop(iterate row offset)
    // generate : conv exec3_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec3_row_group0_col_group0
    // generate : conv exec3_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 18; i1++) { // generate : load_block
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((59, odata), (60, data), 1), imce_1_1 -> imce_2_1, imce_2_2

      } // endgenerate
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var5 = __builtin_IMCE_GET_CREG((short)0);
    var6 = __builtin_IMCE_GET_CREG((short)1);
    var7 = __builtin_IMCE_GET_CREG((short)2);
    var8 = __builtin_IMCE_GET_CREG((short)3);
    __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    // endgenerate : conv exec3_row_group0_col_group0
    // endgenerate: conv exec3_row_group0_col_group0
    // generate: conv exec3_row_group0_col_group1
    for (int i1 = 0; i1 < 14; i1++) { // generate : conv exec3_row_group0_col_group1

      // generate: load_block
      // generate : load_block. loop count == 1
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((59, odata), (60, data), 1), imce_1_1 -> imce_2_1, imce_2_2

      } // endgenerate
      // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
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
    __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
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
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((59, odata), (60, data), 1), imce_1_1 -> imce_2_1, imce_2_2

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      // endgenerate : conv exec3_row_group1_col_group0
      // endgenerate: conv exec3_row_group1_col_group0
      // generate: conv exec3_row_group1_col_group1
      for (int i2 = 0; i2 < 14; i2++) { // generate : conv exec3_row_group1_col_group1

        // generate: load_block
        // generate : load_block. loop count == 1
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((59, odata), (60, data), 1), imce_1_1 -> imce_2_1, imce_2_2

        } // endgenerate
        // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var5 = __builtin_IMCE_GET_CREG((short)0);
        var6 = __builtin_IMCE_GET_CREG((short)1);
        var7 = __builtin_IMCE_GET_CREG((short)2);
        var8 = __builtin_IMCE_GET_CREG((short)3);
        __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
        __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
        __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
        __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
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
      __builtin_IMCE_SEND(2, var5, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      __builtin_IMCE_SEND(2, var6, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      __builtin_IMCE_SEND(2, var7, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      __builtin_IMCE_SEND(2, var8, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
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
      __builtin_IMCE_SEND(2, 0, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      __builtin_IMCE_SEND(2, 0, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      __builtin_IMCE_SEND(2, 0, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
      __builtin_IMCE_SEND(2, 0, 2, 0); // TensorEdge((60, odata), ((61, 48), rhs)), imce_2_2 -> imce_2_1
    } // endgenerate : conv exec3_row_group2_col_group0
    // endgenerate: conv exec3_row_group2_col_group0
    // endgenerate : conv exec3_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec3_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec3
    __builtin_IMCE_STOP();
  }
  else if (hid == 2 && wid == 3) { // imce_2_3
    // generate: TensorEdge((-58, fused_scale), (67, fused_scale)), fused_scale write

    var98 = __builtin_IMCE_RECV(1);
    var99 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge((-58, fused_scale), (67, fused_scale)), fused_scale write
    // generate: TensorEdge((-59, fused_bias), (67, fused_bias)), fused_bias write

    var100 = __builtin_IMCE_RECV(1);
    var101 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge((-59, fused_bias), (67, fused_bias)), fused_bias write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 256; i1++) { // generate : call_created_loop
      // generate: bn_standalone

      short16 var202, var203;
      __builtin_IMCE_SETFLAG(1);
      var102 = __builtin_IMCE_RECV(2); // TensorEdge((66, odata), (67, data)), imce_1_3 -> imce_2_3
      var103 = __builtin_IMCE_RECV(2); // TensorEdge((66, odata), (67, data)), imce_1_3 -> imce_2_3
      var202 = __builtin_IMCE_RECV(2); // TensorEdge((66, odata), (67, data)), imce_1_3 -> imce_2_3
      var203 = __builtin_IMCE_RECV(2); // TensorEdge((66, odata), (67, data)), imce_1_3 -> imce_2_3
      __builtin_IMCE_SETFLAG(0);
      // generate: batch_norm


      var108 = __builtin_IMCE_MULTL(var102, var98, 15);
      var109 = __builtin_IMCE_MULTH(var102, var98, 15);
      var110 = __builtin_IMCE_SUBI(0, 1);
      var111 = __builtin_IMCE_SRLI(var110, 1);
      var112 = __builtin_IMCE_SRAI(var109, 15);
      var113 = __builtin_IMCE_SRAI(var108, 15);
      var114 = __builtin_IMCE_XOR(var112, var113, 15);
      var115 = __builtin_IMCE_XOR(var112, var111, 15);
      var116 = __builtin_IMCE_XOR(var114, var110, 15);
      var117 = __builtin_IMCE_AND(var114, var115, 15);
      var118 = __builtin_IMCE_AND(var116, var108, 15);
      var106 = __builtin_IMCE_OR(var117, var118, 15);
      var104 = __builtin_IMCE_ADD(var106, var100, 15);

      var119 = __builtin_IMCE_MULTL(var103, var99, 15);
      var120 = __builtin_IMCE_MULTH(var103, var99, 15);
      var121 = __builtin_IMCE_SUBI(0, 1);
      var122 = __builtin_IMCE_SRLI(var121, 1);
      var123 = __builtin_IMCE_SRAI(var120, 15);
      var124 = __builtin_IMCE_SRAI(var119, 15);
      var125 = __builtin_IMCE_XOR(var123, var124, 15);
      var126 = __builtin_IMCE_XOR(var123, var122, 15);
      var127 = __builtin_IMCE_XOR(var125, var121, 15);
      var128 = __builtin_IMCE_AND(var125, var126, 15);
      var129 = __builtin_IMCE_AND(var127, var119, 15);
      var107 = __builtin_IMCE_OR(var128, var129, 15);
      var105 = __builtin_IMCE_ADD(var107, var101, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var104, 2, 0); // TensorEdge((67, odata), (68, lhs)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SEND(1, var105, 2, 0); // TensorEdge((67, odata), (68, lhs)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SEND(1, 0, 2, 0); // TensorEdge((67, odata), (68, lhs)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SEND(1, 0, 2, 0); // TensorEdge((67, odata), (68, lhs)), imce_2_3 -> imce_3_3
      // endgenerate: bn_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 2 && wid == 4) { // imce_2_4
  }
  else if (hid == 3 && wid == 1) { // imce_3_1
    // generate: TensorEdge((-49, fused_scale), (62, fused_scale)), fused_scale write

    var130 = __builtin_IMCE_RECV(1);
    var131 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge((-49, fused_scale), (62, fused_scale)), fused_scale write
    // generate: TensorEdge((-50, fused_bias), (62, fused_bias)), fused_bias write

    var132 = __builtin_IMCE_RECV(1);
    var133 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge((-50, fused_bias), (62, fused_bias)), fused_bias write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 256; i1++) { // generate : call_created_loop
      // generate: bn_standalone

      short16 var234, var235;
      __builtin_IMCE_SETFLAG(1);
      var134 = __builtin_IMCE_RECV(2); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
      var135 = __builtin_IMCE_RECV(2); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
      var234 = __builtin_IMCE_RECV(2); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
      var235 = __builtin_IMCE_RECV(2); // TensorEdge(((61, 48), odata), (62, data)), imce_2_1 -> imce_3_1
      __builtin_IMCE_SETFLAG(0);
      // generate: batch_norm


      var140 = __builtin_IMCE_MULTL(var134, var130, 15);
      var141 = __builtin_IMCE_MULTH(var134, var130, 15);
      var142 = __builtin_IMCE_SUBI(0, 1);
      var143 = __builtin_IMCE_SRLI(var142, 1);
      var144 = __builtin_IMCE_SRAI(var141, 15);
      var145 = __builtin_IMCE_SRAI(var140, 15);
      var146 = __builtin_IMCE_XOR(var144, var145, 15);
      var147 = __builtin_IMCE_XOR(var144, var143, 15);
      var148 = __builtin_IMCE_XOR(var146, var142, 15);
      var149 = __builtin_IMCE_AND(var146, var147, 15);
      var150 = __builtin_IMCE_AND(var148, var140, 15);
      var138 = __builtin_IMCE_OR(var149, var150, 15);
      var136 = __builtin_IMCE_ADD(var138, var132, 15);

      var151 = __builtin_IMCE_MULTL(var135, var131, 15);
      var152 = __builtin_IMCE_MULTH(var135, var131, 15);
      var153 = __builtin_IMCE_SUBI(0, 1);
      var154 = __builtin_IMCE_SRLI(var153, 1);
      var155 = __builtin_IMCE_SRAI(var152, 15);
      var156 = __builtin_IMCE_SRAI(var151, 15);
      var157 = __builtin_IMCE_XOR(var155, var156, 15);
      var158 = __builtin_IMCE_XOR(var155, var154, 15);
      var159 = __builtin_IMCE_XOR(var157, var153, 15);
      var160 = __builtin_IMCE_AND(var157, var158, 15);
      var161 = __builtin_IMCE_AND(var159, var151, 15);
      var139 = __builtin_IMCE_OR(var160, var161, 15);
      var137 = __builtin_IMCE_ADD(var139, var133, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var136, 2, 0); // TensorEdge((62, odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var137, 2, 0); // TensorEdge((62, odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, 0, 2, 0); // TensorEdge((62, odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, 0, 2, 0); // TensorEdge((62, odata), (70, func_out0), 0), imce_3_1 -> inode_3_0
      // endgenerate: bn_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 2) { // imce_3_2
    // generate: add const

    var162 = __builtin_IMCE_RECV(1);
    var163 = __builtin_IMCE_RECV(1);
    // endgenerate: add const
    // generate: call_created_loop
    for (int i1 = 0; i1 < 1024; i1++) { // generate : call_created_loop
      // generate: imcflow.vecops_wrapper

      var164 = __builtin_IMCE_RECV(2); // TensorEdge((68, odata), ((69, 63), lhs)), imce_3_3 -> imce_3_2
      // generate: imcflow.vecops_block
      // generate: add

      var165 = __builtin_IMCE_ADD(var162, var164, 15);
      // endgenerate: add
      // endgenerate: imcflow.vecops_block
      __builtin_IMCE_SEND(1, var165, 2, 0); // TensorEdge(((69, 63), odata), (70, func_out1), 1), imce_3_2 -> inode_2_0
      // endgenerate: imcflow.vecops_wrapper
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 3) { // imce_3_3
    // generate: mult const

    var166 = __builtin_IMCE_RECV(1);
    var167 = __builtin_IMCE_RECV(1);
    // endgenerate: mult const
    // generate: call_created_loop
    for (int i1 = 0; i1 < 1024; i1++) { // generate : call_created_loop
      // generate: multiply standalone

      var168 = __builtin_IMCE_RECV(2); // TensorEdge((67, odata), (68, lhs)), imce_2_3 -> imce_3_3
      // generate: multl


      var170 = __builtin_IMCE_MULTL(var166, var168, 15);
      var171 = __builtin_IMCE_MULTH(var166, var168, 15);
      var172 = __builtin_IMCE_SUBI(0, 1);
      var173 = __builtin_IMCE_SRLI(var172, 1);
      var174 = __builtin_IMCE_SRAI(var171, 15);
      var175 = __builtin_IMCE_SRAI(var170, 15);
      var176 = __builtin_IMCE_XOR(var174, var175, 15);
      var177 = __builtin_IMCE_XOR(var174, var173, 15);
      var178 = __builtin_IMCE_XOR(var176, var172, 15);
      var179 = __builtin_IMCE_AND(var176, var177, 15);
      var180 = __builtin_IMCE_AND(var178, var170, 15);
      var169 = __builtin_IMCE_OR(var179, var180, 15);
      // endgenerate: multl
      __builtin_IMCE_SEND(1, var169, 2, 0); // TensorEdge((68, odata), ((69, 63), lhs)), imce_3_3 -> imce_3_2
      // endgenerate: multiply standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 4) { // imce_3_4
  }
}
