#include "../common_decl.h"
void tvmgen_default_tvmgen_default_imcflow_main_46_round_imcflow_region3_main_20() {
  int hid = __builtin_IMCE_GET_CORE_HID();
  int wid = __builtin_IMCE_GET_CORE_WID();
  short16 var1; // (ConvBlock(gid: 87), 0)
  short16 var2; // (ConvBlock(gid: 87), 1)
  short16 var3; // (ConvBlock(gid: 87), 2)
  short16 var4; // (ConvBlock(gid: 87), 3)
  short16 var5; // (ConvBlock(gid: 81), 0)
  short16 var6; // (ConvBlock(gid: 81), 1)
  short16 var7; // (ConvBlock(gid: 81), 2)
  short16 var8; // (ConvBlock(gid: 81), 3)
  short16 var9; // (TensorEdge((87, odata), ((88, 82), rhs)), 0)
  short16 var10; // (TensorEdge((87, odata), ((88, 82), rhs)), 1)
  short16 var11; // (TensorEdge((87, odata), ((88, 82), rhs)), 2)
  short16 var12; // (TensorEdge((87, odata), ((88, 82), rhs)), 3)
  short16 var13; // (ConvBlock(gid: 94), 0)
  short16 var14; // (ConvBlock(gid: 94), 1)
  short16 var15; // (ConvBlock(gid: 94), 2)
  short16 var16; // (ConvBlock(gid: 94), 3)
  short16 var17; // (ConvBlock(gid: 91), 0)
  short16 var18; // (ConvBlock(gid: 91), 1)
  short16 var19; // (ConvBlock(gid: 91), 2)
  short16 var20; // (ConvBlock(gid: 91), 3)
  short16 var21; // (TensorEdge((94, odata), ((95, 92), rhs)), 0)
  short16 var22; // (TensorEdge((94, odata), ((95, 92), rhs)), 1)
  short16 var23; // (TensorEdge((94, odata), ((95, 92), rhs)), 2)
  short16 var24; // (TensorEdge((94, odata), ((95, 92), rhs)), 3)
  short16 var25; // (ConvBlock(gid: 75), 0)
  short16 var26; // (ConvBlock(gid: 75), 1)
  short16 var27; // (ConvBlock(gid: 75), 2)
  short16 var28; // (ConvBlock(gid: 75), 3)
  short16 var29; // (TensorEdge(((95, 92), odata), ((96, 76), lhs)), 0)
  short16 var30; // (TensorEdge(((95, 92), odata), ((96, 76), lhs)), 1)
  short16 var31; // (TensorEdge(((95, 92), odata), ((96, 76), lhs)), 2)
  short16 var32; // (TensorEdge(((95, 92), odata), ((96, 76), lhs)), 3)
  short16 var33; // (ConvBlock(gid: 103), 0)
  short16 var34; // (ConvBlock(gid: 103), 1)
  short16 var35; // (ConvBlock(gid: 103), 2)
  short16 var36; // (ConvBlock(gid: 103), 3)
  short16 var37; // (TensorEdge((-93, min), (102, min)), 0)
  short16 var38; // (TensorEdge((-94, max), (102, max)), 0)
  short16 var39; // (TensorEdge((84, odata), (102, data)), 0)
  short16 var40; // (TensorEdge((84, odata), (102, data)), 1)
  short16 var41; // (MinmaxQuantBlock(gid: 102), 0)
  short16 var42; // (MinmaxQuantBlock(gid: 102), 1)
  short16 var43; // (MinmaxQuantBlock(gid: 102), 2)
  short16 var44; // (MinmaxQuantBlock(gid: 102), 3)
  short16 var45; // (TensorEdge((-61, odata), (84, lhs)), 0)
  short16 var46; // (TensorEdge((-62, odata), (84, rhs)), 0)
  short16 var47; // (TensorEdge((-61, odata), (84, lhs)), 1)
  short16 var48; // (TensorEdge((-62, odata), (84, rhs)), 1)
  short16 var49; // (AddBlock(gid: 84), 0)
  short16 var50; // (AddBlock(gid: 84), 1)
  short16 var51; // (TensorEdge((-76, min), (85, min)), 0)
  short16 var52; // (TensorEdge((-77, max), (85, max)), 0)
  short16 var53; // (TensorEdge((84, odata), (85, data)), 0)
  short16 var54; // (TensorEdge((84, odata), (85, data)), 1)
  short16 var55; // (MinmaxQuantBlock(gid: 85), 0)
  short16 var56; // (MinmaxQuantBlock(gid: 85), 1)
  short16 var57; // (MinmaxQuantBlock(gid: 85), 2)
  short16 var58; // (MinmaxQuantBlock(gid: 85), 3)
  short16 var59; // (TensorEdge((-96, config), (103, config)), 0)
  short16 var60; // (TensorEdge((102, odata), (103, data)), 0)
  short16 var61; // (TensorEdge((102, odata), (103, data)), 1)
  short16 var62; // (TensorEdge((102, odata), (103, data)), 2)
  short16 var63; // (TensorEdge((102, odata), (103, data)), 3)
  short16 var64; // (TensorEdge(((88, -75), config), ((88, 81), config)), 0)
  short16 var65; // (AddBlock(gid: 82), 0)
  short16 var66; // (AddBlock(gid: 82), 1)
  short16 var67; // (AddBlock(gid: 82), 2)
  short16 var68; // (AddBlock(gid: 82), 3)
  short16 var69; // (TensorEdge((-79, config), (87, config)), 0)
  short16 var70; // (TensorEdge((86, odata), (87, data), 1), 0)
  short16 var71; // (TensorEdge((86, odata), (87, data), 1), 1)
  short16 var72; // (TensorEdge((86, odata), (87, data), 1), 2)
  short16 var73; // (TensorEdge((86, odata), (87, data), 1), 3)
  short16 var74; // (TensorEdge(((104, -91), fused_scale), ((104, 98), fused_scale)), 0)
  short16 var75; // (TensorEdge(((104, -91), fused_scale), ((104, 98), fused_scale)), 1)
  short16 var76; // (TensorEdge(((104, -91), fused_scale), ((104, 98), fused_scale)), 2)
  short16 var77; // (TensorEdge(((104, -91), fused_scale), ((104, 98), fused_scale)), 3)
  short16 var78; // (TensorEdge(((104, -92), fused_bias), ((104, 98), fused_bias)), 0)
  short16 var79; // (TensorEdge(((104, -92), fused_bias), ((104, 98), fused_bias)), 1)
  short16 var80; // (TensorEdge(((104, -92), fused_bias), ((104, 98), fused_bias)), 2)
  short16 var81; // (TensorEdge(((104, -92), fused_bias), ((104, 98), fused_bias)), 3)
  short16 var82; // (TensorEdge(((104, -90), scale), ((104, 99), rhs)), 0)
  short16 var83; // (TensorEdge(((104, -90), scale), ((104, 99), rhs)), 1)
  short16 var84; // (TensorEdge(((104, -90), scale), ((104, 99), rhs)), 2)
  short16 var85; // (TensorEdge(((104, -90), scale), ((104, 99), rhs)), 3)
  short16 var86; // (TensorEdge((-97, scale), ((104, 100), rhs)), 0)
  short16 var87; // (TensorEdge((-97, scale), ((104, 100), rhs)), 1)
  short16 var88; // (TensorEdge((-97, scale), ((104, 100), rhs)), 2)
  short16 var89; // (TensorEdge((-97, scale), ((104, 100), rhs)), 3)
  short16 var90; // (TensorEdge((103, odata), ((104, 98), data)), 0)
  short16 var91; // (TensorEdge((103, odata), ((104, 98), data)), 1)
  short16 var92; // (TensorEdge((103, odata), ((104, 98), data)), 2)
  short16 var93; // (TensorEdge((103, odata), ((104, 98), data)), 3)
  short16 var94; // (AddBlock(gid: 100), 0)
  short16 var95; // (AddBlock(gid: 100), 1)
  short16 var96; // (AddBlock(gid: 100), 2)
  short16 var97; // (AddBlock(gid: 100), 3)
  short16 var98; // (BatchNormBlock(gid: 98), 0)
  short16 var99; // (BatchNormBlock(gid: 98), 0, 'mult_result')
  short16 var100; // (BatchNormBlock(gid: 98), 1)
  short16 var101; // (BatchNormBlock(gid: 98), 1, 'mult_result')
  short16 var102; // (BatchNormBlock(gid: 98), 2)
  short16 var103; // (BatchNormBlock(gid: 98), 2, 'mult_result')
  short16 var104; // (BatchNormBlock(gid: 98), 3)
  short16 var105; // (BatchNormBlock(gid: 98), 3, 'mult_result')
  short16 var106; // ((BatchNormBlock(gid: 98), 0), 'L')
  short16 var107; // ((BatchNormBlock(gid: 98), 0), 'H')
  short16 var108; // ((BatchNormBlock(gid: 98), 0), 'neg1')
  short16 var109; // ((BatchNormBlock(gid: 98), 0), 'const_7fff')
  short16 var110; // ((BatchNormBlock(gid: 98), 0), 'H_sign')
  short16 var111; // ((BatchNormBlock(gid: 98), 0), 'L_sign')
  short16 var112; // ((BatchNormBlock(gid: 98), 0), 'mismatch')
  short16 var113; // ((BatchNormBlock(gid: 98), 0), 'saturate_val')
  short16 var114; // ((BatchNormBlock(gid: 98), 0), 'not_mismatch')
  short16 var115; // ((BatchNormBlock(gid: 98), 0), 'part1')
  short16 var116; // ((BatchNormBlock(gid: 98), 0), 'part2')
  short16 var117; // ((BatchNormBlock(gid: 98), 1), 'L')
  short16 var118; // ((BatchNormBlock(gid: 98), 1), 'H')
  short16 var119; // ((BatchNormBlock(gid: 98), 1), 'neg1')
  short16 var120; // ((BatchNormBlock(gid: 98), 1), 'const_7fff')
  short16 var121; // ((BatchNormBlock(gid: 98), 1), 'H_sign')
  short16 var122; // ((BatchNormBlock(gid: 98), 1), 'L_sign')
  short16 var123; // ((BatchNormBlock(gid: 98), 1), 'mismatch')
  short16 var124; // ((BatchNormBlock(gid: 98), 1), 'saturate_val')
  short16 var125; // ((BatchNormBlock(gid: 98), 1), 'not_mismatch')
  short16 var126; // ((BatchNormBlock(gid: 98), 1), 'part1')
  short16 var127; // ((BatchNormBlock(gid: 98), 1), 'part2')
  short16 var128; // ((BatchNormBlock(gid: 98), 2), 'L')
  short16 var129; // ((BatchNormBlock(gid: 98), 2), 'H')
  short16 var130; // ((BatchNormBlock(gid: 98), 2), 'neg1')
  short16 var131; // ((BatchNormBlock(gid: 98), 2), 'const_7fff')
  short16 var132; // ((BatchNormBlock(gid: 98), 2), 'H_sign')
  short16 var133; // ((BatchNormBlock(gid: 98), 2), 'L_sign')
  short16 var134; // ((BatchNormBlock(gid: 98), 2), 'mismatch')
  short16 var135; // ((BatchNormBlock(gid: 98), 2), 'saturate_val')
  short16 var136; // ((BatchNormBlock(gid: 98), 2), 'not_mismatch')
  short16 var137; // ((BatchNormBlock(gid: 98), 2), 'part1')
  short16 var138; // ((BatchNormBlock(gid: 98), 2), 'part2')
  short16 var139; // ((BatchNormBlock(gid: 98), 3), 'L')
  short16 var140; // ((BatchNormBlock(gid: 98), 3), 'H')
  short16 var141; // ((BatchNormBlock(gid: 98), 3), 'neg1')
  short16 var142; // ((BatchNormBlock(gid: 98), 3), 'const_7fff')
  short16 var143; // ((BatchNormBlock(gid: 98), 3), 'H_sign')
  short16 var144; // ((BatchNormBlock(gid: 98), 3), 'L_sign')
  short16 var145; // ((BatchNormBlock(gid: 98), 3), 'mismatch')
  short16 var146; // ((BatchNormBlock(gid: 98), 3), 'saturate_val')
  short16 var147; // ((BatchNormBlock(gid: 98), 3), 'not_mismatch')
  short16 var148; // ((BatchNormBlock(gid: 98), 3), 'part1')
  short16 var149; // ((BatchNormBlock(gid: 98), 3), 'part2')
  short16 var150; // (MultlBlock(gid: 99), 0)
  short16 var151; // (MultlBlock(gid: 99), 1)
  short16 var152; // (MultlBlock(gid: 99), 2)
  short16 var153; // (MultlBlock(gid: 99), 3)
  short16 var154; // ((MultlBlock(gid: 99), 0), 'L')
  short16 var155; // ((MultlBlock(gid: 99), 0), 'H')
  short16 var156; // ((MultlBlock(gid: 99), 0), 'neg1')
  short16 var157; // ((MultlBlock(gid: 99), 0), 'const_7fff')
  short16 var158; // ((MultlBlock(gid: 99), 0), 'H_sign')
  short16 var159; // ((MultlBlock(gid: 99), 0), 'L_sign')
  short16 var160; // ((MultlBlock(gid: 99), 0), 'mismatch')
  short16 var161; // ((MultlBlock(gid: 99), 0), 'saturate_val')
  short16 var162; // ((MultlBlock(gid: 99), 0), 'not_mismatch')
  short16 var163; // ((MultlBlock(gid: 99), 0), 'part1')
  short16 var164; // ((MultlBlock(gid: 99), 0), 'part2')
  short16 var165; // ((MultlBlock(gid: 99), 1), 'L')
  short16 var166; // ((MultlBlock(gid: 99), 1), 'H')
  short16 var167; // ((MultlBlock(gid: 99), 1), 'neg1')
  short16 var168; // ((MultlBlock(gid: 99), 1), 'const_7fff')
  short16 var169; // ((MultlBlock(gid: 99), 1), 'H_sign')
  short16 var170; // ((MultlBlock(gid: 99), 1), 'L_sign')
  short16 var171; // ((MultlBlock(gid: 99), 1), 'mismatch')
  short16 var172; // ((MultlBlock(gid: 99), 1), 'saturate_val')
  short16 var173; // ((MultlBlock(gid: 99), 1), 'not_mismatch')
  short16 var174; // ((MultlBlock(gid: 99), 1), 'part1')
  short16 var175; // ((MultlBlock(gid: 99), 1), 'part2')
  short16 var176; // ((MultlBlock(gid: 99), 2), 'L')
  short16 var177; // ((MultlBlock(gid: 99), 2), 'H')
  short16 var178; // ((MultlBlock(gid: 99), 2), 'neg1')
  short16 var179; // ((MultlBlock(gid: 99), 2), 'const_7fff')
  short16 var180; // ((MultlBlock(gid: 99), 2), 'H_sign')
  short16 var181; // ((MultlBlock(gid: 99), 2), 'L_sign')
  short16 var182; // ((MultlBlock(gid: 99), 2), 'mismatch')
  short16 var183; // ((MultlBlock(gid: 99), 2), 'saturate_val')
  short16 var184; // ((MultlBlock(gid: 99), 2), 'not_mismatch')
  short16 var185; // ((MultlBlock(gid: 99), 2), 'part1')
  short16 var186; // ((MultlBlock(gid: 99), 2), 'part2')
  short16 var187; // ((MultlBlock(gid: 99), 3), 'L')
  short16 var188; // ((MultlBlock(gid: 99), 3), 'H')
  short16 var189; // ((MultlBlock(gid: 99), 3), 'neg1')
  short16 var190; // ((MultlBlock(gid: 99), 3), 'const_7fff')
  short16 var191; // ((MultlBlock(gid: 99), 3), 'H_sign')
  short16 var192; // ((MultlBlock(gid: 99), 3), 'L_sign')
  short16 var193; // ((MultlBlock(gid: 99), 3), 'mismatch')
  short16 var194; // ((MultlBlock(gid: 99), 3), 'saturate_val')
  short16 var195; // ((MultlBlock(gid: 99), 3), 'not_mismatch')
  short16 var196; // ((MultlBlock(gid: 99), 3), 'part1')
  short16 var197; // ((MultlBlock(gid: 99), 3), 'part2')
  short16 var198; // (TensorEdge(((89, -68), fused_scale), ((89, 78), fused_scale)), 0)
  short16 var199; // (TensorEdge(((89, -68), fused_scale), ((89, 78), fused_scale)), 1)
  short16 var200; // (TensorEdge(((89, -68), fused_scale), ((89, 78), fused_scale)), 2)
  short16 var201; // (TensorEdge(((89, -68), fused_scale), ((89, 78), fused_scale)), 3)
  short16 var202; // (TensorEdge(((89, -69), fused_bias), ((89, 78), fused_bias)), 0)
  short16 var203; // (TensorEdge(((89, -69), fused_bias), ((89, 78), fused_bias)), 1)
  short16 var204; // (TensorEdge(((89, -69), fused_bias), ((89, 78), fused_bias)), 2)
  short16 var205; // (TensorEdge(((89, -69), fused_bias), ((89, 78), fused_bias)), 3)
  short16 var206; // (TensorEdge(((89, -70), min), ((89, 79), min)), 0)
  short16 var207; // (TensorEdge(((89, -71), max), ((89, 79), max)), 0)
  short16 var208; // (TensorEdge(((88, 82), odata), ((89, 78), data)), 0)
  short16 var209; // (TensorEdge(((88, 82), odata), ((89, 78), data)), 1)
  short16 var210; // (TensorEdge(((88, 82), odata), ((89, 78), data)), 2)
  short16 var211; // (TensorEdge(((88, 82), odata), ((89, 78), data)), 3)
  short16 var212; // (MinmaxQuantBlock(gid: 79), 0)
  short16 var213; // (MinmaxQuantBlock(gid: 79), 1)
  short16 var214; // (MinmaxQuantBlock(gid: 79), 2)
  short16 var215; // (MinmaxQuantBlock(gid: 79), 3)
  short16 var216; // (BatchNormBlock(gid: 78), 0)
  short16 var217; // (BatchNormBlock(gid: 78), 0, 'mult_result')
  short16 var218; // (BatchNormBlock(gid: 78), 1)
  short16 var219; // (BatchNormBlock(gid: 78), 1, 'mult_result')
  short16 var220; // (BatchNormBlock(gid: 78), 2)
  short16 var221; // (BatchNormBlock(gid: 78), 2, 'mult_result')
  short16 var222; // (BatchNormBlock(gid: 78), 3)
  short16 var223; // (BatchNormBlock(gid: 78), 3, 'mult_result')
  short16 var224; // ((BatchNormBlock(gid: 78), 0), 'L')
  short16 var225; // ((BatchNormBlock(gid: 78), 0), 'H')
  short16 var226; // ((BatchNormBlock(gid: 78), 0), 'neg1')
  short16 var227; // ((BatchNormBlock(gid: 78), 0), 'const_7fff')
  short16 var228; // ((BatchNormBlock(gid: 78), 0), 'H_sign')
  short16 var229; // ((BatchNormBlock(gid: 78), 0), 'L_sign')
  short16 var230; // ((BatchNormBlock(gid: 78), 0), 'mismatch')
  short16 var231; // ((BatchNormBlock(gid: 78), 0), 'saturate_val')
  short16 var232; // ((BatchNormBlock(gid: 78), 0), 'not_mismatch')
  short16 var233; // ((BatchNormBlock(gid: 78), 0), 'part1')
  short16 var234; // ((BatchNormBlock(gid: 78), 0), 'part2')
  short16 var235; // ((BatchNormBlock(gid: 78), 1), 'L')
  short16 var236; // ((BatchNormBlock(gid: 78), 1), 'H')
  short16 var237; // ((BatchNormBlock(gid: 78), 1), 'neg1')
  short16 var238; // ((BatchNormBlock(gid: 78), 1), 'const_7fff')
  short16 var239; // ((BatchNormBlock(gid: 78), 1), 'H_sign')
  short16 var240; // ((BatchNormBlock(gid: 78), 1), 'L_sign')
  short16 var241; // ((BatchNormBlock(gid: 78), 1), 'mismatch')
  short16 var242; // ((BatchNormBlock(gid: 78), 1), 'saturate_val')
  short16 var243; // ((BatchNormBlock(gid: 78), 1), 'not_mismatch')
  short16 var244; // ((BatchNormBlock(gid: 78), 1), 'part1')
  short16 var245; // ((BatchNormBlock(gid: 78), 1), 'part2')
  short16 var246; // ((BatchNormBlock(gid: 78), 2), 'L')
  short16 var247; // ((BatchNormBlock(gid: 78), 2), 'H')
  short16 var248; // ((BatchNormBlock(gid: 78), 2), 'neg1')
  short16 var249; // ((BatchNormBlock(gid: 78), 2), 'const_7fff')
  short16 var250; // ((BatchNormBlock(gid: 78), 2), 'H_sign')
  short16 var251; // ((BatchNormBlock(gid: 78), 2), 'L_sign')
  short16 var252; // ((BatchNormBlock(gid: 78), 2), 'mismatch')
  short16 var253; // ((BatchNormBlock(gid: 78), 2), 'saturate_val')
  short16 var254; // ((BatchNormBlock(gid: 78), 2), 'not_mismatch')
  short16 var255; // ((BatchNormBlock(gid: 78), 2), 'part1')
  short16 var256; // ((BatchNormBlock(gid: 78), 2), 'part2')
  short16 var257; // ((BatchNormBlock(gid: 78), 3), 'L')
  short16 var258; // ((BatchNormBlock(gid: 78), 3), 'H')
  short16 var259; // ((BatchNormBlock(gid: 78), 3), 'neg1')
  short16 var260; // ((BatchNormBlock(gid: 78), 3), 'const_7fff')
  short16 var261; // ((BatchNormBlock(gid: 78), 3), 'H_sign')
  short16 var262; // ((BatchNormBlock(gid: 78), 3), 'L_sign')
  short16 var263; // ((BatchNormBlock(gid: 78), 3), 'mismatch')
  short16 var264; // ((BatchNormBlock(gid: 78), 3), 'saturate_val')
  short16 var265; // ((BatchNormBlock(gid: 78), 3), 'not_mismatch')
  short16 var266; // ((BatchNormBlock(gid: 78), 3), 'part1')
  short16 var267; // ((BatchNormBlock(gid: 78), 3), 'part2')
  short16 var268; // (TensorEdge((-85, config), (94, config)), 0)
  short16 var269; // (TensorEdge((90, odata), (94, data), 1), 0)
  short16 var270; // (TensorEdge((90, odata), (94, data), 1), 1)
  short16 var271; // (TensorEdge((90, odata), (94, data), 1), 2)
  short16 var272; // (TensorEdge((90, odata), (94, data), 1), 3)
  short16 var273; // (TensorEdge((-86, fused_scale), (97, fused_scale)), 0)
  short16 var274; // (TensorEdge((-86, fused_scale), (97, fused_scale)), 1)
  short16 var275; // (TensorEdge((-86, fused_scale), (97, fused_scale)), 2)
  short16 var276; // (TensorEdge((-86, fused_scale), (97, fused_scale)), 3)
  short16 var277; // (TensorEdge((-87, fused_bias), (97, fused_bias)), 0)
  short16 var278; // (TensorEdge((-87, fused_bias), (97, fused_bias)), 1)
  short16 var279; // (TensorEdge((-87, fused_bias), (97, fused_bias)), 2)
  short16 var280; // (TensorEdge((-87, fused_bias), (97, fused_bias)), 3)
  short16 var281; // (TensorEdge(((96, 76), odata), (97, data)), 0)
  short16 var282; // (TensorEdge(((96, 76), odata), (97, data)), 1)
  short16 var283; // (TensorEdge(((96, 76), odata), (97, data)), 2)
  short16 var284; // (TensorEdge(((96, 76), odata), (97, data)), 3)
  short16 var285; // (BatchNormBlock(gid: 97), 0)
  short16 var286; // (BatchNormBlock(gid: 97), 1)
  short16 var287; // (BatchNormBlock(gid: 97), 2)
  short16 var288; // (BatchNormBlock(gid: 97), 3)
  short16 var289; // (BatchNormBlock(gid: 97), 0, 'mult_result')
  short16 var290; // (BatchNormBlock(gid: 97), 1, 'mult_result')
  short16 var291; // (BatchNormBlock(gid: 97), 2, 'mult_result')
  short16 var292; // (BatchNormBlock(gid: 97), 3, 'mult_result')
  short16 var293; // ((BatchNormBlock(gid: 97), 0), 'L')
  short16 var294; // ((BatchNormBlock(gid: 97), 0), 'H')
  short16 var295; // ((BatchNormBlock(gid: 97), 0), 'neg1')
  short16 var296; // ((BatchNormBlock(gid: 97), 0), 'const_7fff')
  short16 var297; // ((BatchNormBlock(gid: 97), 0), 'H_sign')
  short16 var298; // ((BatchNormBlock(gid: 97), 0), 'L_sign')
  short16 var299; // ((BatchNormBlock(gid: 97), 0), 'mismatch')
  short16 var300; // ((BatchNormBlock(gid: 97), 0), 'saturate_val')
  short16 var301; // ((BatchNormBlock(gid: 97), 0), 'not_mismatch')
  short16 var302; // ((BatchNormBlock(gid: 97), 0), 'part1')
  short16 var303; // ((BatchNormBlock(gid: 97), 0), 'part2')
  short16 var304; // ((BatchNormBlock(gid: 97), 1), 'L')
  short16 var305; // ((BatchNormBlock(gid: 97), 1), 'H')
  short16 var306; // ((BatchNormBlock(gid: 97), 1), 'neg1')
  short16 var307; // ((BatchNormBlock(gid: 97), 1), 'const_7fff')
  short16 var308; // ((BatchNormBlock(gid: 97), 1), 'H_sign')
  short16 var309; // ((BatchNormBlock(gid: 97), 1), 'L_sign')
  short16 var310; // ((BatchNormBlock(gid: 97), 1), 'mismatch')
  short16 var311; // ((BatchNormBlock(gid: 97), 1), 'saturate_val')
  short16 var312; // ((BatchNormBlock(gid: 97), 1), 'not_mismatch')
  short16 var313; // ((BatchNormBlock(gid: 97), 1), 'part1')
  short16 var314; // ((BatchNormBlock(gid: 97), 1), 'part2')
  short16 var315; // ((BatchNormBlock(gid: 97), 2), 'L')
  short16 var316; // ((BatchNormBlock(gid: 97), 2), 'H')
  short16 var317; // ((BatchNormBlock(gid: 97), 2), 'neg1')
  short16 var318; // ((BatchNormBlock(gid: 97), 2), 'const_7fff')
  short16 var319; // ((BatchNormBlock(gid: 97), 2), 'H_sign')
  short16 var320; // ((BatchNormBlock(gid: 97), 2), 'L_sign')
  short16 var321; // ((BatchNormBlock(gid: 97), 2), 'mismatch')
  short16 var322; // ((BatchNormBlock(gid: 97), 2), 'saturate_val')
  short16 var323; // ((BatchNormBlock(gid: 97), 2), 'not_mismatch')
  short16 var324; // ((BatchNormBlock(gid: 97), 2), 'part1')
  short16 var325; // ((BatchNormBlock(gid: 97), 2), 'part2')
  short16 var326; // ((BatchNormBlock(gid: 97), 3), 'L')
  short16 var327; // ((BatchNormBlock(gid: 97), 3), 'H')
  short16 var328; // ((BatchNormBlock(gid: 97), 3), 'neg1')
  short16 var329; // ((BatchNormBlock(gid: 97), 3), 'const_7fff')
  short16 var330; // ((BatchNormBlock(gid: 97), 3), 'H_sign')
  short16 var331; // ((BatchNormBlock(gid: 97), 3), 'L_sign')
  short16 var332; // ((BatchNormBlock(gid: 97), 3), 'mismatch')
  short16 var333; // ((BatchNormBlock(gid: 97), 3), 'saturate_val')
  short16 var334; // ((BatchNormBlock(gid: 97), 3), 'not_mismatch')
  short16 var335; // ((BatchNormBlock(gid: 97), 3), 'part1')
  short16 var336; // ((BatchNormBlock(gid: 97), 3), 'part2')
  short16 var337; // (TensorEdge(((96, -66), config), ((96, 75), config)), 0)
  short16 var338; // (AddBlock(gid: 76), 0)
  short16 var339; // (AddBlock(gid: 76), 1)
  short16 var340; // (AddBlock(gid: 76), 2)
  short16 var341; // (AddBlock(gid: 76), 3)
  short16 var342; // (TensorEdge(((95, -83), config), ((95, 91), config)), 0)
  short16 var343; // (AddBlock(gid: 92), 0)
  short16 var344; // (AddBlock(gid: 92), 1)
  short16 var345; // (AddBlock(gid: 92), 2)
  short16 var346; // (AddBlock(gid: 92), 3)
  if (hid == 0 && wid == 1) { // imce_0_1
    // generate: TensorEdge((-93, min), (102, min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge((-93, min), (102, min)), min write
    // generate: TensorEdge((-94, max), (102, max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge((-94, max), (102, max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 256; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      __builtin_IMCE_SETFLAG(2);
      __builtin_IMCE_STANDBY(2, 2); 
      __builtin_IMCE_SETFLAG(0); 
      var39 = __builtin_IMCE_RECV(2); // TensorEdge((84, odata), (102, data)), imce_0_2 -> imce_0_1
      var40 = __builtin_IMCE_RECV(2); // TensorEdge((84, odata), (102, data)), imce_0_2 -> imce_0_1
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var39, 0, 15, 0);
      __builtin_IMCE_MM_QUANT(var40, 0, 15, 1);
      var41 = __builtin_IMCE_GET_QREG(0);
      var42 = __builtin_IMCE_GET_QREG(1);
      var43 = __builtin_IMCE_GET_QREG(2);
      var44 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize

      __builtin_IMCE_STANDBY(6, 1);       
      __builtin_IMCE_SEND(5, var41, 0, 0); // TensorEdge((102, odata), (103, data)), imce_0_1 -> imce_1_1
      __builtin_IMCE_SEND(5, var42, 0, 0); // TensorEdge((102, odata), (103, data)), imce_0_1 -> imce_1_1
      __builtin_IMCE_SEND(5, var43, 0, 0); // TensorEdge((102, odata), (103, data)), imce_0_1 -> imce_1_1
      __builtin_IMCE_SEND(5, var44, 0, 0); // TensorEdge((102, odata), (103, data)), imce_0_1 -> imce_1_1
      // endgenerate: min_max_quantize_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 0 && wid == 2) { // imce_0_2
    // generate: call_created_loop
    for (int i1 = 0; i1 < 256; i1++) { // generate : call_created_loop
      // generate: add standalone

      __builtin_IMCE_SETFLAG(1);
      __builtin_IMCE_STANDBY(0, 1); 
      __builtin_IMCE_STANDBY(5, 1); 
      __builtin_IMCE_SETFLAG(0); 
      var45 = __builtin_IMCE_RECV(2); // TensorEdge((-61, odata), (84, lhs)), inode_0_0 -> imce_0_2
      var46 = __builtin_IMCE_RECV(3); // TensorEdge((-62, odata), (84, rhs)), inode_1_0 -> imce_0_2
      __builtin_IMCE_SETFLAG(1);
      __builtin_IMCE_STANDBY(0, 1); 
      __builtin_IMCE_STANDBY(5, 1); 
      __builtin_IMCE_SETFLAG(0); 
      var47 = __builtin_IMCE_RECV(2); // TensorEdge((-61, odata), (84, lhs)), inode_0_0 -> imce_0_2
      var48 = __builtin_IMCE_RECV(3); // TensorEdge((-62, odata), (84, rhs)), inode_1_0 -> imce_0_2
      // generate: add

      var49 = __builtin_IMCE_ADD(var45, var46, 15);
      var50 = __builtin_IMCE_ADD(var47, var48, 15);
      // endgenerate: add

      __builtin_IMCE_STANDBY(1, 2);
      __builtin_IMCE_STANDBY(3, 2);
      __builtin_IMCE_SETFLAG(2);
      __builtin_IMCE_STANDBY(1, 0);
      __builtin_IMCE_STANDBY(3, 0);
      __builtin_IMCE_SETFLAG(0);
      __builtin_IMCE_SEND(2, var49, 2, 0); // TensorEdge((84, odata), (102, data)),TensorEdge((84, odata), (85, data)), imce_0_2 -> imce_0_3
      __builtin_IMCE_SEND(2, var50, 2, 0); // TensorEdge((84, odata), (102, data)),TensorEdge((84, odata), (85, data)), imce_0_2 -> imce_0_3
      // endgenerate: add standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 0 && wid == 3) { // imce_0_3
    // generate: TensorEdge((-76, min), (85, min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge((-76, min), (85, min)), min write
    // generate: TensorEdge((-77, max), (85, max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge((-77, max), (85, max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 256; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      __builtin_IMCE_SETFLAG(2);
      __builtin_IMCE_STANDBY(2, 2); 
      __builtin_IMCE_SETFLAG(0); 
      var53 = __builtin_IMCE_RECV(2); // TensorEdge((84, odata), (85, data)), imce_0_2 -> imce_0_3
      var54 = __builtin_IMCE_RECV(2); // TensorEdge((84, odata), (85, data)), imce_0_2 -> imce_0_3
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var53, 0, 15, 0);
      __builtin_IMCE_MM_QUANT(var54, 0, 15, 1);
      var55 = __builtin_IMCE_GET_QREG(0);
      var56 = __builtin_IMCE_GET_QREG(1);
      var57 = __builtin_IMCE_GET_QREG(2);
      var58 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      __builtin_IMCE_STANDBY(7, 3);
      __builtin_IMCE_STANDBY(8, 3);
      __builtin_IMCE_SEND(1, var55, 0, 0); // TensorEdge((85, odata), (86, data)), imce_0_3 -> imce_1_2
      __builtin_IMCE_SEND(1, var56, 0, 0); // TensorEdge((85, odata), (86, data)), imce_0_3 -> imce_1_2
      __builtin_IMCE_SEND(1, var57, 0, 0); // TensorEdge((85, odata), (86, data)), imce_0_3 -> imce_1_2
      __builtin_IMCE_SEND(1, var58, 0, 0); // TensorEdge((85, odata), (86, data)), imce_0_3 -> imce_1_2
      // endgenerate: min_max_quantize_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 0 && wid == 4) { // imce_0_4
  }
  else if (hid == 1 && wid == 1) { // imce_1_1
    // generate: TensorEdge((-96, config), (103, config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge((-96, config), (103, config)), config write
    // generate: conv exec11
    // generate: conv exec11_row_group0_outer_loop(iterate row offset)
    // generate : conv exec11_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec11_row_group0_col_group0
    // generate : conv exec11_row_group0_col_group0. loop count == 1

    // generate: load_block
    // generate : load_block. loop count == 1
    __builtin_IMCE_SETFLAG(1);
    for (int i1 = 0; i1 < 4; i1++) { // generate
      __builtin_IMCE_LOAD_LB(0); // TensorEdge((102, odata), (103, data)), imce_0_1 -> imce_1_1

    } // endgenerate
    __builtin_IMCE_SETFLAG(0);
    // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var33 = __builtin_IMCE_GET_CREG((short)0);
    var34 = __builtin_IMCE_GET_CREG((short)1);
    var35 = __builtin_IMCE_GET_CREG((short)2);
    var36 = __builtin_IMCE_GET_CREG((short)3);
    __builtin_IMCE_SEND(6, var33, 2, 0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1
    __builtin_IMCE_SEND(6, var34, 2, 0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1
    __builtin_IMCE_SEND(6, var35, 2, 0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1
    __builtin_IMCE_SEND(6, var36, 2, 0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1
    // endgenerate : conv exec11_row_group0_col_group0
    // endgenerate: conv exec11_row_group0_col_group0
    // generate: conv exec11_row_group0_col_group1
    for (int i1 = 0; i1 < 7; i1++) { // generate : conv exec11_row_group0_col_group1

      // generate: load_block
      for (int i2 = 0; i2 < 2; i2++) { // generate : load_block
        __builtin_IMCE_SETFLAG(1);
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((102, odata), (103, data)), imce_0_1 -> imce_1_1

        } // endgenerate
        __builtin_IMCE_SETFLAG(0);
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var33 = __builtin_IMCE_GET_CREG((short)0);
      var34 = __builtin_IMCE_GET_CREG((short)1);
      var35 = __builtin_IMCE_GET_CREG((short)2);
      var36 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(6, var33, 2, 0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(6, var34, 2, 0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(6, var35, 2, 0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(6, var36, 2, 0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1
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
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((102, odata), (103, data)), imce_0_1 -> imce_1_1

        } // endgenerate
        __builtin_IMCE_SETFLAG(0);
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var33 = __builtin_IMCE_GET_CREG((short)0);
      var34 = __builtin_IMCE_GET_CREG((short)1);
      var35 = __builtin_IMCE_GET_CREG((short)2);
      var36 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(6, var33, 2, 0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(6, var34, 2, 0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(6, var35, 2, 0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(6, var36, 2, 0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1
      // endgenerate : conv exec11_row_group1_col_group0
      // endgenerate: conv exec11_row_group1_col_group0
      // generate: conv exec11_row_group1_col_group1
      for (int i2 = 0; i2 < 7; i2++) { // generate : conv exec11_row_group1_col_group1

        // generate: load_block
        for (int i3 = 0; i3 < 2; i3++) { // generate : load_block
          __builtin_IMCE_SETFLAG(1);
          for (int i4 = 0; i4 < 4; i4++) { // generate
            __builtin_IMCE_LOAD_LB(0); // TensorEdge((102, odata), (103, data)), imce_0_1 -> imce_1_1

          } // endgenerate
          __builtin_IMCE_SETFLAG(0);
        } // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var33 = __builtin_IMCE_GET_CREG((short)0);
        var34 = __builtin_IMCE_GET_CREG((short)1);
        var35 = __builtin_IMCE_GET_CREG((short)2);
        var36 = __builtin_IMCE_GET_CREG((short)3);
        __builtin_IMCE_SEND(6, var33, 2, 0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1
        __builtin_IMCE_SEND(6, var34, 2, 0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1
        __builtin_IMCE_SEND(6, var35, 2, 0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1
        __builtin_IMCE_SEND(6, var36, 2, 0); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1
      } // endgenerate : conv exec11_row_group1_col_group1
      // endgenerate: conv exec11_row_group1_col_group1
    } // endgenerate : conv exec11_row_group1_outer_loop(iterate row offset)
    // endgenerate: conv exec11_row_group1_outer_loop(iterate row offset)
    // generate: conv exec11_tail_loop
    for (int i1 = 0; i1 < 17; i1++) { // generate : conv exec11_tail_loop
      __builtin_IMCE_SETFLAG(1);
      for (int i2 = 0; i2 < 4; i2++) { // generate : conv exec11_tail_loop.inner
        __builtin_IMCE_RECV(0);
      }
      __builtin_IMCE_SETFLAG(0);
    } // endgenerate : conv exec11_tail_loop
    // endgenerate: conv exec11_tail_loop
    // endgenerate: conv exec11
    __builtin_IMCE_STOP();
  }
  else if (hid == 1 && wid == 2) { // imce_1_2
    // generate: TensorEdge(((88, -75), config), ((88, 81), config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge(((88, -75), config), ((88, 81), config)), config write
    // generate: conv exec7
    // generate: conv exec7_row_group0_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 8; i1++) { // generate : conv exec7_row_group0_outer_loop(iterate row offset)
      // generate: conv exec7_row_group0_col_group0
      // generate : conv exec7_row_group0_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 18; i2++) { // generate : load_block
        __builtin_IMCE_SETFLAG(3);
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((86, odata), ((88, 81), data), 0), imce_0_3 -> imce_1_2

        } // endgenerate
        __builtin_IMCE_SETFLAG(0);
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
      __builtin_IMCE_SETFLAG(1);
      var9 = __builtin_IMCE_RECV(2); // TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
      // endgenerate: TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
      // generate: TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
      var10 = __builtin_IMCE_RECV(2); // TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
      // endgenerate: TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
      // generate: TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
      var11 = __builtin_IMCE_RECV(2); // TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
      // endgenerate: TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
      // generate: TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
      var12 = __builtin_IMCE_RECV(2); // TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
      __builtin_IMCE_SETFLAG(0);
      // endgenerate: TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
      // generate: add

      var65 = __builtin_IMCE_ADD(var5, var9, 15);
      var66 = __builtin_IMCE_ADD(var6, var10, 15);
      var67 = __builtin_IMCE_ADD(var7, var11, 15);
      var68 = __builtin_IMCE_ADD(var8, var12, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(1, var65, 2, 0); // TensorEdge(((88, 82), odata), ((89, 78), data)), imce_1_2 -> imce_2_2
      __builtin_IMCE_SEND(1, var66, 2, 0); // TensorEdge(((88, 82), odata), ((89, 78), data)), imce_1_2 -> imce_2_2
      __builtin_IMCE_SEND(1, var67, 2, 0); // TensorEdge(((88, 82), odata), ((89, 78), data)), imce_1_2 -> imce_2_2
      __builtin_IMCE_SEND(1, var68, 2, 0); // TensorEdge(((88, 82), odata), ((89, 78), data)), imce_1_2 -> imce_2_2
      // endgenerate : conv exec7_row_group0_col_group0
      // endgenerate: conv exec7_row_group0_col_group0
      // generate: conv exec7_row_group0_col_group1
      for (int i2 = 0; i2 < 7; i2++) { // generate : conv exec7_row_group0_col_group1

        // generate: load_block
        for (int i3 = 0; i3 < 2; i3++) { // generate : load_block
          __builtin_IMCE_SETFLAG(3);
          for (int i4 = 0; i4 < 4; i4++) { // generate
            __builtin_IMCE_LOAD_LB(0); // TensorEdge((86, odata), ((88, 81), data), 0), imce_0_3 -> imce_1_2

          } // endgenerate
          __builtin_IMCE_SETFLAG(0);
        } // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var5 = __builtin_IMCE_GET_CREG((short)0);
        var6 = __builtin_IMCE_GET_CREG((short)1);
        var7 = __builtin_IMCE_GET_CREG((short)2);
        var8 = __builtin_IMCE_GET_CREG((short)3);
        // generate: TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
        __builtin_IMCE_SETFLAG(1);
        var9 = __builtin_IMCE_RECV(2); // TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
        // endgenerate: TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
        // generate: TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
        var10 = __builtin_IMCE_RECV(2); // TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
        // endgenerate: TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
        // generate: TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
        var11 = __builtin_IMCE_RECV(2); // TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
        // endgenerate: TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
        // generate: TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
        var12 = __builtin_IMCE_RECV(2); // TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
        __builtin_IMCE_SETFLAG(0);
        // endgenerate: TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
        // generate: add

        var65 = __builtin_IMCE_ADD(var5, var9, 15);
        var66 = __builtin_IMCE_ADD(var6, var10, 15);
        var67 = __builtin_IMCE_ADD(var7, var11, 15);
        var68 = __builtin_IMCE_ADD(var8, var12, 15);
        // endgenerate: add
        __builtin_IMCE_SEND(1, var65, 2, 0); // TensorEdge(((88, 82), odata), ((89, 78), data)), imce_1_2 -> imce_2_2
        __builtin_IMCE_SEND(1, var66, 2, 0); // TensorEdge(((88, 82), odata), ((89, 78), data)), imce_1_2 -> imce_2_2
        __builtin_IMCE_SEND(1, var67, 2, 0); // TensorEdge(((88, 82), odata), ((89, 78), data)), imce_1_2 -> imce_2_2
        __builtin_IMCE_SEND(1, var68, 2, 0); // TensorEdge(((88, 82), odata), ((89, 78), data)), imce_1_2 -> imce_2_2
      } // endgenerate : conv exec7_row_group0_col_group1
      // endgenerate: conv exec7_row_group0_col_group1
    } // endgenerate : conv exec7_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec7_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec7
    __builtin_IMCE_STOP();
  }
  else if (hid == 1 && wid == 3) { // imce_1_3
    // generate: TensorEdge((-79, config), (87, config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge((-79, config), (87, config)), config write
    // generate: conv exec6
    // generate: conv exec6_row_group0_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 8; i1++) { // generate : conv exec6_row_group0_outer_loop(iterate row offset)
      // generate: conv exec6_row_group0_col_group0
      // generate : conv exec6_row_group0_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 18; i2++) { // generate : load_block
        __builtin_IMCE_SETFLAG(3);
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((86, odata), (87, data), 1), imce_0_3 -> imce_1_3

        } // endgenerate
        __builtin_IMCE_SETFLAG(0);
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var1 = __builtin_IMCE_GET_CREG((short)0);
      var2 = __builtin_IMCE_GET_CREG((short)1);
      var3 = __builtin_IMCE_GET_CREG((short)2);
      var4 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_STANDBY(7, 1);
      __builtin_IMCE_SEND(2, var1, 2, 0); // TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
      __builtin_IMCE_SEND(2, var2, 2, 0); // TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
      __builtin_IMCE_SEND(2, var3, 2, 0); // TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
      __builtin_IMCE_SEND(2, var4, 2, 0); // TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
      // endgenerate : conv exec6_row_group0_col_group0
      // endgenerate: conv exec6_row_group0_col_group0
      // generate: conv exec6_row_group0_col_group1
      for (int i2 = 0; i2 < 7; i2++) { // generate : conv exec6_row_group0_col_group1

        // generate: load_block
        for (int i3 = 0; i3 < 2; i3++) { // generate : load_block
          __builtin_IMCE_SETFLAG(3);
          for (int i4 = 0; i4 < 4; i4++) { // generate
            __builtin_IMCE_LOAD_LB(0); // TensorEdge((86, odata), (87, data), 1), imce_0_3 -> imce_1_3

          } // endgenerate
          __builtin_IMCE_SETFLAG(0);
        } // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var1 = __builtin_IMCE_GET_CREG((short)0);
        var2 = __builtin_IMCE_GET_CREG((short)1);
        var3 = __builtin_IMCE_GET_CREG((short)2);
        var4 = __builtin_IMCE_GET_CREG((short)3);
        __builtin_IMCE_STANDBY(7, 1);
        __builtin_IMCE_SEND(2, var1, 2, 0); // TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
        __builtin_IMCE_SEND(2, var2, 2, 0); // TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
        __builtin_IMCE_SEND(2, var3, 2, 0); // TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
        __builtin_IMCE_SEND(2, var4, 2, 0); // TensorEdge((87, odata), ((88, 82), rhs)), imce_1_3 -> imce_1_2
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
    // generate: TensorEdge(((104, -91), fused_scale), ((104, 98), fused_scale)), fused_scale write

    var74 = __builtin_IMCE_RECV(1);
    var75 = __builtin_IMCE_RECV(1);
    var76 = __builtin_IMCE_RECV(1);
    var77 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((104, -91), fused_scale), ((104, 98), fused_scale)), fused_scale write
    // generate: TensorEdge(((104, -92), fused_bias), ((104, 98), fused_bias)), fused_bias write

    var78 = __builtin_IMCE_RECV(1);
    var79 = __builtin_IMCE_RECV(1);
    var80 = __builtin_IMCE_RECV(1);
    var81 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((104, -92), fused_bias), ((104, 98), fused_bias)), fused_bias write
    // generate: mult const

    var82 = __builtin_IMCE_RECV(1);
    var83 = __builtin_IMCE_RECV(1);
    var84 = __builtin_IMCE_RECV(1);
    var85 = __builtin_IMCE_RECV(1);
    // endgenerate: mult const
    // generate: add const

    var86 = __builtin_IMCE_RECV(1);
    var87 = __builtin_IMCE_RECV(1);
    var88 = __builtin_IMCE_RECV(1);
    var89 = __builtin_IMCE_RECV(1);
    // endgenerate: add const
    // generate: call_created_loop
    for (int i1 = 0; i1 < 64; i1++) { // generate : call_created_loop
      // generate: imcflow.vecops_wrapper

      var90 = __builtin_IMCE_RECV(2); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1
      var91 = __builtin_IMCE_RECV(2); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1
      var92 = __builtin_IMCE_RECV(2); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1
      var93 = __builtin_IMCE_RECV(2); // TensorEdge((103, odata), ((104, 98), data)), imce_1_1 -> imce_2_1
      // generate: imcflow.vecops_block
      // generate: batch_norm


      var106 = __builtin_IMCE_MULTL(var90, var74, 15);
      var107 = __builtin_IMCE_MULTH(var90, var74, 15);
      var108 = __builtin_IMCE_SUBI(0, 1);
      var109 = __builtin_IMCE_SRLI(var108, 1);
      var110 = __builtin_IMCE_SRAI(var107, 15);
      var111 = __builtin_IMCE_SRAI(var106, 15);
      var112 = __builtin_IMCE_XOR(var110, var111, 15);
      var113 = __builtin_IMCE_XOR(var110, var109, 15);
      var114 = __builtin_IMCE_XOR(var112, var108, 15);
      var115 = __builtin_IMCE_AND(var112, var113, 15);
      var116 = __builtin_IMCE_AND(var114, var106, 15);
      var99 = __builtin_IMCE_OR(var115, var116, 15);
      var98 = __builtin_IMCE_ADD(var99, var78, 15);

      var117 = __builtin_IMCE_MULTL(var91, var75, 15);
      var118 = __builtin_IMCE_MULTH(var91, var75, 15);
      var119 = __builtin_IMCE_SUBI(0, 1);
      var120 = __builtin_IMCE_SRLI(var119, 1);
      var121 = __builtin_IMCE_SRAI(var118, 15);
      var122 = __builtin_IMCE_SRAI(var117, 15);
      var123 = __builtin_IMCE_XOR(var121, var122, 15);
      var124 = __builtin_IMCE_XOR(var121, var120, 15);
      var125 = __builtin_IMCE_XOR(var123, var119, 15);
      var126 = __builtin_IMCE_AND(var123, var124, 15);
      var127 = __builtin_IMCE_AND(var125, var117, 15);
      var101 = __builtin_IMCE_OR(var126, var127, 15);
      var100 = __builtin_IMCE_ADD(var101, var79, 15);

      var128 = __builtin_IMCE_MULTL(var92, var76, 15);
      var129 = __builtin_IMCE_MULTH(var92, var76, 15);
      var130 = __builtin_IMCE_SUBI(0, 1);
      var131 = __builtin_IMCE_SRLI(var130, 1);
      var132 = __builtin_IMCE_SRAI(var129, 15);
      var133 = __builtin_IMCE_SRAI(var128, 15);
      var134 = __builtin_IMCE_XOR(var132, var133, 15);
      var135 = __builtin_IMCE_XOR(var132, var131, 15);
      var136 = __builtin_IMCE_XOR(var134, var130, 15);
      var137 = __builtin_IMCE_AND(var134, var135, 15);
      var138 = __builtin_IMCE_AND(var136, var128, 15);
      var103 = __builtin_IMCE_OR(var137, var138, 15);
      var102 = __builtin_IMCE_ADD(var103, var80, 15);

      var139 = __builtin_IMCE_MULTL(var93, var77, 15);
      var140 = __builtin_IMCE_MULTH(var93, var77, 15);
      var141 = __builtin_IMCE_SUBI(0, 1);
      var142 = __builtin_IMCE_SRLI(var141, 1);
      var143 = __builtin_IMCE_SRAI(var140, 15);
      var144 = __builtin_IMCE_SRAI(var139, 15);
      var145 = __builtin_IMCE_XOR(var143, var144, 15);
      var146 = __builtin_IMCE_XOR(var143, var142, 15);
      var147 = __builtin_IMCE_XOR(var145, var141, 15);
      var148 = __builtin_IMCE_AND(var145, var146, 15);
      var149 = __builtin_IMCE_AND(var147, var139, 15);
      var105 = __builtin_IMCE_OR(var148, var149, 15);
      var104 = __builtin_IMCE_ADD(var105, var81, 15);
      // endgenerate: batch_norm
      // generate: multl


      var154 = __builtin_IMCE_MULTL(var82, var98, 15);
      var155 = __builtin_IMCE_MULTH(var82, var98, 15);
      var156 = __builtin_IMCE_SUBI(0, 1);
      var157 = __builtin_IMCE_SRLI(var156, 1);
      var158 = __builtin_IMCE_SRAI(var155, 15);
      var159 = __builtin_IMCE_SRAI(var154, 15);
      var160 = __builtin_IMCE_XOR(var158, var159, 15);
      var161 = __builtin_IMCE_XOR(var158, var157, 15);
      var162 = __builtin_IMCE_XOR(var160, var156, 15);
      var163 = __builtin_IMCE_AND(var160, var161, 15);
      var164 = __builtin_IMCE_AND(var162, var154, 15);
      var150 = __builtin_IMCE_OR(var163, var164, 15);

      var165 = __builtin_IMCE_MULTL(var83, var100, 15);
      var166 = __builtin_IMCE_MULTH(var83, var100, 15);
      var167 = __builtin_IMCE_SUBI(0, 1);
      var168 = __builtin_IMCE_SRLI(var167, 1);
      var169 = __builtin_IMCE_SRAI(var166, 15);
      var170 = __builtin_IMCE_SRAI(var165, 15);
      var171 = __builtin_IMCE_XOR(var169, var170, 15);
      var172 = __builtin_IMCE_XOR(var169, var168, 15);
      var173 = __builtin_IMCE_XOR(var171, var167, 15);
      var174 = __builtin_IMCE_AND(var171, var172, 15);
      var175 = __builtin_IMCE_AND(var173, var165, 15);
      var151 = __builtin_IMCE_OR(var174, var175, 15);

      var176 = __builtin_IMCE_MULTL(var84, var102, 15);
      var177 = __builtin_IMCE_MULTH(var84, var102, 15);
      var178 = __builtin_IMCE_SUBI(0, 1);
      var179 = __builtin_IMCE_SRLI(var178, 1);
      var180 = __builtin_IMCE_SRAI(var177, 15);
      var181 = __builtin_IMCE_SRAI(var176, 15);
      var182 = __builtin_IMCE_XOR(var180, var181, 15);
      var183 = __builtin_IMCE_XOR(var180, var179, 15);
      var184 = __builtin_IMCE_XOR(var182, var178, 15);
      var185 = __builtin_IMCE_AND(var182, var183, 15);
      var186 = __builtin_IMCE_AND(var184, var176, 15);
      var152 = __builtin_IMCE_OR(var185, var186, 15);

      var187 = __builtin_IMCE_MULTL(var85, var104, 15);
      var188 = __builtin_IMCE_MULTH(var85, var104, 15);
      var189 = __builtin_IMCE_SUBI(0, 1);
      var190 = __builtin_IMCE_SRLI(var189, 1);
      var191 = __builtin_IMCE_SRAI(var188, 15);
      var192 = __builtin_IMCE_SRAI(var187, 15);
      var193 = __builtin_IMCE_XOR(var191, var192, 15);
      var194 = __builtin_IMCE_XOR(var191, var190, 15);
      var195 = __builtin_IMCE_XOR(var193, var189, 15);
      var196 = __builtin_IMCE_AND(var193, var194, 15);
      var197 = __builtin_IMCE_AND(var195, var187, 15);
      var153 = __builtin_IMCE_OR(var196, var197, 15);
      // endgenerate: multl
      // generate: add

      var94 = __builtin_IMCE_ADD(var86, var150, 15);
      var95 = __builtin_IMCE_ADD(var87, var151, 15);
      var96 = __builtin_IMCE_ADD(var88, var152, 15);
      var97 = __builtin_IMCE_ADD(var89, var153, 15);
      // endgenerate: add
      // endgenerate: imcflow.vecops_block
      __builtin_IMCE_SEND(1, var94, 2, 0); // TensorEdge(((104, 100), odata), (105, func_out1), 1), imce_2_1 -> inode_2_0
      __builtin_IMCE_SEND(1, var95, 2, 0); // TensorEdge(((104, 100), odata), (105, func_out1), 1), imce_2_1 -> inode_2_0
      __builtin_IMCE_SEND(1, var96, 2, 0); // TensorEdge(((104, 100), odata), (105, func_out1), 1), imce_2_1 -> inode_2_0
      __builtin_IMCE_SEND(1, var97, 2, 0); // TensorEdge(((104, 100), odata), (105, func_out1), 1), imce_2_1 -> inode_2_0
      // endgenerate: imcflow.vecops_wrapper
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 2 && wid == 2) { // imce_2_2
    // generate: TensorEdge(((89, -68), fused_scale), ((89, 78), fused_scale)), fused_scale write

    var198 = __builtin_IMCE_RECV(1);
    var199 = __builtin_IMCE_RECV(1);
    var200 = __builtin_IMCE_RECV(1);
    var201 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((89, -68), fused_scale), ((89, 78), fused_scale)), fused_scale write
    // generate: TensorEdge(((89, -69), fused_bias), ((89, 78), fused_bias)), fused_bias write

    var202 = __builtin_IMCE_RECV(1);
    var203 = __builtin_IMCE_RECV(1);
    var204 = __builtin_IMCE_RECV(1);
    var205 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((89, -69), fused_bias), ((89, 78), fused_bias)), fused_bias write
    // generate: TensorEdge(((89, -70), min), ((89, 79), min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge(((89, -70), min), ((89, 79), min)), min write
    // generate: TensorEdge(((89, -71), max), ((89, 79), max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge(((89, -71), max), ((89, 79), max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 64; i1++) { // generate : call_created_loop
      // generate: imcflow.preop-minmax_wrapper

      var208 = __builtin_IMCE_RECV(2); // TensorEdge(((88, 82), odata), ((89, 78), data)), imce_1_2 -> imce_2_2
      var209 = __builtin_IMCE_RECV(2); // TensorEdge(((88, 82), odata), ((89, 78), data)), imce_1_2 -> imce_2_2
      var210 = __builtin_IMCE_RECV(2); // TensorEdge(((88, 82), odata), ((89, 78), data)), imce_1_2 -> imce_2_2
      var211 = __builtin_IMCE_RECV(2); // TensorEdge(((88, 82), odata), ((89, 78), data)), imce_1_2 -> imce_2_2
      // generate: imcflow.preop-minmax_block
      // generate: batch_norm


      var224 = __builtin_IMCE_MULTL(var208, var198, 15);
      var225 = __builtin_IMCE_MULTH(var208, var198, 15);
      var226 = __builtin_IMCE_SUBI(0, 1);
      var227 = __builtin_IMCE_SRLI(var226, 1);
      var228 = __builtin_IMCE_SRAI(var225, 15);
      var229 = __builtin_IMCE_SRAI(var224, 15);
      var230 = __builtin_IMCE_XOR(var228, var229, 15);
      var231 = __builtin_IMCE_XOR(var228, var227, 15);
      var232 = __builtin_IMCE_XOR(var230, var226, 15);
      var233 = __builtin_IMCE_AND(var230, var231, 15);
      var234 = __builtin_IMCE_AND(var232, var224, 15);
      var217 = __builtin_IMCE_OR(var233, var234, 15);
      var216 = __builtin_IMCE_ADD(var217, var202, 15);

      var235 = __builtin_IMCE_MULTL(var209, var199, 15);
      var236 = __builtin_IMCE_MULTH(var209, var199, 15);
      var237 = __builtin_IMCE_SUBI(0, 1);
      var238 = __builtin_IMCE_SRLI(var237, 1);
      var239 = __builtin_IMCE_SRAI(var236, 15);
      var240 = __builtin_IMCE_SRAI(var235, 15);
      var241 = __builtin_IMCE_XOR(var239, var240, 15);
      var242 = __builtin_IMCE_XOR(var239, var238, 15);
      var243 = __builtin_IMCE_XOR(var241, var237, 15);
      var244 = __builtin_IMCE_AND(var241, var242, 15);
      var245 = __builtin_IMCE_AND(var243, var235, 15);
      var219 = __builtin_IMCE_OR(var244, var245, 15);
      var218 = __builtin_IMCE_ADD(var219, var203, 15);

      var246 = __builtin_IMCE_MULTL(var210, var200, 15);
      var247 = __builtin_IMCE_MULTH(var210, var200, 15);
      var248 = __builtin_IMCE_SUBI(0, 1);
      var249 = __builtin_IMCE_SRLI(var248, 1);
      var250 = __builtin_IMCE_SRAI(var247, 15);
      var251 = __builtin_IMCE_SRAI(var246, 15);
      var252 = __builtin_IMCE_XOR(var250, var251, 15);
      var253 = __builtin_IMCE_XOR(var250, var249, 15);
      var254 = __builtin_IMCE_XOR(var252, var248, 15);
      var255 = __builtin_IMCE_AND(var252, var253, 15);
      var256 = __builtin_IMCE_AND(var254, var246, 15);
      var221 = __builtin_IMCE_OR(var255, var256, 15);
      var220 = __builtin_IMCE_ADD(var221, var204, 15);

      var257 = __builtin_IMCE_MULTL(var211, var201, 15);
      var258 = __builtin_IMCE_MULTH(var211, var201, 15);
      var259 = __builtin_IMCE_SUBI(0, 1);
      var260 = __builtin_IMCE_SRLI(var259, 1);
      var261 = __builtin_IMCE_SRAI(var258, 15);
      var262 = __builtin_IMCE_SRAI(var257, 15);
      var263 = __builtin_IMCE_XOR(var261, var262, 15);
      var264 = __builtin_IMCE_XOR(var261, var260, 15);
      var265 = __builtin_IMCE_XOR(var263, var259, 15);
      var266 = __builtin_IMCE_AND(var263, var264, 15);
      var267 = __builtin_IMCE_AND(var265, var257, 15);
      var223 = __builtin_IMCE_OR(var266, var267, 15);
      var222 = __builtin_IMCE_ADD(var223, var205, 15);
      // endgenerate: batch_norm
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var216, 0, 15, 0);
      __builtin_IMCE_MM_QUANT(var218, 0, 15, 1);
      __builtin_IMCE_MM_QUANT(var220, 0, 15, 2);
      __builtin_IMCE_MM_QUANT(var222, 0, 15, 3);
      var212 = __builtin_IMCE_GET_QREG(0);
      var213 = __builtin_IMCE_GET_QREG(1);
      var214 = __builtin_IMCE_GET_QREG(2);
      var215 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      // endgenerate: imcflow.preop-minmax_block
      __builtin_IMCE_STANDBY(13, 1);
      __builtin_IMCE_SEND(1, var212, 0, 0); // TensorEdge(((89, 79), odata), (90, data)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(1, var213, 0, 0); // TensorEdge(((89, 79), odata), (90, data)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(1, var214, 0, 0); // TensorEdge(((89, 79), odata), (90, data)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(1, var215, 0, 0); // TensorEdge(((89, 79), odata), (90, data)), imce_2_2 -> imce_3_2
      // endgenerate: imcflow.preop-minmax_wrapper
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 2 && wid == 3) { // imce_2_3
    // generate: TensorEdge((-85, config), (94, config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge((-85, config), (94, config)), config write
    // generate: conv exec8
    // generate: conv exec8_row_group0_outer_loop(iterate row offset)
    // generate : conv exec8_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec8_row_group0_col_group0
    // generate : conv exec8_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 10; i1++) { // generate : load_block
      __builtin_IMCE_SETFLAG(1);
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((90, odata), (94, data), 1), imce_2_2 -> imce_2_3

      } // endgenerate
      __builtin_IMCE_SETFLAG(0);
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var13 = __builtin_IMCE_GET_CREG((short)0);
    var14 = __builtin_IMCE_GET_CREG((short)1);
    var15 = __builtin_IMCE_GET_CREG((short)2);
    var16 = __builtin_IMCE_GET_CREG((short)3);
    __builtin_IMCE_STANDBY(18, 1);
    __builtin_IMCE_SEND(2, var13, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    __builtin_IMCE_SEND(2, var14, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    __builtin_IMCE_SEND(2, var15, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    __builtin_IMCE_SEND(2, var16, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    // endgenerate : conv exec8_row_group0_col_group0
    // endgenerate: conv exec8_row_group0_col_group0
    // generate: conv exec8_row_group0_col_group1
    for (int i1 = 0; i1 < 6; i1++) { // generate : conv exec8_row_group0_col_group1

      // generate: load_block
      // generate : load_block. loop count == 1
      __builtin_IMCE_SETFLAG(1);
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((90, odata), (94, data), 1), imce_2_2 -> imce_2_3

      } // endgenerate
      __builtin_IMCE_SETFLAG(0);
      // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var13 = __builtin_IMCE_GET_CREG((short)0);
      var14 = __builtin_IMCE_GET_CREG((short)1);
      var15 = __builtin_IMCE_GET_CREG((short)2);
      var16 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_STANDBY(18, 1);
      __builtin_IMCE_SEND(2, var13, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SEND(2, var14, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SEND(2, var15, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SEND(2, var16, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
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
    __builtin_IMCE_STANDBY(18, 1);
    __builtin_IMCE_SEND(2, var13, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    __builtin_IMCE_SEND(2, var14, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    __builtin_IMCE_SEND(2, var15, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    __builtin_IMCE_SEND(2, var16, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
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
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((90, odata), (94, data), 1), imce_2_2 -> imce_2_3

        } // endgenerate
        __builtin_IMCE_SETFLAG(0);
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var13 = __builtin_IMCE_GET_CREG((short)0);
      var14 = __builtin_IMCE_GET_CREG((short)1);
      var15 = __builtin_IMCE_GET_CREG((short)2);
      var16 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_STANDBY(18, 1);
      __builtin_IMCE_SEND(2, var13, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SEND(2, var14, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SEND(2, var15, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SEND(2, var16, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // endgenerate : conv exec8_row_group1_col_group0
      // endgenerate: conv exec8_row_group1_col_group0
      // generate: conv exec8_row_group1_col_group1
      for (int i2 = 0; i2 < 6; i2++) { // generate : conv exec8_row_group1_col_group1

        // generate: load_block
        // generate : load_block. loop count == 1
        __builtin_IMCE_SETFLAG(1);
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((90, odata), (94, data), 1), imce_2_2 -> imce_2_3

        } // endgenerate
        __builtin_IMCE_SETFLAG(0);
        // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var13 = __builtin_IMCE_GET_CREG((short)0);
        var14 = __builtin_IMCE_GET_CREG((short)1);
        var15 = __builtin_IMCE_GET_CREG((short)2);
        var16 = __builtin_IMCE_GET_CREG((short)3);
        __builtin_IMCE_STANDBY(18, 1);
        __builtin_IMCE_SEND(2, var13, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
        __builtin_IMCE_SEND(2, var14, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
        __builtin_IMCE_SEND(2, var15, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
        __builtin_IMCE_SEND(2, var16, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
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
      __builtin_IMCE_STANDBY(18, 1);
      __builtin_IMCE_SEND(2, var13, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SEND(2, var14, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SEND(2, var15, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SEND(2, var16, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
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
      __builtin_IMCE_STANDBY(18, 1);
      __builtin_IMCE_SEND(2, 0, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SEND(2, 0, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SEND(2, 0, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SEND(2, 0, 2, 0); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
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
    // generate: TensorEdge((-86, fused_scale), (97, fused_scale)), fused_scale write

    var273 = __builtin_IMCE_RECV(1);
    var274 = __builtin_IMCE_RECV(1);
    var275 = __builtin_IMCE_RECV(1);
    var276 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge((-86, fused_scale), (97, fused_scale)), fused_scale write
    // generate: TensorEdge((-87, fused_bias), (97, fused_bias)), fused_bias write

    var277 = __builtin_IMCE_RECV(1);
    var278 = __builtin_IMCE_RECV(1);
    var279 = __builtin_IMCE_RECV(1);
    var280 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge((-87, fused_bias), (97, fused_bias)), fused_bias write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 64; i1++) { // generate : call_created_loop
      // generate: bn_standalone

      var281 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
      var282 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
      var283 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
      var284 = __builtin_IMCE_RECV(2); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
      // generate: batch_norm


      var293 = __builtin_IMCE_MULTL(var281, var273, 15);
      var294 = __builtin_IMCE_MULTH(var281, var273, 15);
      var295 = __builtin_IMCE_SUBI(0, 1);
      var296 = __builtin_IMCE_SRLI(var295, 1);
      var297 = __builtin_IMCE_SRAI(var294, 15);
      var298 = __builtin_IMCE_SRAI(var293, 15);
      var299 = __builtin_IMCE_XOR(var297, var298, 15);
      var300 = __builtin_IMCE_XOR(var297, var296, 15);
      var301 = __builtin_IMCE_XOR(var299, var295, 15);
      var302 = __builtin_IMCE_AND(var299, var300, 15);
      var303 = __builtin_IMCE_AND(var301, var293, 15);
      var289 = __builtin_IMCE_OR(var302, var303, 15);
      var285 = __builtin_IMCE_ADD(var289, var277, 15);

      var304 = __builtin_IMCE_MULTL(var282, var274, 15);
      var305 = __builtin_IMCE_MULTH(var282, var274, 15);
      var306 = __builtin_IMCE_SUBI(0, 1);
      var307 = __builtin_IMCE_SRLI(var306, 1);
      var308 = __builtin_IMCE_SRAI(var305, 15);
      var309 = __builtin_IMCE_SRAI(var304, 15);
      var310 = __builtin_IMCE_XOR(var308, var309, 15);
      var311 = __builtin_IMCE_XOR(var308, var307, 15);
      var312 = __builtin_IMCE_XOR(var310, var306, 15);
      var313 = __builtin_IMCE_AND(var310, var311, 15);
      var314 = __builtin_IMCE_AND(var312, var304, 15);
      var290 = __builtin_IMCE_OR(var313, var314, 15);
      var286 = __builtin_IMCE_ADD(var290, var278, 15);

      var315 = __builtin_IMCE_MULTL(var283, var275, 15);
      var316 = __builtin_IMCE_MULTH(var283, var275, 15);
      var317 = __builtin_IMCE_SUBI(0, 1);
      var318 = __builtin_IMCE_SRLI(var317, 1);
      var319 = __builtin_IMCE_SRAI(var316, 15);
      var320 = __builtin_IMCE_SRAI(var315, 15);
      var321 = __builtin_IMCE_XOR(var319, var320, 15);
      var322 = __builtin_IMCE_XOR(var319, var318, 15);
      var323 = __builtin_IMCE_XOR(var321, var317, 15);
      var324 = __builtin_IMCE_AND(var321, var322, 15);
      var325 = __builtin_IMCE_AND(var323, var315, 15);
      var291 = __builtin_IMCE_OR(var324, var325, 15);
      var287 = __builtin_IMCE_ADD(var291, var279, 15);

      var326 = __builtin_IMCE_MULTL(var284, var276, 15);
      var327 = __builtin_IMCE_MULTH(var284, var276, 15);
      var328 = __builtin_IMCE_SUBI(0, 1);
      var329 = __builtin_IMCE_SRLI(var328, 1);
      var330 = __builtin_IMCE_SRAI(var327, 15);
      var331 = __builtin_IMCE_SRAI(var326, 15);
      var332 = __builtin_IMCE_XOR(var330, var331, 15);
      var333 = __builtin_IMCE_XOR(var330, var329, 15);
      var334 = __builtin_IMCE_XOR(var332, var328, 15);
      var335 = __builtin_IMCE_AND(var332, var333, 15);
      var336 = __builtin_IMCE_AND(var334, var326, 15);
      var292 = __builtin_IMCE_OR(var335, var336, 15);
      var288 = __builtin_IMCE_ADD(var292, var280, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var285, 2, 0); // TensorEdge((97, odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var286, 2, 0); // TensorEdge((97, odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var287, 2, 0); // TensorEdge((97, odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var288, 2, 0); // TensorEdge((97, odata), (105, func_out0), 0), imce_3_1 -> inode_3_0
      // endgenerate: bn_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 2) { // imce_3_2
    // generate: TensorEdge(((96, -66), config), ((96, 75), config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge(((96, -66), config), ((96, 75), config)), config write
    // generate: conv exec10
    // generate: conv exec10_row_group0_outer_loop(iterate row offset)
    // generate : conv exec10_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec10_row_group0_col_group0
    // generate : conv exec10_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 10; i1++) { // generate : load_block
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((90, odata), ((96, 75), data), 2), imce_2_2 -> imce_3_2

      } // endgenerate
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var25 = __builtin_IMCE_GET_CREG((short)0);
    var26 = __builtin_IMCE_GET_CREG((short)1);
    var27 = __builtin_IMCE_GET_CREG((short)2);
    var28 = __builtin_IMCE_GET_CREG((short)3);
    // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    __builtin_IMCE_SETFLAG(1);
    var29 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    var30 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    var31 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    var32 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    __builtin_IMCE_SETFLAG(0);
    // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    // generate: add

    var338 = __builtin_IMCE_ADD(var29, var25, 15);
    var339 = __builtin_IMCE_ADD(var30, var26, 15);
    var340 = __builtin_IMCE_ADD(var31, var27, 15);
    var341 = __builtin_IMCE_ADD(var32, var28, 15);
    // endgenerate: add
    __builtin_IMCE_SEND(1, var338, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(1, var339, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(1, var340, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(1, var341, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
    // endgenerate : conv exec10_row_group0_col_group0
    // endgenerate: conv exec10_row_group0_col_group0
    // generate: conv exec10_row_group0_col_group1
    //! we split 6 loop into 5 and 1 loop because next is padding. we should optimize timing
    for (int i1 = 0; i1 < 5; i1++) { // generate : conv exec10_row_group0_col_group1

      // generate: load_block
      // generate : load_block. loop count == 1
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((90, odata), ((96, 75), data), 2), imce_2_2 -> imce_3_2

      } // endgenerate
      // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var25 = __builtin_IMCE_GET_CREG((short)0);
      var26 = __builtin_IMCE_GET_CREG((short)1);
      var27 = __builtin_IMCE_GET_CREG((short)2);
      var28 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SETFLAG(1);
      var29 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      var30 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      var31 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      var32 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SETFLAG(0);
      // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // generate: add

      var338 = __builtin_IMCE_ADD(var29, var25, 15);
      var339 = __builtin_IMCE_ADD(var30, var26, 15);
      var340 = __builtin_IMCE_ADD(var31, var27, 15);
      var341 = __builtin_IMCE_ADD(var32, var28, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(1, var338, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var339, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var340, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var341, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
    } // endgenerate : conv exec10_row_group0_col_group1
    // endgenerate: conv exec10_row_group0_col_group1
    // generate: conv exec10_row_group0_col_group2
    // generate : conv exec10_row_group0_col_group2. loop count == 1

    //! this is last iteration before padding
    //! just do 3 load_lb
    for (int i2 = 0; i2 < 3; i2++) { // generate
      __builtin_IMCE_LOAD_LB(0); // TensorEdge((90, odata), ((96, 75), data), 2), imce_2_2 -> imce_3_2
    } // endgenerate

    //! we wait IMCE node that send psum into this node
    __builtin_IMCE_SETFLAG(1);
    __builtin_IMCE_STANDBY(18, 5);
    __builtin_IMCE_LOAD_LB(0); // TensorEdge((90, odata), ((96, 75), data), 2), imce_2_2 -> imce_3_2
    // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var25 = __builtin_IMCE_GET_CREG((short)0);
    var26 = __builtin_IMCE_GET_CREG((short)1);
    var27 = __builtin_IMCE_GET_CREG((short)2);
    var28 = __builtin_IMCE_GET_CREG((short)3);
    //! move SET_FLAG(1) above due to approve imce 3.3 send psum fast
    // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    var29 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    var30 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    var31 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    var32 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    __builtin_IMCE_SETFLAG(0);
    // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    // generate: add

    var338 = __builtin_IMCE_ADD(var29, var25, 15);
    var339 = __builtin_IMCE_ADD(var30, var26, 15);
    var340 = __builtin_IMCE_ADD(var31, var27, 15);
    var341 = __builtin_IMCE_ADD(var32, var28, 15);
    // endgenerate: add
    __builtin_IMCE_SEND(1, var338, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(1, var339, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(1, var340, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(1, var341, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1

    // generate: load_block
    // loop ignored with loop count == 0 : load_block
    // endgenerate: load_block
    //! problem point!!. step_hs arrived here before this inst -> stall
    // PC addr = b4
    __builtin_IMCE_STEP();


    var25 = __builtin_IMCE_GET_CREG((short)0);
    var26 = __builtin_IMCE_GET_CREG((short)1);
    var27 = __builtin_IMCE_GET_CREG((short)2);
    var28 = __builtin_IMCE_GET_CREG((short)3);
    // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    __builtin_IMCE_SETFLAG(1);
    var29 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    var30 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    var31 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    var32 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    __builtin_IMCE_SETFLAG(0);
    // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    // generate: add

    var338 = __builtin_IMCE_ADD(var29, var25, 15);
    var339 = __builtin_IMCE_ADD(var30, var26, 15);
    var340 = __builtin_IMCE_ADD(var31, var27, 15);
    var341 = __builtin_IMCE_ADD(var32, var28, 15);
    // endgenerate: add
    __builtin_IMCE_SEND(1, var338, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(1, var339, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(1, var340, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(1, var341, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
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
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((90, odata), ((96, 75), data), 2), imce_2_2 -> imce_3_2

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var25 = __builtin_IMCE_GET_CREG((short)0);
      var26 = __builtin_IMCE_GET_CREG((short)1);
      var27 = __builtin_IMCE_GET_CREG((short)2);
      var28 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SETFLAG(1);
      var29 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      var30 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      var31 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      var32 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SETFLAG(0);
      // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // generate: add

      var338 = __builtin_IMCE_ADD(var29, var25, 15);
      var339 = __builtin_IMCE_ADD(var30, var26, 15);
      var340 = __builtin_IMCE_ADD(var31, var27, 15);
      var341 = __builtin_IMCE_ADD(var32, var28, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(1, var338, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var339, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var340, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var341, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
      // endgenerate : conv exec10_row_group1_col_group0
      // endgenerate: conv exec10_row_group1_col_group0
      // generate: conv exec10_row_group1_col_group1
      for (int i2 = 0; i2 < 6; i2++) { // generate : conv exec10_row_group1_col_group1

        // generate: load_block
        // generate : load_block. loop count == 1
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((90, odata), ((96, 75), data), 2), imce_2_2 -> imce_3_2

        } // endgenerate
        // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var25 = __builtin_IMCE_GET_CREG((short)0);
        var26 = __builtin_IMCE_GET_CREG((short)1);
        var27 = __builtin_IMCE_GET_CREG((short)2);
        var28 = __builtin_IMCE_GET_CREG((short)3);
        // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
        __builtin_IMCE_SETFLAG(1);
        var29 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
        // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
        // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
        var30 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
        // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
        // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
        var31 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
        // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
        // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
        var32 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
        __builtin_IMCE_SETFLAG(0);
        // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
        // generate: add

        var338 = __builtin_IMCE_ADD(var29, var25, 15);
        var339 = __builtin_IMCE_ADD(var30, var26, 15);
        var340 = __builtin_IMCE_ADD(var31, var27, 15);
        var341 = __builtin_IMCE_ADD(var32, var28, 15);
        // endgenerate: add
        __builtin_IMCE_SEND(1, var338, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
        __builtin_IMCE_SEND(1, var339, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
        __builtin_IMCE_SEND(1, var340, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
        __builtin_IMCE_SEND(1, var341, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
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
      // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SETFLAG(1);
      var29 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      var30 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      var31 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      var32 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SETFLAG(0);
      // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // generate: add

      var338 = __builtin_IMCE_ADD(var29, var25, 15);
      var339 = __builtin_IMCE_ADD(var30, var26, 15);
      var340 = __builtin_IMCE_ADD(var31, var27, 15);
      var341 = __builtin_IMCE_ADD(var32, var28, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(1, var338, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var339, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var340, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var341, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
      // endgenerate : conv exec10_row_group1_col_group2
      // endgenerate: conv exec10_row_group1_col_group2
    } // endgenerate : conv exec10_row_group1_outer_loop(iterate row offset)
    // endgenerate: conv exec10_row_group1_outer_loop(iterate row offset)
    // generate: conv exec10_row_group2_outer_loop(iterate row offset)
    // generate : conv exec10_row_group2_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec10_row_group2_col_group0
    for (int i1 = 0; i1 < 8; i1++) { // generate : conv exec10_row_group2_col_group0
      // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SETFLAG(1);
      var29 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      var30 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      var31 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // generate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      var32 = __builtin_IMCE_RECV(2); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SETFLAG(0);
      // endgenerate: TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // generate: add

      var338 = __builtin_IMCE_ADD(var29, 0, 15);
      var339 = __builtin_IMCE_ADD(var30, 0, 15);
      var340 = __builtin_IMCE_ADD(var31, 0, 15);
      var341 = __builtin_IMCE_ADD(var32, 0, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(1, var338, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var339, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var340, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var341, 2, 0); // TensorEdge(((96, 76), odata), (97, data)), imce_3_2 -> imce_3_1
    } // endgenerate : conv exec10_row_group2_col_group0
    // endgenerate: conv exec10_row_group2_col_group0
    // endgenerate : conv exec10_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec10_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec10
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 3) { // imce_3_3
    // generate: TensorEdge(((95, -83), config), ((95, 91), config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge(((95, -83), config), ((95, 91), config)), config write
    // generate: conv exec9
    // generate: conv exec9_row_group0_outer_loop(iterate row offset)
    // generate : conv exec9_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec9_row_group0_col_group0
    // generate : conv exec9_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 10; i1++) { // generate : load_block
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((90, odata), ((95, 91), data), 0), imce_2_2 -> imce_3_2, imce_3_3

      } // endgenerate
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var17 = __builtin_IMCE_GET_CREG((short)0);
    var18 = __builtin_IMCE_GET_CREG((short)1);
    var19 = __builtin_IMCE_GET_CREG((short)2);
    var20 = __builtin_IMCE_GET_CREG((short)3);
    // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    __builtin_IMCE_SETFLAG(1);
    var21 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    var22 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    var23 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    var24 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    __builtin_IMCE_SETFLAG(0);
    // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    // generate: add

    var343 = __builtin_IMCE_ADD(var17, var21, 15);
    var344 = __builtin_IMCE_ADD(var18, var22, 15);
    var345 = __builtin_IMCE_ADD(var19, var23, 15);
    var346 = __builtin_IMCE_ADD(var20, var24, 15);
    // endgenerate: add
    __builtin_IMCE_STANDBY(17, 1);
    __builtin_IMCE_SEND(2, var343, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    __builtin_IMCE_SEND(2, var344, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    __builtin_IMCE_SEND(2, var345, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    __builtin_IMCE_SEND(2, var346, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    // endgenerate : conv exec9_row_group0_col_group0
    // endgenerate: conv exec9_row_group0_col_group0
    // generate: conv exec9_row_group0_col_group1
    for (int i1 = 0; i1 < 6; i1++) { // generate : conv exec9_row_group0_col_group1

      // generate: load_block
      // generate : load_block. loop count == 1
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((90, odata), ((95, 91), data), 0), imce_2_2 -> imce_3_2, imce_3_3

      } // endgenerate
      // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var17 = __builtin_IMCE_GET_CREG((short)0);
      var18 = __builtin_IMCE_GET_CREG((short)1);
      var19 = __builtin_IMCE_GET_CREG((short)2);
      var20 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SETFLAG(1);
      var21 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      var22 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      var23 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      var24 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SETFLAG(0);
      // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // generate: add

      var343 = __builtin_IMCE_ADD(var17, var21, 15);
      var344 = __builtin_IMCE_ADD(var18, var22, 15);
      var345 = __builtin_IMCE_ADD(var19, var23, 15);
      var346 = __builtin_IMCE_ADD(var20, var24, 15);
      // endgenerate: add
      __builtin_IMCE_STANDBY(17, 1);
      __builtin_IMCE_SEND(2, var343, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(2, var344, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(2, var345, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(2, var346, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    } // endgenerate : conv exec9_row_group0_col_group1
    // endgenerate: conv exec9_row_group0_col_group1
    // generate: conv exec9_row_group0_col_group2
    // generate : conv exec9_row_group0_col_group2. loop count == 1

    //! here we notify psum send event to IMCE 3.2
    __builtin_IMCE_SETFLAG(5);

    // generate: load_block
    // loop ignored with loop count == 0 : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var17 = __builtin_IMCE_GET_CREG((short)0);
    var18 = __builtin_IMCE_GET_CREG((short)1);
    var19 = __builtin_IMCE_GET_CREG((short)2);
    var20 = __builtin_IMCE_GET_CREG((short)3);
    // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    __builtin_IMCE_SETFLAG(1);
    var21 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    var22 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    var23 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    var24 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    __builtin_IMCE_SETFLAG(0);
    // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
    // generate: add

    var343 = __builtin_IMCE_ADD(var17, var21, 15);
    var344 = __builtin_IMCE_ADD(var18, var22, 15);
    var345 = __builtin_IMCE_ADD(var19, var23, 15);
    var346 = __builtin_IMCE_ADD(var20, var24, 15);
    // endgenerate: add
    __builtin_IMCE_STANDBY(17, 1);
    __builtin_IMCE_SEND(2, var343, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    __builtin_IMCE_SEND(2, var344, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    __builtin_IMCE_SEND(2, var345, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    __builtin_IMCE_SEND(2, var346, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
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
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((90, odata), ((95, 91), data), 0), imce_2_2 -> imce_3_2, imce_3_3

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var17 = __builtin_IMCE_GET_CREG((short)0);
      var18 = __builtin_IMCE_GET_CREG((short)1);
      var19 = __builtin_IMCE_GET_CREG((short)2);
      var20 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SETFLAG(1);
      var21 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      var22 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      var23 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      var24 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SETFLAG(0);
      // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // generate: add

      var343 = __builtin_IMCE_ADD(var17, var21, 15);
      var344 = __builtin_IMCE_ADD(var18, var22, 15);
      var345 = __builtin_IMCE_ADD(var19, var23, 15);
      var346 = __builtin_IMCE_ADD(var20, var24, 15);
      // endgenerate: add
      __builtin_IMCE_STANDBY(17, 1);
      __builtin_IMCE_SEND(2, var343, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(2, var344, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(2, var345, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(2, var346, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // endgenerate : conv exec9_row_group1_col_group0
      // endgenerate: conv exec9_row_group1_col_group0
      // generate: conv exec9_row_group1_col_group1
      for (int i2 = 0; i2 < 6; i2++) { // generate : conv exec9_row_group1_col_group1

        // generate: load_block
        // generate : load_block. loop count == 1
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((90, odata), ((95, 91), data), 0), imce_2_2 -> imce_3_2, imce_3_3

        } // endgenerate
        // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var17 = __builtin_IMCE_GET_CREG((short)0);
        var18 = __builtin_IMCE_GET_CREG((short)1);
        var19 = __builtin_IMCE_GET_CREG((short)2);
        var20 = __builtin_IMCE_GET_CREG((short)3);
        // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
        __builtin_IMCE_SETFLAG(1);
        var21 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
        // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
        // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
        var22 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
        // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
        // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
        var23 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
        // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
        // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
        var24 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
        __builtin_IMCE_SETFLAG(0);
        // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
        // generate: add

        var343 = __builtin_IMCE_ADD(var17, var21, 15);
        var344 = __builtin_IMCE_ADD(var18, var22, 15);
        var345 = __builtin_IMCE_ADD(var19, var23, 15);
        var346 = __builtin_IMCE_ADD(var20, var24, 15);
        // endgenerate: add
        __builtin_IMCE_STANDBY(17, 1);
        __builtin_IMCE_SEND(2, var343, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
        __builtin_IMCE_SEND(2, var344, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
        __builtin_IMCE_SEND(2, var345, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
        __builtin_IMCE_SEND(2, var346, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
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
      // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SETFLAG(1);
      var21 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      var22 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      var23 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      var24 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SETFLAG(0);
      // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // generate: add

      var343 = __builtin_IMCE_ADD(var17, var21, 15);
      var344 = __builtin_IMCE_ADD(var18, var22, 15);
      var345 = __builtin_IMCE_ADD(var19, var23, 15);
      var346 = __builtin_IMCE_ADD(var20, var24, 15);
      // endgenerate: add
      __builtin_IMCE_STANDBY(17, 1);
      __builtin_IMCE_SEND(2, var343, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(2, var344, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(2, var345, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(2, var346, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      // endgenerate : conv exec9_row_group1_col_group2
      // endgenerate: conv exec9_row_group1_col_group2
    } // endgenerate : conv exec9_row_group1_outer_loop(iterate row offset)
    // endgenerate: conv exec9_row_group1_outer_loop(iterate row offset)
    // generate: conv exec9_row_group2_outer_loop(iterate row offset)
    // generate : conv exec9_row_group2_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec9_row_group2_col_group0
    for (int i1 = 0; i1 < 8; i1++) { // generate : conv exec9_row_group2_col_group0
      // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SETFLAG(1);
      var21 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      var22 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      var23 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // generate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      var24 = __builtin_IMCE_RECV(2); // TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SETFLAG(0);
      // endgenerate: TensorEdge((94, odata), ((95, 92), rhs)), imce_2_3 -> imce_3_3
      // generate: add

      var343 = __builtin_IMCE_ADD(0, var21, 15);
      var344 = __builtin_IMCE_ADD(0, var22, 15);
      var345 = __builtin_IMCE_ADD(0, var23, 15);
      var346 = __builtin_IMCE_ADD(0, var24, 15);
      // endgenerate: add
      __builtin_IMCE_STANDBY(17, 1);
      __builtin_IMCE_SEND(2, var343, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(2, var344, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(2, var345, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
      __builtin_IMCE_SEND(2, var346, 2, 0); // TensorEdge(((95, 92), odata), ((96, 76), lhs)), imce_3_3 -> imce_3_2
    } // endgenerate : conv exec9_row_group2_col_group0
    // endgenerate: conv exec9_row_group2_col_group0
    // endgenerate : conv exec9_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec9_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec9
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 4) { // imce_3_4
  }
}
