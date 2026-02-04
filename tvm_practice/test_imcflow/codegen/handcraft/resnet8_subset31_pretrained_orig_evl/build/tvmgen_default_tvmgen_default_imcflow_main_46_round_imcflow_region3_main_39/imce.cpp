#include "../common_decl.h"
void tvmgen_default_tvmgen_default_imcflow_main_46_round_imcflow_region3_main_39() {
  int hid = __builtin_IMCE_GET_CORE_HID();
  int wid = __builtin_IMCE_GET_CORE_WID();
  short16 var1; // (ConvBlock(gid: 91), 0)
  short16 var2; // (ConvBlock(gid: 91), 1)
  short16 var3; // (ConvBlock(gid: 91), 2)
  short16 var4; // (ConvBlock(gid: 91), 3)
  short16 var5; // (ConvBlock(gid: 83), 0)
  short16 var6; // (ConvBlock(gid: 83), 1)
  short16 var7; // (ConvBlock(gid: 83), 2)
  short16 var8; // (ConvBlock(gid: 83), 3)
  short16 var9; // (TensorEdge((91, odata), ((92, 84), rhs)), 0)
  short16 var10; // (TensorEdge((91, odata), ((92, 84), rhs)), 1)
  short16 var11; // (TensorEdge((91, odata), ((92, 84), rhs)), 2)
  short16 var12; // (TensorEdge((91, odata), ((92, 84), rhs)), 3)
  short16 var13; // (ConvBlock(gid: 98), 0)
  short16 var14; // (ConvBlock(gid: 98), 1)
  short16 var15; // (ConvBlock(gid: 98), 2)
  short16 var16; // (ConvBlock(gid: 98), 3)
  short16 var17; // (ConvBlock(gid: 95), 0)
  short16 var18; // (ConvBlock(gid: 95), 1)
  short16 var19; // (ConvBlock(gid: 95), 2)
  short16 var20; // (ConvBlock(gid: 95), 3)
  short16 var21; // (TensorEdge((98, odata), ((99, 96), rhs)), 0)
  short16 var22; // (TensorEdge((98, odata), ((99, 96), rhs)), 1)
  short16 var23; // (TensorEdge((98, odata), ((99, 96), rhs)), 2)
  short16 var24; // (TensorEdge((98, odata), ((99, 96), rhs)), 3)
  short16 var25; // (ConvBlock(gid: 77), 0)
  short16 var26; // (ConvBlock(gid: 77), 1)
  short16 var27; // (ConvBlock(gid: 77), 2)
  short16 var28; // (ConvBlock(gid: 77), 3)
  short16 var29; // (TensorEdge(((99, 96), odata), ((100, 78), lhs)), 0)
  short16 var30; // (TensorEdge(((99, 96), odata), ((100, 78), lhs)), 1)
  short16 var31; // (TensorEdge(((99, 96), odata), ((100, 78), lhs)), 2)
  short16 var32; // (TensorEdge(((99, 96), odata), ((100, 78), lhs)), 3)
  short16 var33; // (ConvBlock(gid: 105), 0)
  short16 var34; // (ConvBlock(gid: 105), 1)
  short16 var35; // (ConvBlock(gid: 105), 2)
  short16 var36; // (ConvBlock(gid: 105), 3)
  short16 var37; // (TensorEdge((-83, config), (91, config)), 0)
  short16 var38; // (TensorEdge((90, odata), (91, data), 1), 0)
  short16 var39; // (TensorEdge((90, odata), (91, data), 1), 1)
  short16 var40; // (TensorEdge((90, odata), (91, data), 1), 2)
  short16 var41; // (TensorEdge((90, odata), (91, data), 1), 3)
  short16 var42; // (TensorEdge((-80, min), (89, min)), 0)
  short16 var43; // (TensorEdge((-81, max), (89, max)), 0)
  short16 var44; // (TensorEdge(((88, 86), odata), (89, data)), 0)
  short16 var45; // (TensorEdge(((88, 86), odata), (89, data)), 1)
  short16 var46; // (MinmaxQuantBlock(gid: 89), 0)
  short16 var47; // (MinmaxQuantBlock(gid: 89), 1)
  short16 var48; // (MinmaxQuantBlock(gid: 89), 2)
  short16 var49; // (MinmaxQuantBlock(gid: 89), 3)
  short16 var50; // (TensorEdge((-63, odata), ((88, 86), lhs)), 0)
  short16 var51; // (TensorEdge((-64, odata), ((88, 86), rhs)), 0)
  short16 var52; // (AddBlock(gid: 86), 0)
  short16 var53; // (TensorEdge(((92, -77), config), ((92, 83), config)), 0)
  short16 var54; // (AddBlock(gid: 84), 0)
  short16 var55; // (AddBlock(gid: 84), 1)
  short16 var56; // (AddBlock(gid: 84), 2)
  short16 var57; // (AddBlock(gid: 84), 3)
  short16 var58; // (TensorEdge((-89, config), (98, config)), 0)
  short16 var59; // (TensorEdge((94, odata), (98, data), 1), 0)
  short16 var60; // (TensorEdge((94, odata), (98, data), 1), 1)
  short16 var61; // (TensorEdge((94, odata), (98, data), 1), 2)
  short16 var62; // (TensorEdge((94, odata), (98, data), 1), 3)
  short16 var63; // (TensorEdge((-95, min), (104, min)), 0)
  short16 var64; // (TensorEdge((-96, max), (104, max)), 0)
  short16 var65; // (TensorEdge(((88, 86), odata), (104, data)), 0)
  short16 var66; // (TensorEdge(((88, 86), odata), (104, data)), 1)
  short16 var67; // (MinmaxQuantBlock(gid: 104), 0)
  short16 var68; // (MinmaxQuantBlock(gid: 104), 1)
  short16 var69; // (MinmaxQuantBlock(gid: 104), 2)
  short16 var70; // (MinmaxQuantBlock(gid: 104), 3)
  short16 var71; // (TensorEdge(((93, -70), fused_scale), ((93, 80), fused_scale)), 0)
  short16 var72; // (TensorEdge(((93, -70), fused_scale), ((93, 80), fused_scale)), 1)
  short16 var73; // (TensorEdge(((93, -70), fused_scale), ((93, 80), fused_scale)), 2)
  short16 var74; // (TensorEdge(((93, -70), fused_scale), ((93, 80), fused_scale)), 3)
  short16 var75; // (TensorEdge(((93, -71), fused_bias), ((93, 80), fused_bias)), 0)
  short16 var76; // (TensorEdge(((93, -71), fused_bias), ((93, 80), fused_bias)), 1)
  short16 var77; // (TensorEdge(((93, -71), fused_bias), ((93, 80), fused_bias)), 2)
  short16 var78; // (TensorEdge(((93, -71), fused_bias), ((93, 80), fused_bias)), 3)
  short16 var79; // (TensorEdge(((93, -72), min), ((93, 81), min)), 0)
  short16 var80; // (TensorEdge(((93, -73), max), ((93, 81), max)), 0)
  short16 var81; // (TensorEdge(((92, 84), odata), ((93, 80), data)), 0)
  short16 var82; // (TensorEdge(((92, 84), odata), ((93, 80), data)), 1)
  short16 var83; // (TensorEdge(((92, 84), odata), ((93, 80), data)), 2)
  short16 var84; // (TensorEdge(((92, 84), odata), ((93, 80), data)), 3)
  short16 var85; // (MinmaxQuantBlock(gid: 81), 0)
  short16 var86; // (MinmaxQuantBlock(gid: 81), 1)
  short16 var87; // (MinmaxQuantBlock(gid: 81), 2)
  short16 var88; // (MinmaxQuantBlock(gid: 81), 3)
  short16 var89; // (BatchNormBlock(gid: 80), 0)
  short16 var90; // (BatchNormBlock(gid: 80), 0, 'mult_result')
  short16 var91; // (BatchNormBlock(gid: 80), 1)
  short16 var92; // (BatchNormBlock(gid: 80), 1, 'mult_result')
  short16 var93; // (BatchNormBlock(gid: 80), 2)
  short16 var94; // (BatchNormBlock(gid: 80), 2, 'mult_result')
  short16 var95; // (BatchNormBlock(gid: 80), 3)
  short16 var96; // (BatchNormBlock(gid: 80), 3, 'mult_result')
  short16 var97; // ((BatchNormBlock(gid: 80), 0), 'L')
  short16 var98; // ((BatchNormBlock(gid: 80), 0), 'H')
  short16 var99; // ((BatchNormBlock(gid: 80), 0), 'neg1')
  short16 var100; // ((BatchNormBlock(gid: 80), 0), 'const_7fff')
  short16 var101; // ((BatchNormBlock(gid: 80), 0), 'H_sign')
  short16 var102; // ((BatchNormBlock(gid: 80), 0), 'L_sign')
  short16 var103; // ((BatchNormBlock(gid: 80), 0), 'mismatch')
  short16 var104; // ((BatchNormBlock(gid: 80), 0), 'saturate_val')
  short16 var105; // ((BatchNormBlock(gid: 80), 0), 'not_mismatch')
  short16 var106; // ((BatchNormBlock(gid: 80), 0), 'part1')
  short16 var107; // ((BatchNormBlock(gid: 80), 0), 'part2')
  short16 var108; // ((BatchNormBlock(gid: 80), 1), 'L')
  short16 var109; // ((BatchNormBlock(gid: 80), 1), 'H')
  short16 var110; // ((BatchNormBlock(gid: 80), 1), 'neg1')
  short16 var111; // ((BatchNormBlock(gid: 80), 1), 'const_7fff')
  short16 var112; // ((BatchNormBlock(gid: 80), 1), 'H_sign')
  short16 var113; // ((BatchNormBlock(gid: 80), 1), 'L_sign')
  short16 var114; // ((BatchNormBlock(gid: 80), 1), 'mismatch')
  short16 var115; // ((BatchNormBlock(gid: 80), 1), 'saturate_val')
  short16 var116; // ((BatchNormBlock(gid: 80), 1), 'not_mismatch')
  short16 var117; // ((BatchNormBlock(gid: 80), 1), 'part1')
  short16 var118; // ((BatchNormBlock(gid: 80), 1), 'part2')
  short16 var119; // ((BatchNormBlock(gid: 80), 2), 'L')
  short16 var120; // ((BatchNormBlock(gid: 80), 2), 'H')
  short16 var121; // ((BatchNormBlock(gid: 80), 2), 'neg1')
  short16 var122; // ((BatchNormBlock(gid: 80), 2), 'const_7fff')
  short16 var123; // ((BatchNormBlock(gid: 80), 2), 'H_sign')
  short16 var124; // ((BatchNormBlock(gid: 80), 2), 'L_sign')
  short16 var125; // ((BatchNormBlock(gid: 80), 2), 'mismatch')
  short16 var126; // ((BatchNormBlock(gid: 80), 2), 'saturate_val')
  short16 var127; // ((BatchNormBlock(gid: 80), 2), 'not_mismatch')
  short16 var128; // ((BatchNormBlock(gid: 80), 2), 'part1')
  short16 var129; // ((BatchNormBlock(gid: 80), 2), 'part2')
  short16 var130; // ((BatchNormBlock(gid: 80), 3), 'L')
  short16 var131; // ((BatchNormBlock(gid: 80), 3), 'H')
  short16 var132; // ((BatchNormBlock(gid: 80), 3), 'neg1')
  short16 var133; // ((BatchNormBlock(gid: 80), 3), 'const_7fff')
  short16 var134; // ((BatchNormBlock(gid: 80), 3), 'H_sign')
  short16 var135; // ((BatchNormBlock(gid: 80), 3), 'L_sign')
  short16 var136; // ((BatchNormBlock(gid: 80), 3), 'mismatch')
  short16 var137; // ((BatchNormBlock(gid: 80), 3), 'saturate_val')
  short16 var138; // ((BatchNormBlock(gid: 80), 3), 'not_mismatch')
  short16 var139; // ((BatchNormBlock(gid: 80), 3), 'part1')
  short16 var140; // ((BatchNormBlock(gid: 80), 3), 'part2')
  short16 var141; // (TensorEdge(((99, -87), config), ((99, 95), config)), 0)
  short16 var142; // (AddBlock(gid: 96), 0)
  short16 var143; // (AddBlock(gid: 96), 1)
  short16 var144; // (AddBlock(gid: 96), 2)
  short16 var145; // (AddBlock(gid: 96), 3)
  short16 var146; // (TensorEdge((-98, config), (105, config)), 0)
  short16 var147; // (TensorEdge((104, odata), (105, data)), 0)
  short16 var148; // (TensorEdge((104, odata), (105, data)), 1)
  short16 var149; // (TensorEdge((104, odata), (105, data)), 2)
  short16 var150; // (TensorEdge((104, odata), (105, data)), 3)
  short16 var151; // (TensorEdge((-101, scale), ((108, 102), rhs)), 0)
  short16 var152; // (TensorEdge((-101, scale), ((108, 102), rhs)), 1)
  short16 var153; // (TensorEdge((-101, scale), ((108, 102), rhs)), 2)
  short16 var154; // (TensorEdge((-101, scale), ((108, 102), rhs)), 3)
  short16 var155; // (TensorEdge((107, odata), ((108, 102), lhs)), 0)
  short16 var156; // (AddBlock(gid: 102), 0)
  short16 var157; // (TensorEdge((-90, fused_scale), (101, fused_scale)), 0)
  short16 var158; // (TensorEdge((-90, fused_scale), (101, fused_scale)), 1)
  short16 var159; // (TensorEdge((-90, fused_scale), (101, fused_scale)), 2)
  short16 var160; // (TensorEdge((-90, fused_scale), (101, fused_scale)), 3)
  short16 var161; // (TensorEdge((-91, fused_bias), (101, fused_bias)), 0)
  short16 var162; // (TensorEdge((-91, fused_bias), (101, fused_bias)), 1)
  short16 var163; // (TensorEdge((-91, fused_bias), (101, fused_bias)), 2)
  short16 var164; // (TensorEdge((-91, fused_bias), (101, fused_bias)), 3)
  short16 var165; // (TensorEdge(((100, 78), odata), (101, data)), 0)
  short16 var166; // (TensorEdge(((100, 78), odata), (101, data)), 1)
  short16 var167; // (TensorEdge(((100, 78), odata), (101, data)), 2)
  short16 var168; // (TensorEdge(((100, 78), odata), (101, data)), 3)
  short16 var169; // (BatchNormBlock(gid: 101), 0)
  short16 var170; // (BatchNormBlock(gid: 101), 1)
  short16 var171; // (BatchNormBlock(gid: 101), 2)
  short16 var172; // (BatchNormBlock(gid: 101), 3)
  short16 var173; // (BatchNormBlock(gid: 101), 0, 'mult_result')
  short16 var174; // (BatchNormBlock(gid: 101), 1, 'mult_result')
  short16 var175; // (BatchNormBlock(gid: 101), 2, 'mult_result')
  short16 var176; // (BatchNormBlock(gid: 101), 3, 'mult_result')
  short16 var177; // ((BatchNormBlock(gid: 101), 0), 'L')
  short16 var178; // ((BatchNormBlock(gid: 101), 0), 'H')
  short16 var179; // ((BatchNormBlock(gid: 101), 0), 'neg1')
  short16 var180; // ((BatchNormBlock(gid: 101), 0), 'const_7fff')
  short16 var181; // ((BatchNormBlock(gid: 101), 0), 'H_sign')
  short16 var182; // ((BatchNormBlock(gid: 101), 0), 'L_sign')
  short16 var183; // ((BatchNormBlock(gid: 101), 0), 'mismatch')
  short16 var184; // ((BatchNormBlock(gid: 101), 0), 'saturate_val')
  short16 var185; // ((BatchNormBlock(gid: 101), 0), 'not_mismatch')
  short16 var186; // ((BatchNormBlock(gid: 101), 0), 'part1')
  short16 var187; // ((BatchNormBlock(gid: 101), 0), 'part2')
  short16 var188; // ((BatchNormBlock(gid: 101), 1), 'L')
  short16 var189; // ((BatchNormBlock(gid: 101), 1), 'H')
  short16 var190; // ((BatchNormBlock(gid: 101), 1), 'neg1')
  short16 var191; // ((BatchNormBlock(gid: 101), 1), 'const_7fff')
  short16 var192; // ((BatchNormBlock(gid: 101), 1), 'H_sign')
  short16 var193; // ((BatchNormBlock(gid: 101), 1), 'L_sign')
  short16 var194; // ((BatchNormBlock(gid: 101), 1), 'mismatch')
  short16 var195; // ((BatchNormBlock(gid: 101), 1), 'saturate_val')
  short16 var196; // ((BatchNormBlock(gid: 101), 1), 'not_mismatch')
  short16 var197; // ((BatchNormBlock(gid: 101), 1), 'part1')
  short16 var198; // ((BatchNormBlock(gid: 101), 1), 'part2')
  short16 var199; // ((BatchNormBlock(gid: 101), 2), 'L')
  short16 var200; // ((BatchNormBlock(gid: 101), 2), 'H')
  short16 var201; // ((BatchNormBlock(gid: 101), 2), 'neg1')
  short16 var202; // ((BatchNormBlock(gid: 101), 2), 'const_7fff')
  short16 var203; // ((BatchNormBlock(gid: 101), 2), 'H_sign')
  short16 var204; // ((BatchNormBlock(gid: 101), 2), 'L_sign')
  short16 var205; // ((BatchNormBlock(gid: 101), 2), 'mismatch')
  short16 var206; // ((BatchNormBlock(gid: 101), 2), 'saturate_val')
  short16 var207; // ((BatchNormBlock(gid: 101), 2), 'not_mismatch')
  short16 var208; // ((BatchNormBlock(gid: 101), 2), 'part1')
  short16 var209; // ((BatchNormBlock(gid: 101), 2), 'part2')
  short16 var210; // ((BatchNormBlock(gid: 101), 3), 'L')
  short16 var211; // ((BatchNormBlock(gid: 101), 3), 'H')
  short16 var212; // ((BatchNormBlock(gid: 101), 3), 'neg1')
  short16 var213; // ((BatchNormBlock(gid: 101), 3), 'const_7fff')
  short16 var214; // ((BatchNormBlock(gid: 101), 3), 'H_sign')
  short16 var215; // ((BatchNormBlock(gid: 101), 3), 'L_sign')
  short16 var216; // ((BatchNormBlock(gid: 101), 3), 'mismatch')
  short16 var217; // ((BatchNormBlock(gid: 101), 3), 'saturate_val')
  short16 var218; // ((BatchNormBlock(gid: 101), 3), 'not_mismatch')
  short16 var219; // ((BatchNormBlock(gid: 101), 3), 'part1')
  short16 var220; // ((BatchNormBlock(gid: 101), 3), 'part2')
  short16 var221; // (TensorEdge(((100, -68), config), ((100, 77), config)), 0)
  short16 var222; // (AddBlock(gid: 78), 0)
  short16 var223; // (AddBlock(gid: 78), 1)
  short16 var224; // (AddBlock(gid: 78), 2)
  short16 var225; // (AddBlock(gid: 78), 3)
  short16 var226; // (TensorEdge((-99, fused_scale), (106, fused_scale)), 0)
  short16 var227; // (TensorEdge((-99, fused_scale), (106, fused_scale)), 1)
  short16 var228; // (TensorEdge((-99, fused_scale), (106, fused_scale)), 2)
  short16 var229; // (TensorEdge((-99, fused_scale), (106, fused_scale)), 3)
  short16 var230; // (TensorEdge((-100, fused_bias), (106, fused_bias)), 0)
  short16 var231; // (TensorEdge((-100, fused_bias), (106, fused_bias)), 1)
  short16 var232; // (TensorEdge((-100, fused_bias), (106, fused_bias)), 2)
  short16 var233; // (TensorEdge((-100, fused_bias), (106, fused_bias)), 3)
  short16 var234; // (TensorEdge((105, odata), (106, data)), 0)
  short16 var235; // (TensorEdge((105, odata), (106, data)), 1)
  short16 var236; // (TensorEdge((105, odata), (106, data)), 2)
  short16 var237; // (TensorEdge((105, odata), (106, data)), 3)
  short16 var238; // (BatchNormBlock(gid: 106), 0)
  short16 var239; // (BatchNormBlock(gid: 106), 1)
  short16 var240; // (BatchNormBlock(gid: 106), 2)
  short16 var241; // (BatchNormBlock(gid: 106), 3)
  short16 var242; // (BatchNormBlock(gid: 106), 0, 'mult_result')
  short16 var243; // (BatchNormBlock(gid: 106), 1, 'mult_result')
  short16 var244; // (BatchNormBlock(gid: 106), 2, 'mult_result')
  short16 var245; // (BatchNormBlock(gid: 106), 3, 'mult_result')
  short16 var246; // ((BatchNormBlock(gid: 106), 0), 'L')
  short16 var247; // ((BatchNormBlock(gid: 106), 0), 'H')
  short16 var248; // ((BatchNormBlock(gid: 106), 0), 'neg1')
  short16 var249; // ((BatchNormBlock(gid: 106), 0), 'const_7fff')
  short16 var250; // ((BatchNormBlock(gid: 106), 0), 'H_sign')
  short16 var251; // ((BatchNormBlock(gid: 106), 0), 'L_sign')
  short16 var252; // ((BatchNormBlock(gid: 106), 0), 'mismatch')
  short16 var253; // ((BatchNormBlock(gid: 106), 0), 'saturate_val')
  short16 var254; // ((BatchNormBlock(gid: 106), 0), 'not_mismatch')
  short16 var255; // ((BatchNormBlock(gid: 106), 0), 'part1')
  short16 var256; // ((BatchNormBlock(gid: 106), 0), 'part2')
  short16 var257; // ((BatchNormBlock(gid: 106), 1), 'L')
  short16 var258; // ((BatchNormBlock(gid: 106), 1), 'H')
  short16 var259; // ((BatchNormBlock(gid: 106), 1), 'neg1')
  short16 var260; // ((BatchNormBlock(gid: 106), 1), 'const_7fff')
  short16 var261; // ((BatchNormBlock(gid: 106), 1), 'H_sign')
  short16 var262; // ((BatchNormBlock(gid: 106), 1), 'L_sign')
  short16 var263; // ((BatchNormBlock(gid: 106), 1), 'mismatch')
  short16 var264; // ((BatchNormBlock(gid: 106), 1), 'saturate_val')
  short16 var265; // ((BatchNormBlock(gid: 106), 1), 'not_mismatch')
  short16 var266; // ((BatchNormBlock(gid: 106), 1), 'part1')
  short16 var267; // ((BatchNormBlock(gid: 106), 1), 'part2')
  short16 var268; // ((BatchNormBlock(gid: 106), 2), 'L')
  short16 var269; // ((BatchNormBlock(gid: 106), 2), 'H')
  short16 var270; // ((BatchNormBlock(gid: 106), 2), 'neg1')
  short16 var271; // ((BatchNormBlock(gid: 106), 2), 'const_7fff')
  short16 var272; // ((BatchNormBlock(gid: 106), 2), 'H_sign')
  short16 var273; // ((BatchNormBlock(gid: 106), 2), 'L_sign')
  short16 var274; // ((BatchNormBlock(gid: 106), 2), 'mismatch')
  short16 var275; // ((BatchNormBlock(gid: 106), 2), 'saturate_val')
  short16 var276; // ((BatchNormBlock(gid: 106), 2), 'not_mismatch')
  short16 var277; // ((BatchNormBlock(gid: 106), 2), 'part1')
  short16 var278; // ((BatchNormBlock(gid: 106), 2), 'part2')
  short16 var279; // ((BatchNormBlock(gid: 106), 3), 'L')
  short16 var280; // ((BatchNormBlock(gid: 106), 3), 'H')
  short16 var281; // ((BatchNormBlock(gid: 106), 3), 'neg1')
  short16 var282; // ((BatchNormBlock(gid: 106), 3), 'const_7fff')
  short16 var283; // ((BatchNormBlock(gid: 106), 3), 'H_sign')
  short16 var284; // ((BatchNormBlock(gid: 106), 3), 'L_sign')
  short16 var285; // ((BatchNormBlock(gid: 106), 3), 'mismatch')
  short16 var286; // ((BatchNormBlock(gid: 106), 3), 'saturate_val')
  short16 var287; // ((BatchNormBlock(gid: 106), 3), 'not_mismatch')
  short16 var288; // ((BatchNormBlock(gid: 106), 3), 'part1')
  short16 var289; // ((BatchNormBlock(gid: 106), 3), 'part2')
  short16 var290; // (TensorEdge((-94, scale), (107, rhs)), 0)
  short16 var291; // (TensorEdge((-94, scale), (107, rhs)), 1)
  short16 var292; // (TensorEdge((-94, scale), (107, rhs)), 2)
  short16 var293; // (TensorEdge((-94, scale), (107, rhs)), 3)
  short16 var294; // (TensorEdge((106, odata), (107, lhs)), 0)
  short16 var295; // (MultlBlock(gid: 107), 0)
  short16 var296; // ((MultlBlock(gid: 107), 0), 'L')
  short16 var297; // ((MultlBlock(gid: 107), 0), 'H')
  short16 var298; // ((MultlBlock(gid: 107), 0), 'neg1')
  short16 var299; // ((MultlBlock(gid: 107), 0), 'const_7fff')
  short16 var300; // ((MultlBlock(gid: 107), 0), 'H_sign')
  short16 var301; // ((MultlBlock(gid: 107), 0), 'L_sign')
  short16 var302; // ((MultlBlock(gid: 107), 0), 'mismatch')
  short16 var303; // ((MultlBlock(gid: 107), 0), 'saturate_val')
  short16 var304; // ((MultlBlock(gid: 107), 0), 'not_mismatch')
  short16 var305; // ((MultlBlock(gid: 107), 0), 'part1')
  short16 var306; // ((MultlBlock(gid: 107), 0), 'part2')
  if (hid == 0 && wid == 1) { // imce_0_1
    // generate: TensorEdge((-83, config), (91, config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge((-83, config), (91, config)), config write
    // generate: conv exec6
    // generate: conv exec6_row_group0_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 8; i1++) { // generate : conv exec6_row_group0_outer_loop(iterate row offset)
      // generate: conv exec6_row_group0_col_group0
      // generate : conv exec6_row_group0_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 18; i2++) { // generate : load_block
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((90, odata), (91, data), 1), imce_0_2 -> imce_0_1

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var1 = __builtin_IMCE_GET_CREG((short)0);
      var2 = __builtin_IMCE_GET_CREG((short)1);
      var3 = __builtin_IMCE_GET_CREG((short)2);
      var4 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(2, var1, 2, 0); // TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
      __builtin_IMCE_SEND(2, var2, 2, 0); // TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
      __builtin_IMCE_SEND(2, var3, 2, 0); // TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
      __builtin_IMCE_SEND(2, var4, 2, 0); // TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
      // endgenerate : conv exec6_row_group0_col_group0
      // endgenerate: conv exec6_row_group0_col_group0
      // generate: conv exec6_row_group0_col_group1
      for (int i2 = 0; i2 < 7; i2++) { // generate : conv exec6_row_group0_col_group1

        // generate: load_block
        for (int i3 = 0; i3 < 2; i3++) { // generate : load_block
          for (int i4 = 0; i4 < 4; i4++) { // generate
            __builtin_IMCE_LOAD_LB(0); // TensorEdge((90, odata), (91, data), 1), imce_0_2 -> imce_0_1

          } // endgenerate
        } // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var1 = __builtin_IMCE_GET_CREG((short)0);
        var2 = __builtin_IMCE_GET_CREG((short)1);
        var3 = __builtin_IMCE_GET_CREG((short)2);
        var4 = __builtin_IMCE_GET_CREG((short)3);
        __builtin_IMCE_SEND(2, var1, 2, 0); // TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
        __builtin_IMCE_SEND(2, var2, 2, 0); // TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
        __builtin_IMCE_SEND(2, var3, 2, 0); // TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
        __builtin_IMCE_SEND(2, var4, 2, 0); // TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
      } // endgenerate : conv exec6_row_group0_col_group1
      // endgenerate: conv exec6_row_group0_col_group1
    } // endgenerate : conv exec6_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec6_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec6
    __builtin_IMCE_STOP();
  }
  else if (hid == 0 && wid == 2) { // imce_0_2
    // generate: TensorEdge((-80, min), (89, min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge((-80, min), (89, min)), min write
    // generate: TensorEdge((-81, max), (89, max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge((-81, max), (89, max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 256; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      var44 = __builtin_IMCE_RECV(2); // TensorEdge(((88, 86), odata), (89, data)), imce_0_3 -> imce_0_2
      var45 = __builtin_IMCE_RECV(2); // TensorEdge(((88, 86), odata), (89, data)), imce_0_3 -> imce_0_2
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var44, 0, 15, 0);
      __builtin_IMCE_MM_QUANT(var45, 0, 15, 1);
      var46 = __builtin_IMCE_GET_QREG(0);
      var47 = __builtin_IMCE_GET_QREG(1);
      var48 = __builtin_IMCE_GET_QREG(2);
      var49 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      __builtin_IMCE_SEND(1, var46, 0, 0); // TensorEdge((89, odata), (90, data)), imce_0_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var47, 0, 0); // TensorEdge((89, odata), (90, data)), imce_0_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var48, 0, 0); // TensorEdge((89, odata), (90, data)), imce_0_2 -> imce_1_1
      __builtin_IMCE_SEND(1, var49, 0, 0); // TensorEdge((89, odata), (90, data)), imce_0_2 -> imce_1_1
      // endgenerate: min_max_quantize_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 0 && wid == 3) { // imce_0_3
    // generate: call_created_loop
    for (int i1 = 0; i1 < 512; i1++) { // generate : call_created_loop
      // generate: imcflow.vecops_wrapper

      var50 = __builtin_IMCE_RECV(2); // TensorEdge((-63, odata), ((88, 86), lhs)), inode_0_0 -> imce_0_3
      var51 = __builtin_IMCE_RECV(3); // TensorEdge((-64, odata), ((88, 86), rhs)), inode_1_0 -> imce_0_3
      // generate: imcflow.vecops_block
      // generate: add

      var52 = __builtin_IMCE_ADD(var50, var51, 15);
      // endgenerate: add
      // endgenerate: imcflow.vecops_block
      __builtin_IMCE_SEND(1, var52, 2, 0); // TensorEdge(((88, 86), odata), (104, data)),TensorEdge(((88, 86), odata), (89, data)), imce_0_3 -> imce_0_2
      // endgenerate: imcflow.vecops_wrapper
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 0 && wid == 4) { // imce_0_4
  }
  else if (hid == 1 && wid == 1) { // imce_1_1
    // generate: TensorEdge(((92, -77), config), ((92, 83), config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge(((92, -77), config), ((92, 83), config)), config write
    // generate: conv exec7
    // generate: conv exec7_row_group0_outer_loop(iterate row offset)
    for (int i1 = 0; i1 < 8; i1++) { // generate : conv exec7_row_group0_outer_loop(iterate row offset)
      // generate: conv exec7_row_group0_col_group0
      // generate : conv exec7_row_group0_col_group0. loop count == 1

      // generate: load_block
      for (int i2 = 0; i2 < 18; i2++) { // generate : load_block
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((90, odata), ((92, 83), data), 0), imce_0_2 -> imce_1_1

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var5 = __builtin_IMCE_GET_CREG((short)0);
      var6 = __builtin_IMCE_GET_CREG((short)1);
      var7 = __builtin_IMCE_GET_CREG((short)2);
      var8 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
      var9 = __builtin_IMCE_RECV(2); // TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
      // endgenerate: TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
      // generate: TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
      var10 = __builtin_IMCE_RECV(2); // TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
      // endgenerate: TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
      // generate: TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
      var11 = __builtin_IMCE_RECV(2); // TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
      // endgenerate: TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
      // generate: TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
      var12 = __builtin_IMCE_RECV(2); // TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
      // endgenerate: TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
      // generate: add

      var54 = __builtin_IMCE_ADD(var5, var9, 15);
      var55 = __builtin_IMCE_ADD(var6, var10, 15);
      var56 = __builtin_IMCE_ADD(var7, var11, 15);
      var57 = __builtin_IMCE_ADD(var8, var12, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(1, var54, 2, 0); // TensorEdge(((92, 84), odata), ((93, 80), data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(1, var55, 2, 0); // TensorEdge(((92, 84), odata), ((93, 80), data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(1, var56, 2, 0); // TensorEdge(((92, 84), odata), ((93, 80), data)), imce_1_1 -> imce_2_1
      __builtin_IMCE_SEND(1, var57, 2, 0); // TensorEdge(((92, 84), odata), ((93, 80), data)), imce_1_1 -> imce_2_1
      // endgenerate : conv exec7_row_group0_col_group0
      // endgenerate: conv exec7_row_group0_col_group0
      // generate: conv exec7_row_group0_col_group1
      for (int i2 = 0; i2 < 7; i2++) { // generate : conv exec7_row_group0_col_group1

        // generate: load_block
        for (int i3 = 0; i3 < 2; i3++) { // generate : load_block
          for (int i4 = 0; i4 < 4; i4++) { // generate
            __builtin_IMCE_LOAD_LB(0); // TensorEdge((90, odata), ((92, 83), data), 0), imce_0_2 -> imce_1_1

          } // endgenerate
        } // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var5 = __builtin_IMCE_GET_CREG((short)0);
        var6 = __builtin_IMCE_GET_CREG((short)1);
        var7 = __builtin_IMCE_GET_CREG((short)2);
        var8 = __builtin_IMCE_GET_CREG((short)3);
        // generate: TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
        var9 = __builtin_IMCE_RECV(2); // TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
        // endgenerate: TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
        // generate: TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
        var10 = __builtin_IMCE_RECV(2); // TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
        // endgenerate: TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
        // generate: TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
        var11 = __builtin_IMCE_RECV(2); // TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
        // endgenerate: TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
        // generate: TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
        var12 = __builtin_IMCE_RECV(2); // TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
        // endgenerate: TensorEdge((91, odata), ((92, 84), rhs)), imce_0_1 -> imce_1_1
        // generate: add

        var54 = __builtin_IMCE_ADD(var5, var9, 15);
        var55 = __builtin_IMCE_ADD(var6, var10, 15);
        var56 = __builtin_IMCE_ADD(var7, var11, 15);
        var57 = __builtin_IMCE_ADD(var8, var12, 15);
        // endgenerate: add
        __builtin_IMCE_SEND(1, var54, 2, 0); // TensorEdge(((92, 84), odata), ((93, 80), data)), imce_1_1 -> imce_2_1
        __builtin_IMCE_SEND(1, var55, 2, 0); // TensorEdge(((92, 84), odata), ((93, 80), data)), imce_1_1 -> imce_2_1
        __builtin_IMCE_SEND(1, var56, 2, 0); // TensorEdge(((92, 84), odata), ((93, 80), data)), imce_1_1 -> imce_2_1
        __builtin_IMCE_SEND(1, var57, 2, 0); // TensorEdge(((92, 84), odata), ((93, 80), data)), imce_1_1 -> imce_2_1
      } // endgenerate : conv exec7_row_group0_col_group1
      // endgenerate: conv exec7_row_group0_col_group1
    } // endgenerate : conv exec7_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec7_row_group0_outer_loop(iterate row offset)
    // endgenerate: conv exec7
    __builtin_IMCE_STOP();
  }
  else if (hid == 1 && wid == 2) { // imce_1_2
    // generate: TensorEdge((-89, config), (98, config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge((-89, config), (98, config)), config write
    // generate: conv exec8
    // generate: conv exec8_row_group0_outer_loop(iterate row offset)
    // generate : conv exec8_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec8_row_group0_col_group0
    // generate : conv exec8_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 10; i1++) { // generate : load_block
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((94, odata), (98, data), 1), imce_2_1 -> imce_2_2, imce_1_2

      } // endgenerate
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var13 = __builtin_IMCE_GET_CREG((short)0);
    var14 = __builtin_IMCE_GET_CREG((short)1);
    var15 = __builtin_IMCE_GET_CREG((short)2);
    var16 = __builtin_IMCE_GET_CREG((short)3);
    __builtin_IMCE_SEND(4, var13, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    __builtin_IMCE_SEND(4, var14, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    __builtin_IMCE_SEND(4, var15, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    __builtin_IMCE_SEND(4, var16, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    // endgenerate : conv exec8_row_group0_col_group0
    // endgenerate: conv exec8_row_group0_col_group0
    // generate: conv exec8_row_group0_col_group1
    for (int i1 = 0; i1 < 6; i1++) { // generate : conv exec8_row_group0_col_group1

      // generate: load_block
      // generate : load_block. loop count == 1
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((94, odata), (98, data), 1), imce_2_1 -> imce_2_2, imce_1_2

      } // endgenerate
      // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var13 = __builtin_IMCE_GET_CREG((short)0);
      var14 = __builtin_IMCE_GET_CREG((short)1);
      var15 = __builtin_IMCE_GET_CREG((short)2);
      var16 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(4, var13, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      __builtin_IMCE_SEND(4, var14, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      __builtin_IMCE_SEND(4, var15, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      __builtin_IMCE_SEND(4, var16, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
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
    __builtin_IMCE_SEND(4, var13, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    __builtin_IMCE_SEND(4, var14, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    __builtin_IMCE_SEND(4, var15, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    __builtin_IMCE_SEND(4, var16, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
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
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((94, odata), (98, data), 1), imce_2_1 -> imce_2_2, imce_1_2

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var13 = __builtin_IMCE_GET_CREG((short)0);
      var14 = __builtin_IMCE_GET_CREG((short)1);
      var15 = __builtin_IMCE_GET_CREG((short)2);
      var16 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(4, var13, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      __builtin_IMCE_SEND(4, var14, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      __builtin_IMCE_SEND(4, var15, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      __builtin_IMCE_SEND(4, var16, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // endgenerate : conv exec8_row_group1_col_group0
      // endgenerate: conv exec8_row_group1_col_group0
      // generate: conv exec8_row_group1_col_group1
      for (int i2 = 0; i2 < 6; i2++) { // generate : conv exec8_row_group1_col_group1

        // generate: load_block
        // generate : load_block. loop count == 1
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((94, odata), (98, data), 1), imce_2_1 -> imce_2_2, imce_1_2

        } // endgenerate
        // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var13 = __builtin_IMCE_GET_CREG((short)0);
        var14 = __builtin_IMCE_GET_CREG((short)1);
        var15 = __builtin_IMCE_GET_CREG((short)2);
        var16 = __builtin_IMCE_GET_CREG((short)3);
        __builtin_IMCE_SEND(4, var13, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
        __builtin_IMCE_SEND(4, var14, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
        __builtin_IMCE_SEND(4, var15, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
        __builtin_IMCE_SEND(4, var16, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
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
      __builtin_IMCE_SEND(4, var13, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      __builtin_IMCE_SEND(4, var14, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      __builtin_IMCE_SEND(4, var15, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      __builtin_IMCE_SEND(4, var16, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
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
      __builtin_IMCE_SEND(4, var13, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      __builtin_IMCE_SEND(4, var14, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      __builtin_IMCE_SEND(4, var15, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      __builtin_IMCE_SEND(4, var16, 2, 0); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    } // endgenerate : conv exec8_row_group2_col_group0
    // endgenerate: conv exec8_row_group2_col_group0
    // endgenerate : conv exec8_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec8_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec8
    __builtin_IMCE_STOP();
  }
  else if (hid == 1 && wid == 3) { // imce_1_3
    // generate: TensorEdge((-95, min), (104, min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge((-95, min), (104, min)), min write
    // generate: TensorEdge((-96, max), (104, max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge((-96, max), (104, max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 256; i1++) { // generate : call_created_loop
      // generate: min_max_quantize_standalone

      var65 = __builtin_IMCE_RECV(2); // TensorEdge(((88, 86), odata), (104, data)), imce_0_3 -> imce_1_3
      var66 = __builtin_IMCE_RECV(2); // TensorEdge(((88, 86), odata), (104, data)), imce_0_3 -> imce_1_3
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var65, 0, 15, 0);
      __builtin_IMCE_MM_QUANT(var66, 0, 15, 1);
      var67 = __builtin_IMCE_GET_QREG(0);
      var68 = __builtin_IMCE_GET_QREG(1);
      var69 = __builtin_IMCE_GET_QREG(2);
      var70 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      __builtin_IMCE_SEND(3, var67, 0, 0); // TensorEdge((104, odata), (105, data)), imce_1_3 -> imce_2_3
      __builtin_IMCE_SEND(3, var68, 0, 0); // TensorEdge((104, odata), (105, data)), imce_1_3 -> imce_2_3
      __builtin_IMCE_SEND(3, var69, 0, 0); // TensorEdge((104, odata), (105, data)), imce_1_3 -> imce_2_3
      __builtin_IMCE_SEND(3, var70, 0, 0); // TensorEdge((104, odata), (105, data)), imce_1_3 -> imce_2_3
      // endgenerate: min_max_quantize_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 1 && wid == 4) { // imce_1_4
  }
  else if (hid == 2 && wid == 1) { // imce_2_1
    // generate: TensorEdge(((93, -70), fused_scale), ((93, 80), fused_scale)), fused_scale write

    var71 = __builtin_IMCE_RECV(1);
    var72 = __builtin_IMCE_RECV(1);
    var73 = __builtin_IMCE_RECV(1);
    var74 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((93, -70), fused_scale), ((93, 80), fused_scale)), fused_scale write
    // generate: TensorEdge(((93, -71), fused_bias), ((93, 80), fused_bias)), fused_bias write

    var75 = __builtin_IMCE_RECV(1);
    var76 = __builtin_IMCE_RECV(1);
    var77 = __builtin_IMCE_RECV(1);
    var78 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge(((93, -71), fused_bias), ((93, 80), fused_bias)), fused_bias write
    // generate: TensorEdge(((93, -72), min), ((93, 81), min)), min write

    __builtin_IMCE_RECV_MIN(1);
    // endgenerate: TensorEdge(((93, -72), min), ((93, 81), min)), min write
    // generate: TensorEdge(((93, -73), max), ((93, 81), max)), max write

    __builtin_IMCE_RECV_MAX(1);
    // endgenerate: TensorEdge(((93, -73), max), ((93, 81), max)), max write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 64; i1++) { // generate : call_created_loop
      // generate: imcflow.preop-minmax_wrapper

      var81 = __builtin_IMCE_RECV(2); // TensorEdge(((92, 84), odata), ((93, 80), data)), imce_1_1 -> imce_2_1
      var82 = __builtin_IMCE_RECV(2); // TensorEdge(((92, 84), odata), ((93, 80), data)), imce_1_1 -> imce_2_1
      var83 = __builtin_IMCE_RECV(2); // TensorEdge(((92, 84), odata), ((93, 80), data)), imce_1_1 -> imce_2_1
      var84 = __builtin_IMCE_RECV(2); // TensorEdge(((92, 84), odata), ((93, 80), data)), imce_1_1 -> imce_2_1
      // generate: imcflow.preop-minmax_block
      // generate: batch_norm


      var97 = __builtin_IMCE_MULTL(var81, var71, 15);
      var98 = __builtin_IMCE_MULTH(var81, var71, 15);
      var99 = __builtin_IMCE_SUBI(0, 1);
      var100 = __builtin_IMCE_SRLI(var99, 1);
      var101 = __builtin_IMCE_SRAI(var98, 15);
      var102 = __builtin_IMCE_SRAI(var97, 15);
      var103 = __builtin_IMCE_XOR(var101, var102, 15);
      var104 = __builtin_IMCE_XOR(var101, var100, 15);
      var105 = __builtin_IMCE_XOR(var103, var99, 15);
      var106 = __builtin_IMCE_AND(var103, var104, 15);
      var107 = __builtin_IMCE_AND(var105, var97, 15);
      var90 = __builtin_IMCE_OR(var106, var107, 15);
      var89 = __builtin_IMCE_ADD(var90, var75, 15);

      var108 = __builtin_IMCE_MULTL(var82, var72, 15);
      var109 = __builtin_IMCE_MULTH(var82, var72, 15);
      var110 = __builtin_IMCE_SUBI(0, 1);
      var111 = __builtin_IMCE_SRLI(var110, 1);
      var112 = __builtin_IMCE_SRAI(var109, 15);
      var113 = __builtin_IMCE_SRAI(var108, 15);
      var114 = __builtin_IMCE_XOR(var112, var113, 15);
      var115 = __builtin_IMCE_XOR(var112, var111, 15);
      var116 = __builtin_IMCE_XOR(var114, var110, 15);
      var117 = __builtin_IMCE_AND(var114, var115, 15);
      var118 = __builtin_IMCE_AND(var116, var108, 15);
      var92 = __builtin_IMCE_OR(var117, var118, 15);
      var91 = __builtin_IMCE_ADD(var92, var76, 15);

      var119 = __builtin_IMCE_MULTL(var83, var73, 15);
      var120 = __builtin_IMCE_MULTH(var83, var73, 15);
      var121 = __builtin_IMCE_SUBI(0, 1);
      var122 = __builtin_IMCE_SRLI(var121, 1);
      var123 = __builtin_IMCE_SRAI(var120, 15);
      var124 = __builtin_IMCE_SRAI(var119, 15);
      var125 = __builtin_IMCE_XOR(var123, var124, 15);
      var126 = __builtin_IMCE_XOR(var123, var122, 15);
      var127 = __builtin_IMCE_XOR(var125, var121, 15);
      var128 = __builtin_IMCE_AND(var125, var126, 15);
      var129 = __builtin_IMCE_AND(var127, var119, 15);
      var94 = __builtin_IMCE_OR(var128, var129, 15);
      var93 = __builtin_IMCE_ADD(var94, var77, 15);

      var130 = __builtin_IMCE_MULTL(var84, var74, 15);
      var131 = __builtin_IMCE_MULTH(var84, var74, 15);
      var132 = __builtin_IMCE_SUBI(0, 1);
      var133 = __builtin_IMCE_SRLI(var132, 1);
      var134 = __builtin_IMCE_SRAI(var131, 15);
      var135 = __builtin_IMCE_SRAI(var130, 15);
      var136 = __builtin_IMCE_XOR(var134, var135, 15);
      var137 = __builtin_IMCE_XOR(var134, var133, 15);
      var138 = __builtin_IMCE_XOR(var136, var132, 15);
      var139 = __builtin_IMCE_AND(var136, var137, 15);
      var140 = __builtin_IMCE_AND(var138, var130, 15);
      var96 = __builtin_IMCE_OR(var139, var140, 15);
      var95 = __builtin_IMCE_ADD(var96, var78, 15);
      // endgenerate: batch_norm
      // generate: min_max_quantize

      __builtin_IMCE_MM_QUANT(var89, 0, 15, 0);
      __builtin_IMCE_MM_QUANT(var91, 0, 15, 1);
      __builtin_IMCE_MM_QUANT(var93, 0, 15, 2);
      __builtin_IMCE_MM_QUANT(var95, 0, 15, 3);
      var85 = __builtin_IMCE_GET_QREG(0);
      var86 = __builtin_IMCE_GET_QREG(1);
      var87 = __builtin_IMCE_GET_QREG(2);
      var88 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize
      // endgenerate: imcflow.preop-minmax_block
      __builtin_IMCE_SEND(2, var85, 0, 0); // TensorEdge(((93, 81), odata), (94, data)), imce_2_1 -> imce_3_2
      __builtin_IMCE_SEND(2, var86, 0, 0); // TensorEdge(((93, 81), odata), (94, data)), imce_2_1 -> imce_3_2
      __builtin_IMCE_SEND(2, var87, 0, 0); // TensorEdge(((93, 81), odata), (94, data)), imce_2_1 -> imce_3_2
      __builtin_IMCE_SEND(2, var88, 0, 0); // TensorEdge(((93, 81), odata), (94, data)), imce_2_1 -> imce_3_2
      // endgenerate: imcflow.preop-minmax_wrapper
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 2 && wid == 2) { // imce_2_2
    // generate: TensorEdge(((99, -87), config), ((99, 95), config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge(((99, -87), config), ((99, 95), config)), config write
    // generate: conv exec9
    // generate: conv exec9_row_group0_outer_loop(iterate row offset)
    // generate : conv exec9_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec9_row_group0_col_group0
    // generate : conv exec9_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 10; i1++) { // generate : load_block
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((94, odata), ((99, 95), data), 0), imce_2_1 -> imce_2_2

      } // endgenerate
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var17 = __builtin_IMCE_GET_CREG((short)0);
    var18 = __builtin_IMCE_GET_CREG((short)1);
    var19 = __builtin_IMCE_GET_CREG((short)2);
    var20 = __builtin_IMCE_GET_CREG((short)3);
    // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    var21 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    var22 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    var23 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    var24 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    // generate: add

    var142 = __builtin_IMCE_ADD(var17, var21, 15);
    var143 = __builtin_IMCE_ADD(var18, var22, 15);
    var144 = __builtin_IMCE_ADD(var19, var23, 15);
    var145 = __builtin_IMCE_ADD(var20, var24, 15);
    // endgenerate: add
    __builtin_IMCE_SEND(3, var142, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    __builtin_IMCE_SEND(3, var143, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    __builtin_IMCE_SEND(3, var144, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    __builtin_IMCE_SEND(3, var145, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    // endgenerate : conv exec9_row_group0_col_group0
    // endgenerate: conv exec9_row_group0_col_group0
    // generate: conv exec9_row_group0_col_group1
    for (int i1 = 0; i1 < 6; i1++) { // generate : conv exec9_row_group0_col_group1

      // generate: load_block
      // generate : load_block. loop count == 1
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((94, odata), ((99, 95), data), 0), imce_2_1 -> imce_2_2

      } // endgenerate
      // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var17 = __builtin_IMCE_GET_CREG((short)0);
      var18 = __builtin_IMCE_GET_CREG((short)1);
      var19 = __builtin_IMCE_GET_CREG((short)2);
      var20 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      var21 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      var22 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      var23 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      var24 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // generate: add

      var142 = __builtin_IMCE_ADD(var17, var21, 15);
      var143 = __builtin_IMCE_ADD(var18, var22, 15);
      var144 = __builtin_IMCE_ADD(var19, var23, 15);
      var145 = __builtin_IMCE_ADD(var20, var24, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(3, var142, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(3, var143, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(3, var144, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(3, var145, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
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
    // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    var21 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    var22 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    var23 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    var24 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
    // generate: add

    var142 = __builtin_IMCE_ADD(var17, var21, 15);
    var143 = __builtin_IMCE_ADD(var18, var22, 15);
    var144 = __builtin_IMCE_ADD(var19, var23, 15);
    var145 = __builtin_IMCE_ADD(var20, var24, 15);
    // endgenerate: add
    __builtin_IMCE_SEND(3, var142, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    __builtin_IMCE_SEND(3, var143, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    __builtin_IMCE_SEND(3, var144, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    __builtin_IMCE_SEND(3, var145, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
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
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((94, odata), ((99, 95), data), 0), imce_2_1 -> imce_2_2

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var17 = __builtin_IMCE_GET_CREG((short)0);
      var18 = __builtin_IMCE_GET_CREG((short)1);
      var19 = __builtin_IMCE_GET_CREG((short)2);
      var20 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      var21 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      var22 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      var23 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      var24 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // generate: add

      var142 = __builtin_IMCE_ADD(var17, var21, 15);
      var143 = __builtin_IMCE_ADD(var18, var22, 15);
      var144 = __builtin_IMCE_ADD(var19, var23, 15);
      var145 = __builtin_IMCE_ADD(var20, var24, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(3, var142, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(3, var143, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(3, var144, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(3, var145, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // endgenerate : conv exec9_row_group1_col_group0
      // endgenerate: conv exec9_row_group1_col_group0
      // generate: conv exec9_row_group1_col_group1
      for (int i2 = 0; i2 < 6; i2++) { // generate : conv exec9_row_group1_col_group1

        // generate: load_block
        // generate : load_block. loop count == 1
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((94, odata), ((99, 95), data), 0), imce_2_1 -> imce_2_2

        } // endgenerate
        // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var17 = __builtin_IMCE_GET_CREG((short)0);
        var18 = __builtin_IMCE_GET_CREG((short)1);
        var19 = __builtin_IMCE_GET_CREG((short)2);
        var20 = __builtin_IMCE_GET_CREG((short)3);
        // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
        var21 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
        // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
        // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
        var22 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
        // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
        // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
        var23 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
        // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
        // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
        var24 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
        // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
        // generate: add

        var142 = __builtin_IMCE_ADD(var17, var21, 15);
        var143 = __builtin_IMCE_ADD(var18, var22, 15);
        var144 = __builtin_IMCE_ADD(var19, var23, 15);
        var145 = __builtin_IMCE_ADD(var20, var24, 15);
        // endgenerate: add
        __builtin_IMCE_SEND(3, var142, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
        __builtin_IMCE_SEND(3, var143, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
        __builtin_IMCE_SEND(3, var144, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
        __builtin_IMCE_SEND(3, var145, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
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
      // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      var21 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      var22 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      var23 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      var24 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // generate: add

      var142 = __builtin_IMCE_ADD(var17, var21, 15);
      var143 = __builtin_IMCE_ADD(var18, var22, 15);
      var144 = __builtin_IMCE_ADD(var19, var23, 15);
      var145 = __builtin_IMCE_ADD(var20, var24, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(3, var142, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(3, var143, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(3, var144, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(3, var145, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
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
      // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      var21 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      var22 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      var23 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // generate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      var24 = __builtin_IMCE_RECV(2); // TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // endgenerate: TensorEdge((98, odata), ((99, 96), rhs)), imce_1_2 -> imce_2_2
      // generate: add

      var142 = __builtin_IMCE_ADD(var17, var21, 15);
      var143 = __builtin_IMCE_ADD(var18, var22, 15);
      var144 = __builtin_IMCE_ADD(var19, var23, 15);
      var145 = __builtin_IMCE_ADD(var20, var24, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(3, var142, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(3, var143, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(3, var144, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      __builtin_IMCE_SEND(3, var145, 2, 0); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    } // endgenerate : conv exec9_row_group2_col_group0
    // endgenerate: conv exec9_row_group2_col_group0
    // endgenerate : conv exec9_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec9_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec9
    __builtin_IMCE_STOP();
  }
  else if (hid == 2 && wid == 3) { // imce_2_3
    // generate: TensorEdge((-98, config), (105, config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge((-98, config), (105, config)), config write
    // generate: conv exec11
    // generate: conv exec11_row_group0_outer_loop(iterate row offset)
    // generate : conv exec11_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec11_row_group0_col_group0
    // generate : conv exec11_row_group0_col_group0. loop count == 1

    // generate: load_block
    // generate : load_block. loop count == 1
    for (int i1 = 0; i1 < 4; i1++) { // generate
      __builtin_IMCE_LOAD_LB(0); // TensorEdge((104, odata), (105, data)), imce_1_3 -> imce_2_3

    } // endgenerate
    // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var33 = __builtin_IMCE_GET_CREG((short)0);
    var34 = __builtin_IMCE_GET_CREG((short)1);
    var35 = __builtin_IMCE_GET_CREG((short)2);
    var36 = __builtin_IMCE_GET_CREG((short)3);
    __builtin_IMCE_SEND(3, var33, 2, 0); // TensorEdge((105, odata), (106, data)), imce_2_3 -> imce_3_3
    __builtin_IMCE_SEND(3, var34, 2, 0); // TensorEdge((105, odata), (106, data)), imce_2_3 -> imce_3_3
    __builtin_IMCE_SEND(3, var35, 2, 0); // TensorEdge((105, odata), (106, data)), imce_2_3 -> imce_3_3
    __builtin_IMCE_SEND(3, var36, 2, 0); // TensorEdge((105, odata), (106, data)), imce_2_3 -> imce_3_3
    // endgenerate : conv exec11_row_group0_col_group0
    // endgenerate: conv exec11_row_group0_col_group0
    // generate: conv exec11_row_group0_col_group1
    for (int i1 = 0; i1 < 7; i1++) { // generate : conv exec11_row_group0_col_group1

      // generate: load_block
      for (int i2 = 0; i2 < 2; i2++) { // generate : load_block
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((104, odata), (105, data)), imce_1_3 -> imce_2_3

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var33 = __builtin_IMCE_GET_CREG((short)0);
      var34 = __builtin_IMCE_GET_CREG((short)1);
      var35 = __builtin_IMCE_GET_CREG((short)2);
      var36 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(3, var33, 2, 0); // TensorEdge((105, odata), (106, data)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SEND(3, var34, 2, 0); // TensorEdge((105, odata), (106, data)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SEND(3, var35, 2, 0); // TensorEdge((105, odata), (106, data)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SEND(3, var36, 2, 0); // TensorEdge((105, odata), (106, data)), imce_2_3 -> imce_3_3
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
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((104, odata), (105, data)), imce_1_3 -> imce_2_3

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var33 = __builtin_IMCE_GET_CREG((short)0);
      var34 = __builtin_IMCE_GET_CREG((short)1);
      var35 = __builtin_IMCE_GET_CREG((short)2);
      var36 = __builtin_IMCE_GET_CREG((short)3);
      __builtin_IMCE_SEND(3, var33, 2, 0); // TensorEdge((105, odata), (106, data)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SEND(3, var34, 2, 0); // TensorEdge((105, odata), (106, data)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SEND(3, var35, 2, 0); // TensorEdge((105, odata), (106, data)), imce_2_3 -> imce_3_3
      __builtin_IMCE_SEND(3, var36, 2, 0); // TensorEdge((105, odata), (106, data)), imce_2_3 -> imce_3_3
      // endgenerate : conv exec11_row_group1_col_group0
      // endgenerate: conv exec11_row_group1_col_group0
      // generate: conv exec11_row_group1_col_group1
      for (int i2 = 0; i2 < 7; i2++) { // generate : conv exec11_row_group1_col_group1

        // generate: load_block
        for (int i3 = 0; i3 < 2; i3++) { // generate : load_block
          for (int i4 = 0; i4 < 4; i4++) { // generate
            __builtin_IMCE_LOAD_LB(0); // TensorEdge((104, odata), (105, data)), imce_1_3 -> imce_2_3

          } // endgenerate
        } // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var33 = __builtin_IMCE_GET_CREG((short)0);
        var34 = __builtin_IMCE_GET_CREG((short)1);
        var35 = __builtin_IMCE_GET_CREG((short)2);
        var36 = __builtin_IMCE_GET_CREG((short)3);
        __builtin_IMCE_SEND(3, var33, 2, 0); // TensorEdge((105, odata), (106, data)), imce_2_3 -> imce_3_3
        __builtin_IMCE_SEND(3, var34, 2, 0); // TensorEdge((105, odata), (106, data)), imce_2_3 -> imce_3_3
        __builtin_IMCE_SEND(3, var35, 2, 0); // TensorEdge((105, odata), (106, data)), imce_2_3 -> imce_3_3
        __builtin_IMCE_SEND(3, var36, 2, 0); // TensorEdge((105, odata), (106, data)), imce_2_3 -> imce_3_3
      } // endgenerate : conv exec11_row_group1_col_group1
      // endgenerate: conv exec11_row_group1_col_group1
    } // endgenerate : conv exec11_row_group1_outer_loop(iterate row offset)
    // endgenerate: conv exec11_row_group1_outer_loop(iterate row offset)
    // generate: conv exec11_tail_loop
    for (int i1 = 0; i1 < 68; i1++) { // generate : conv exec11_tail_loop
      __builtin_IMCE_RECV(0);
    } // endgenerate : conv exec11_tail_loop
    // endgenerate: conv exec11_tail_loop
    // endgenerate: conv exec11
    __builtin_IMCE_STOP();
  }
  else if (hid == 2 && wid == 4) { // imce_2_4
    // generate: add const

    var151 = __builtin_IMCE_RECV(1);
    var152 = __builtin_IMCE_RECV(1);
    var153 = __builtin_IMCE_RECV(1);
    var154 = __builtin_IMCE_RECV(1);
    // endgenerate: add const
    // generate: call_created_loop
    for (int i1 = 0; i1 < 256; i1++) { // generate : call_created_loop
      // generate: imcflow.vecops_wrapper

      var155 = __builtin_IMCE_RECV(2); // TensorEdge((107, odata), ((108, 102), lhs)), imce_3_4 -> imce_2_4
      // generate: imcflow.vecops_block
      // generate: add

      var156 = __builtin_IMCE_ADD(var151, var155, 15);
      // endgenerate: add
      // endgenerate: imcflow.vecops_block
      __builtin_IMCE_SEND(1, var156, 2, 0); // TensorEdge(((108, 102), odata), (109, func_out1), 1), imce_2_4 -> inode_2_0
      // endgenerate: imcflow.vecops_wrapper
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 1) { // imce_3_1
    // generate: TensorEdge((-90, fused_scale), (101, fused_scale)), fused_scale write

    var157 = __builtin_IMCE_RECV(1);
    var158 = __builtin_IMCE_RECV(1);
    var159 = __builtin_IMCE_RECV(1);
    var160 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge((-90, fused_scale), (101, fused_scale)), fused_scale write
    // generate: TensorEdge((-91, fused_bias), (101, fused_bias)), fused_bias write

    var161 = __builtin_IMCE_RECV(1);
    var162 = __builtin_IMCE_RECV(1);
    var163 = __builtin_IMCE_RECV(1);
    var164 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge((-91, fused_bias), (101, fused_bias)), fused_bias write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 64; i1++) { // generate : call_created_loop
      // generate: bn_standalone

      var165 = __builtin_IMCE_RECV(2); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
      var166 = __builtin_IMCE_RECV(2); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
      var167 = __builtin_IMCE_RECV(2); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
      var168 = __builtin_IMCE_RECV(2); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
      // generate: batch_norm


      var177 = __builtin_IMCE_MULTL(var165, var157, 15);
      var178 = __builtin_IMCE_MULTH(var165, var157, 15);
      var179 = __builtin_IMCE_SUBI(0, 1);
      var180 = __builtin_IMCE_SRLI(var179, 1);
      var181 = __builtin_IMCE_SRAI(var178, 15);
      var182 = __builtin_IMCE_SRAI(var177, 15);
      var183 = __builtin_IMCE_XOR(var181, var182, 15);
      var184 = __builtin_IMCE_XOR(var181, var180, 15);
      var185 = __builtin_IMCE_XOR(var183, var179, 15);
      var186 = __builtin_IMCE_AND(var183, var184, 15);
      var187 = __builtin_IMCE_AND(var185, var177, 15);
      var173 = __builtin_IMCE_OR(var186, var187, 15);
      var169 = __builtin_IMCE_ADD(var173, var161, 15);

      var188 = __builtin_IMCE_MULTL(var166, var158, 15);
      var189 = __builtin_IMCE_MULTH(var166, var158, 15);
      var190 = __builtin_IMCE_SUBI(0, 1);
      var191 = __builtin_IMCE_SRLI(var190, 1);
      var192 = __builtin_IMCE_SRAI(var189, 15);
      var193 = __builtin_IMCE_SRAI(var188, 15);
      var194 = __builtin_IMCE_XOR(var192, var193, 15);
      var195 = __builtin_IMCE_XOR(var192, var191, 15);
      var196 = __builtin_IMCE_XOR(var194, var190, 15);
      var197 = __builtin_IMCE_AND(var194, var195, 15);
      var198 = __builtin_IMCE_AND(var196, var188, 15);
      var174 = __builtin_IMCE_OR(var197, var198, 15);
      var170 = __builtin_IMCE_ADD(var174, var162, 15);

      var199 = __builtin_IMCE_MULTL(var167, var159, 15);
      var200 = __builtin_IMCE_MULTH(var167, var159, 15);
      var201 = __builtin_IMCE_SUBI(0, 1);
      var202 = __builtin_IMCE_SRLI(var201, 1);
      var203 = __builtin_IMCE_SRAI(var200, 15);
      var204 = __builtin_IMCE_SRAI(var199, 15);
      var205 = __builtin_IMCE_XOR(var203, var204, 15);
      var206 = __builtin_IMCE_XOR(var203, var202, 15);
      var207 = __builtin_IMCE_XOR(var205, var201, 15);
      var208 = __builtin_IMCE_AND(var205, var206, 15);
      var209 = __builtin_IMCE_AND(var207, var199, 15);
      var175 = __builtin_IMCE_OR(var208, var209, 15);
      var171 = __builtin_IMCE_ADD(var175, var163, 15);

      var210 = __builtin_IMCE_MULTL(var168, var160, 15);
      var211 = __builtin_IMCE_MULTH(var168, var160, 15);
      var212 = __builtin_IMCE_SUBI(0, 1);
      var213 = __builtin_IMCE_SRLI(var212, 1);
      var214 = __builtin_IMCE_SRAI(var211, 15);
      var215 = __builtin_IMCE_SRAI(var210, 15);
      var216 = __builtin_IMCE_XOR(var214, var215, 15);
      var217 = __builtin_IMCE_XOR(var214, var213, 15);
      var218 = __builtin_IMCE_XOR(var216, var212, 15);
      var219 = __builtin_IMCE_AND(var216, var217, 15);
      var220 = __builtin_IMCE_AND(var218, var210, 15);
      var176 = __builtin_IMCE_OR(var219, var220, 15);
      var172 = __builtin_IMCE_ADD(var176, var164, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var169, 2, 0); // TensorEdge((101, odata), (109, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var170, 2, 0); // TensorEdge((101, odata), (109, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var171, 2, 0); // TensorEdge((101, odata), (109, func_out0), 0), imce_3_1 -> inode_3_0
      __builtin_IMCE_SEND(1, var172, 2, 0); // TensorEdge((101, odata), (109, func_out0), 0), imce_3_1 -> inode_3_0
      // endgenerate: bn_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 2) { // imce_3_2
    // generate: TensorEdge(((100, -68), config), ((100, 77), config)), config write

    __builtin_IMCE_RECV_CFG(1);
    // endgenerate: TensorEdge(((100, -68), config), ((100, 77), config)), config write
    // generate: conv exec10
    // generate: conv exec10_row_group0_outer_loop(iterate row offset)
    // generate : conv exec10_row_group0_outer_loop(iterate row offset). loop count == 1
    // generate: conv exec10_row_group0_col_group0
    // generate : conv exec10_row_group0_col_group0. loop count == 1

    // generate: load_block
    for (int i1 = 0; i1 < 10; i1++) { // generate : load_block
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((94, odata), ((100, 77), data), 2), imce_2_1 -> imce_3_2

      } // endgenerate
    } // endgenerate : load_block
    // endgenerate: load_block
    __builtin_IMCE_STEP();


    var25 = __builtin_IMCE_GET_CREG((short)0);
    var26 = __builtin_IMCE_GET_CREG((short)1);
    var27 = __builtin_IMCE_GET_CREG((short)2);
    var28 = __builtin_IMCE_GET_CREG((short)3);
    // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    var29 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    var30 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    var31 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    var32 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    // generate: add

    var222 = __builtin_IMCE_ADD(var25, var29, 15);
    var223 = __builtin_IMCE_ADD(var26, var30, 15);
    var224 = __builtin_IMCE_ADD(var27, var31, 15);
    var225 = __builtin_IMCE_ADD(var28, var32, 15);
    // endgenerate: add
    __builtin_IMCE_SEND(1, var222, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(1, var223, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(1, var224, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(1, var225, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
    // endgenerate : conv exec10_row_group0_col_group0
    // endgenerate: conv exec10_row_group0_col_group0
    // generate: conv exec10_row_group0_col_group1
    for (int i1 = 0; i1 < 6; i1++) { // generate : conv exec10_row_group0_col_group1

      // generate: load_block
      // generate : load_block. loop count == 1
      for (int i2 = 0; i2 < 4; i2++) { // generate
        __builtin_IMCE_LOAD_LB(0); // TensorEdge((94, odata), ((100, 77), data), 2), imce_2_1 -> imce_3_2

      } // endgenerate
      // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var25 = __builtin_IMCE_GET_CREG((short)0);
      var26 = __builtin_IMCE_GET_CREG((short)1);
      var27 = __builtin_IMCE_GET_CREG((short)2);
      var28 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      var29 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      var30 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      var31 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      var32 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // generate: add

      var222 = __builtin_IMCE_ADD(var25, var29, 15);
      var223 = __builtin_IMCE_ADD(var26, var30, 15);
      var224 = __builtin_IMCE_ADD(var27, var31, 15);
      var225 = __builtin_IMCE_ADD(var28, var32, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(1, var222, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var223, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var224, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var225, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
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
    // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    var29 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    var30 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    var31 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    var32 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
    // generate: add

    var222 = __builtin_IMCE_ADD(var25, var29, 15);
    var223 = __builtin_IMCE_ADD(var26, var30, 15);
    var224 = __builtin_IMCE_ADD(var27, var31, 15);
    var225 = __builtin_IMCE_ADD(var28, var32, 15);
    // endgenerate: add
    __builtin_IMCE_SEND(1, var222, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(1, var223, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(1, var224, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
    __builtin_IMCE_SEND(1, var225, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
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
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((94, odata), ((100, 77), data), 2), imce_2_1 -> imce_3_2

        } // endgenerate
      } // endgenerate : load_block
      // endgenerate: load_block
      __builtin_IMCE_STEP();


      var25 = __builtin_IMCE_GET_CREG((short)0);
      var26 = __builtin_IMCE_GET_CREG((short)1);
      var27 = __builtin_IMCE_GET_CREG((short)2);
      var28 = __builtin_IMCE_GET_CREG((short)3);
      // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      var29 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      var30 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      var31 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      var32 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // generate: add

      var222 = __builtin_IMCE_ADD(var25, var29, 15);
      var223 = __builtin_IMCE_ADD(var26, var30, 15);
      var224 = __builtin_IMCE_ADD(var27, var31, 15);
      var225 = __builtin_IMCE_ADD(var28, var32, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(1, var222, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var223, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var224, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var225, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
      // endgenerate : conv exec10_row_group1_col_group0
      // endgenerate: conv exec10_row_group1_col_group0
      // generate: conv exec10_row_group1_col_group1
      for (int i2 = 0; i2 < 6; i2++) { // generate : conv exec10_row_group1_col_group1

        // generate: load_block
        // generate : load_block. loop count == 1
        for (int i3 = 0; i3 < 4; i3++) { // generate
          __builtin_IMCE_LOAD_LB(0); // TensorEdge((94, odata), ((100, 77), data), 2), imce_2_1 -> imce_3_2

        } // endgenerate
        // endgenerate : load_block
        // endgenerate: load_block
        __builtin_IMCE_STEP();


        var25 = __builtin_IMCE_GET_CREG((short)0);
        var26 = __builtin_IMCE_GET_CREG((short)1);
        var27 = __builtin_IMCE_GET_CREG((short)2);
        var28 = __builtin_IMCE_GET_CREG((short)3);
        // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
        var29 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
        // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
        // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
        var30 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
        // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
        // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
        var31 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
        // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
        // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
        var32 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
        // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
        // generate: add

        var222 = __builtin_IMCE_ADD(var25, var29, 15);
        var223 = __builtin_IMCE_ADD(var26, var30, 15);
        var224 = __builtin_IMCE_ADD(var27, var31, 15);
        var225 = __builtin_IMCE_ADD(var28, var32, 15);
        // endgenerate: add
        __builtin_IMCE_SEND(1, var222, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
        __builtin_IMCE_SEND(1, var223, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
        __builtin_IMCE_SEND(1, var224, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
        __builtin_IMCE_SEND(1, var225, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
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
      // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      var29 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      var30 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      var31 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      var32 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // generate: add

      var222 = __builtin_IMCE_ADD(var25, var29, 15);
      var223 = __builtin_IMCE_ADD(var26, var30, 15);
      var224 = __builtin_IMCE_ADD(var27, var31, 15);
      var225 = __builtin_IMCE_ADD(var28, var32, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(1, var222, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var223, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var224, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var225, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
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
      // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      var29 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      var30 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      var31 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // generate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      var32 = __builtin_IMCE_RECV(2); // TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // endgenerate: TensorEdge(((99, 96), odata), ((100, 78), lhs)), imce_2_2 -> imce_3_2
      // generate: add

      var222 = __builtin_IMCE_ADD(var25, var29, 15);
      var223 = __builtin_IMCE_ADD(var26, var30, 15);
      var224 = __builtin_IMCE_ADD(var27, var31, 15);
      var225 = __builtin_IMCE_ADD(var28, var32, 15);
      // endgenerate: add
      __builtin_IMCE_SEND(1, var222, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var223, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var224, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
      __builtin_IMCE_SEND(1, var225, 2, 0); // TensorEdge(((100, 78), odata), (101, data)), imce_3_2 -> imce_3_1
    } // endgenerate : conv exec10_row_group2_col_group0
    // endgenerate: conv exec10_row_group2_col_group0
    // endgenerate : conv exec10_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec10_row_group2_outer_loop(iterate row offset)
    // endgenerate: conv exec10
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 3) { // imce_3_3
    // generate: TensorEdge((-99, fused_scale), (106, fused_scale)), fused_scale write

    var226 = __builtin_IMCE_RECV(1);
    var227 = __builtin_IMCE_RECV(1);
    var228 = __builtin_IMCE_RECV(1);
    var229 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge((-99, fused_scale), (106, fused_scale)), fused_scale write
    // generate: TensorEdge((-100, fused_bias), (106, fused_bias)), fused_bias write

    var230 = __builtin_IMCE_RECV(1);
    var231 = __builtin_IMCE_RECV(1);
    var232 = __builtin_IMCE_RECV(1);
    var233 = __builtin_IMCE_RECV(1);
    // endgenerate: TensorEdge((-100, fused_bias), (106, fused_bias)), fused_bias write
    // generate: call_created_loop
    for (int i1 = 0; i1 < 64; i1++) { // generate : call_created_loop
      // generate: bn_standalone

      var234 = __builtin_IMCE_RECV(2); // TensorEdge((105, odata), (106, data)), imce_2_3 -> imce_3_3
      var235 = __builtin_IMCE_RECV(2); // TensorEdge((105, odata), (106, data)), imce_2_3 -> imce_3_3
      var236 = __builtin_IMCE_RECV(2); // TensorEdge((105, odata), (106, data)), imce_2_3 -> imce_3_3
      var237 = __builtin_IMCE_RECV(2); // TensorEdge((105, odata), (106, data)), imce_2_3 -> imce_3_3
      // generate: batch_norm


      var246 = __builtin_IMCE_MULTL(var234, var226, 15);
      var247 = __builtin_IMCE_MULTH(var234, var226, 15);
      var248 = __builtin_IMCE_SUBI(0, 1);
      var249 = __builtin_IMCE_SRLI(var248, 1);
      var250 = __builtin_IMCE_SRAI(var247, 15);
      var251 = __builtin_IMCE_SRAI(var246, 15);
      var252 = __builtin_IMCE_XOR(var250, var251, 15);
      var253 = __builtin_IMCE_XOR(var250, var249, 15);
      var254 = __builtin_IMCE_XOR(var252, var248, 15);
      var255 = __builtin_IMCE_AND(var252, var253, 15);
      var256 = __builtin_IMCE_AND(var254, var246, 15);
      var242 = __builtin_IMCE_OR(var255, var256, 15);
      var238 = __builtin_IMCE_ADD(var242, var230, 15);

      var257 = __builtin_IMCE_MULTL(var235, var227, 15);
      var258 = __builtin_IMCE_MULTH(var235, var227, 15);
      var259 = __builtin_IMCE_SUBI(0, 1);
      var260 = __builtin_IMCE_SRLI(var259, 1);
      var261 = __builtin_IMCE_SRAI(var258, 15);
      var262 = __builtin_IMCE_SRAI(var257, 15);
      var263 = __builtin_IMCE_XOR(var261, var262, 15);
      var264 = __builtin_IMCE_XOR(var261, var260, 15);
      var265 = __builtin_IMCE_XOR(var263, var259, 15);
      var266 = __builtin_IMCE_AND(var263, var264, 15);
      var267 = __builtin_IMCE_AND(var265, var257, 15);
      var243 = __builtin_IMCE_OR(var266, var267, 15);
      var239 = __builtin_IMCE_ADD(var243, var231, 15);

      var268 = __builtin_IMCE_MULTL(var236, var228, 15);
      var269 = __builtin_IMCE_MULTH(var236, var228, 15);
      var270 = __builtin_IMCE_SUBI(0, 1);
      var271 = __builtin_IMCE_SRLI(var270, 1);
      var272 = __builtin_IMCE_SRAI(var269, 15);
      var273 = __builtin_IMCE_SRAI(var268, 15);
      var274 = __builtin_IMCE_XOR(var272, var273, 15);
      var275 = __builtin_IMCE_XOR(var272, var271, 15);
      var276 = __builtin_IMCE_XOR(var274, var270, 15);
      var277 = __builtin_IMCE_AND(var274, var275, 15);
      var278 = __builtin_IMCE_AND(var276, var268, 15);
      var244 = __builtin_IMCE_OR(var277, var278, 15);
      var240 = __builtin_IMCE_ADD(var244, var232, 15);

      var279 = __builtin_IMCE_MULTL(var237, var229, 15);
      var280 = __builtin_IMCE_MULTH(var237, var229, 15);
      var281 = __builtin_IMCE_SUBI(0, 1);
      var282 = __builtin_IMCE_SRLI(var281, 1);
      var283 = __builtin_IMCE_SRAI(var280, 15);
      var284 = __builtin_IMCE_SRAI(var279, 15);
      var285 = __builtin_IMCE_XOR(var283, var284, 15);
      var286 = __builtin_IMCE_XOR(var283, var282, 15);
      var287 = __builtin_IMCE_XOR(var285, var281, 15);
      var288 = __builtin_IMCE_AND(var285, var286, 15);
      var289 = __builtin_IMCE_AND(var287, var279, 15);
      var245 = __builtin_IMCE_OR(var288, var289, 15);
      var241 = __builtin_IMCE_ADD(var245, var233, 15);
      // endgenerate: batch_norm
      __builtin_IMCE_SEND(1, var238, 2, 0); // TensorEdge((106, odata), (107, lhs)), imce_3_3 -> imce_3_4
      __builtin_IMCE_SEND(1, var239, 2, 0); // TensorEdge((106, odata), (107, lhs)), imce_3_3 -> imce_3_4
      __builtin_IMCE_SEND(1, var240, 2, 0); // TensorEdge((106, odata), (107, lhs)), imce_3_3 -> imce_3_4
      __builtin_IMCE_SEND(1, var241, 2, 0); // TensorEdge((106, odata), (107, lhs)), imce_3_3 -> imce_3_4
      // endgenerate: bn_standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
  else if (hid == 3 && wid == 4) { // imce_3_4
    // generate: mult const

    var290 = __builtin_IMCE_RECV(1);
    var291 = __builtin_IMCE_RECV(1);
    var292 = __builtin_IMCE_RECV(1);
    var293 = __builtin_IMCE_RECV(1);
    // endgenerate: mult const
    // generate: call_created_loop
    for (int i1 = 0; i1 < 256; i1++) { // generate : call_created_loop
      // generate: multiply standalone

      var294 = __builtin_IMCE_RECV(2); // TensorEdge((106, odata), (107, lhs)), imce_3_3 -> imce_3_4
      // generate: multl


      var296 = __builtin_IMCE_MULTL(var290, var294, 15);
      var297 = __builtin_IMCE_MULTH(var290, var294, 15);
      var298 = __builtin_IMCE_SUBI(0, 1);
      var299 = __builtin_IMCE_SRLI(var298, 1);
      var300 = __builtin_IMCE_SRAI(var297, 15);
      var301 = __builtin_IMCE_SRAI(var296, 15);
      var302 = __builtin_IMCE_XOR(var300, var301, 15);
      var303 = __builtin_IMCE_XOR(var300, var299, 15);
      var304 = __builtin_IMCE_XOR(var302, var298, 15);
      var305 = __builtin_IMCE_AND(var302, var303, 15);
      var306 = __builtin_IMCE_AND(var304, var296, 15);
      var295 = __builtin_IMCE_OR(var305, var306, 15);
      // endgenerate: multl
      __builtin_IMCE_SEND(1, var295, 2, 0); // TensorEdge((107, odata), ((108, 102), lhs)), imce_3_4 -> imce_2_4
      // endgenerate: multiply standalone
    } // endgenerate : call_created_loop
    // endgenerate: call_created_loop
    __builtin_IMCE_STOP();
  }
}
