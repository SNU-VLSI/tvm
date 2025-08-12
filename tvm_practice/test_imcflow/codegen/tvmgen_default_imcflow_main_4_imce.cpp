void tvmgen_default_imcflow_main_4() {
  int hid = __builtin_IMCE_GET_CORE_HID();
  int wid = __builtin_IMCE_GET_CORE_WID();

  short16 var1;
  short16 var2;
  short16 var3;
  short16 var4;
  short16 var5;
  short16 var6;
  short16 var7;
  short16 var8;
  short16 var9;
  short16 var10;
  short16 var11;
  short16 var12;
  short16 var13;
  short16 var14;
  short16 var15;
  short16 var16;
  short16 var17;
  short16 var18;
  short16 var19;
  short16 var20;
  short16 var21;
  short16 var22;
  short16 var23;
  short16 var24;
  short16 var25;
  short16 var26;
  short16 var27;
  short16 var28;
  short16 var29;
  short16 var30;
  short16 var31;
  short16 var32;
  short16 var33;
  short16 var34;
  short16 var35;
  short16 var36;
  short16 var37;
  short16 var38;
  short16 var39;
  short16 var40;
  short16 var41;
  short16 var42;
  short16 var43;
  short16 var44;
  short16 var45;
  short16 var46;
  short16 var47;
  short16 var48;
  short16 var49;
  short16 var50;
  short16 var51;
  short16 var52;
  short16 var53;
  short16 var54;
  short16 var55;
  short16 var56;
  short16 var57;
  short16 var58;
  short16 var59;
  short16 var60;
  short16 var61;
  short16 var62;
  short16 var63;
  short16 var64;
  short16 var65;
  short16 var66;
  short16 var67;
  short16 var68;
  short16 var69;
  short16 var70;
  short16 var71;
  short16 var72;
  short16 var73;
  short16 var74;
  short16 var75;
  short16 var76;
  short16 var77;
  short16 var78;
  short16 var79;
  short16 var80;
  short16 var81;
  short16 var82;
  short16 var83;
  short16 var84;
  short16 var85;
  short16 var86;
  short16 var87;
  short16 var88;
  short16 var89;
  short16 var90;
  short16 var91;
  short16 var92;
  short16 var93;
  short16 var94;
  short16 var95;
  short16 var96;
  short16 var97;
  short16 var98;
  short16 var99;
  short16 var100;
  short16 var101;
  short16 var102;
  short16 var103;
  short16 var104;
  short16 var105;
  short16 var106;
  short16 var107;
  short16 var108;
  short16 var109;
  short16 var110;
  short16 var111;
  short16 var112;
  short16 var113;
  short16 var114;
  short16 var115;
  short16 var116;
  short16 var117;
  short16 var118;
  short16 var119;
  short16 var120;
  short16 var121;
  short16 var122;
  short16 var123;
  short16 var124;
  short16 var125;
  short16 var126;
  short16 var127;
  short16 var128;
  short16 var129;
  short16 var130;
  short16 var131;
  short16 var132;
  short16 var133;
  short16 var134;
  short16 var135;
  short16 var136;
  short16 var137;
  short16 var138;
  short16 var139;
  short16 var140;
  short16 var141;
  short16 var142;
  short16 var143;
  short16 var144;
  short16 var145;
  short16 var146;
  short16 var147;
  short16 var148;
  short16 var149;
  short16 var150;
  short16 var151;
  short16 var152;
  short16 var153;
  short16 var154;
  short16 var155;
  short16 var156;
  short16 var157;
  short16 var158;
  short16 var159;
  short16 var160;
  short16 var161;
  short16 var162;
  short16 var163;
  short16 var164;
  short16 var165;
  short16 var166;
  short16 var167;
  short16 var168;
  short16 var169;
  short16 var170;
  short16 var171;
  short16 var172;
  short16 var173;
  short16 var174;
  short16 var175;
  short16 var176;
  short16 var177;
  short16 var178;
  short16 var179;
  short16 var180;
  short16 var181;
  short16 var182;
  short16 var183;
  short16 var184;
  short16 var185;
  short16 var186;
  short16 var187;
  short16 var188;
  short16 var189;
  short16 var190;
  short16 var191;
  short16 var192;
  short16 var193;
  short16 var194;
  short16 var195;
  short16 var196;
  short16 var197;
  short16 var198;
  short16 var199;
  short16 var200;
  short16 var201;
  short16 var202;
  short16 var203;
  short16 var204;
  short16 var205;
  short16 var206;
  short16 var207;
  short16 var208;
  short16 var209;
  short16 var210;
  short16 var211;
  short16 var212;
  short16 var213;
  short16 var214;
  short16 var215;
  short16 var216;
  short16 var217;
  short16 var218;
  short16 var219;
  short16 var220;
  short16 var221;
  short16 var222;
  short16 var223;
  short16 var224;
  short16 var225;
  short16 var226;
  short16 var227;
  short16 var228;

if (hid == 0 && wid == 1) {
}
else if (hid == 0 && wid == 2) {
}
else if (hid == 0 && wid == 3) {
  // generate: weight write
  for (int i1 = 0; i1 < 8192; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(1);
  } // endgenerate: load_block
  // endgenerate: weight write
  // generate: min write
  var1 = __builtin_IMCE_RECV(1);
  // endgenerate: min write
  // generate: max write
  var2 = __builtin_IMCE_RECV(1);
  // endgenerate: max write
  // generate: conv exec
  for (int i1 = 0; i1 < 18; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
  } // endgenerate: load_block
  __builtin_IMCE_STEP();

  var3 = __builtin_IMCE_GET_CREG((short)0);
  var4 = __builtin_IMCE_GET_CREG((short)1);
  var5 = __builtin_IMCE_GET_CREG((short)2);
  var6 = __builtin_IMCE_GET_CREG((short)3);


  // generate: div
  var11 = __builtin_IMCE_RECV(1);
  var12 = __builtin_IMCE_DIV(var3, var11, 15);
  var13 = __builtin_IMCE_RECV(1);
  var14 = __builtin_IMCE_DIV(var4, var13, 15);
  var15 = __builtin_IMCE_RECV(1);
  var16 = __builtin_IMCE_DIV(var5, var15, 15);
  var17 = __builtin_IMCE_RECV(1);
  var18 = __builtin_IMCE_DIV(var6, var17, 15);
  // endgenerate: div


  // generate: add
  var19 = __builtin_IMCE_RECV(3);
  var20 = __builtin_IMCE_ADD(var19, var12, 15);
  var21 = __builtin_IMCE_RECV(3);
  var22 = __builtin_IMCE_ADD(var21, var14, 15);
  var23 = __builtin_IMCE_RECV(3);
  var24 = __builtin_IMCE_ADD(var23, var16, 15);
  var25 = __builtin_IMCE_RECV(3);
  var26 = __builtin_IMCE_ADD(var25, var18, 15);
  // endgenerate: add


  // generate: min_max_quantize
  __builtin_IMCE_MM_QUANT(var27, 0, 15, 4);
  var28 = __builtin_IMCE_GET_QREG(0);
  __builtin_IMCE_MM_QUANT(var29, 0, 15, 5);
  var30 = __builtin_IMCE_GET_QREG(1);
  __builtin_IMCE_MM_QUANT(var31, 0, 15, 6);
  var32 = __builtin_IMCE_GET_QREG(2);
  __builtin_IMCE_MM_QUANT(var33, 0, 15, 7);
  var34 = __builtin_IMCE_GET_QREG(3);
  // endgenerate: min_max_quantize


  // generate: concat
  var35 = __builtin_IMCE_RECV(2);
  var7 = __builtin_IMCE_OR(var28, var35, 15);
  var36 = __builtin_IMCE_RECV(2);
  var8 = __builtin_IMCE_OR(var30, var36, 15);
  var37 = __builtin_IMCE_RECV(2);
  var9 = __builtin_IMCE_OR(var32, var37, 15);
  var38 = __builtin_IMCE_RECV(2);
  var10 = __builtin_IMCE_OR(var34, var38, 15);
  // endgenerate: concat


  __builtin_IMCE_SEND(0, var7, 2, 0);
  __builtin_IMCE_SEND(0, var8, 2, 0);
  __builtin_IMCE_SEND(0, var9, 2, 0);
  __builtin_IMCE_SEND(0, var10, 2, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: inner_loop
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_STEP();

    var3 = __builtin_IMCE_GET_CREG((short)0);
    var4 = __builtin_IMCE_GET_CREG((short)1);
    var5 = __builtin_IMCE_GET_CREG((short)2);
    var6 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var11 = __builtin_IMCE_RECV(1);
    var12 = __builtin_IMCE_DIV(var3, var11, 15);
    var13 = __builtin_IMCE_RECV(1);
    var14 = __builtin_IMCE_DIV(var4, var13, 15);
    var15 = __builtin_IMCE_RECV(1);
    var16 = __builtin_IMCE_DIV(var5, var15, 15);
    var17 = __builtin_IMCE_RECV(1);
    var18 = __builtin_IMCE_DIV(var6, var17, 15);
    // endgenerate: div


    // generate: add
    var19 = __builtin_IMCE_RECV(3);
    var20 = __builtin_IMCE_ADD(var19, var12, 15);
    var21 = __builtin_IMCE_RECV(3);
    var22 = __builtin_IMCE_ADD(var21, var14, 15);
    var23 = __builtin_IMCE_RECV(3);
    var24 = __builtin_IMCE_ADD(var23, var16, 15);
    var25 = __builtin_IMCE_RECV(3);
    var26 = __builtin_IMCE_ADD(var25, var18, 15);
    // endgenerate: add


    // generate: min_max_quantize
    __builtin_IMCE_MM_QUANT(var27, 0, 15, 4);
    var28 = __builtin_IMCE_GET_QREG(0);
    __builtin_IMCE_MM_QUANT(var29, 0, 15, 5);
    var30 = __builtin_IMCE_GET_QREG(1);
    __builtin_IMCE_MM_QUANT(var31, 0, 15, 6);
    var32 = __builtin_IMCE_GET_QREG(2);
    __builtin_IMCE_MM_QUANT(var33, 0, 15, 7);
    var34 = __builtin_IMCE_GET_QREG(3);
    // endgenerate: min_max_quantize


    // generate: concat
    var35 = __builtin_IMCE_RECV(2);
    var7 = __builtin_IMCE_OR(var28, var35, 15);
    var36 = __builtin_IMCE_RECV(2);
    var8 = __builtin_IMCE_OR(var30, var36, 15);
    var37 = __builtin_IMCE_RECV(2);
    var9 = __builtin_IMCE_OR(var32, var37, 15);
    var38 = __builtin_IMCE_RECV(2);
    var10 = __builtin_IMCE_OR(var34, var38, 15);
    // endgenerate: concat


    __builtin_IMCE_SEND(0, var7, 2, 0);
    __builtin_IMCE_SEND(0, var8, 2, 0);
    __builtin_IMCE_SEND(0, var9, 2, 0);
    __builtin_IMCE_SEND(0, var10, 2, 0);


  } // endgenerate: inner_loop

  __builtin_IMCE_STEP();

  var3 = __builtin_IMCE_GET_CREG((short)0);
  var4 = __builtin_IMCE_GET_CREG((short)1);
  var5 = __builtin_IMCE_GET_CREG((short)2);
  var6 = __builtin_IMCE_GET_CREG((short)3);


  // generate: div
  var11 = __builtin_IMCE_RECV(1);
  var12 = __builtin_IMCE_DIV(var3, var11, 15);
  var13 = __builtin_IMCE_RECV(1);
  var14 = __builtin_IMCE_DIV(var4, var13, 15);
  var15 = __builtin_IMCE_RECV(1);
  var16 = __builtin_IMCE_DIV(var5, var15, 15);
  var17 = __builtin_IMCE_RECV(1);
  var18 = __builtin_IMCE_DIV(var6, var17, 15);
  // endgenerate: div


  // generate: add
  var19 = __builtin_IMCE_RECV(3);
  var20 = __builtin_IMCE_ADD(var19, var12, 15);
  var21 = __builtin_IMCE_RECV(3);
  var22 = __builtin_IMCE_ADD(var21, var14, 15);
  var23 = __builtin_IMCE_RECV(3);
  var24 = __builtin_IMCE_ADD(var23, var16, 15);
  var25 = __builtin_IMCE_RECV(3);
  var26 = __builtin_IMCE_ADD(var25, var18, 15);
  // endgenerate: add


  // generate: min_max_quantize
  __builtin_IMCE_MM_QUANT(var27, 0, 15, 4);
  var28 = __builtin_IMCE_GET_QREG(0);
  __builtin_IMCE_MM_QUANT(var29, 0, 15, 5);
  var30 = __builtin_IMCE_GET_QREG(1);
  __builtin_IMCE_MM_QUANT(var31, 0, 15, 6);
  var32 = __builtin_IMCE_GET_QREG(2);
  __builtin_IMCE_MM_QUANT(var33, 0, 15, 7);
  var34 = __builtin_IMCE_GET_QREG(3);
  // endgenerate: min_max_quantize


  // generate: concat
  var35 = __builtin_IMCE_RECV(2);
  var7 = __builtin_IMCE_OR(var28, var35, 15);
  var36 = __builtin_IMCE_RECV(2);
  var8 = __builtin_IMCE_OR(var30, var36, 15);
  var37 = __builtin_IMCE_RECV(2);
  var9 = __builtin_IMCE_OR(var32, var37, 15);
  var38 = __builtin_IMCE_RECV(2);
  var10 = __builtin_IMCE_OR(var34, var38, 15);
  // endgenerate: concat


  __builtin_IMCE_SEND(0, var7, 2, 0);
  __builtin_IMCE_SEND(0, var8, 2, 0);
  __builtin_IMCE_SEND(0, var9, 2, 0);
  __builtin_IMCE_SEND(0, var10, 2, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: outer_loop
    for (int i2 = 0; i2 < 2; i2++) { // generate: load_block
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
    } // endgenerate: load_block
    __builtin_IMCE_STEP();

    var3 = __builtin_IMCE_GET_CREG((short)0);
    var4 = __builtin_IMCE_GET_CREG((short)1);
    var5 = __builtin_IMCE_GET_CREG((short)2);
    var6 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var11 = __builtin_IMCE_RECV(1);
    var12 = __builtin_IMCE_DIV(var3, var11, 15);
    var13 = __builtin_IMCE_RECV(1);
    var14 = __builtin_IMCE_DIV(var4, var13, 15);
    var15 = __builtin_IMCE_RECV(1);
    var16 = __builtin_IMCE_DIV(var5, var15, 15);
    var17 = __builtin_IMCE_RECV(1);
    var18 = __builtin_IMCE_DIV(var6, var17, 15);
    // endgenerate: div


    // generate: add
    var19 = __builtin_IMCE_RECV(3);
    var20 = __builtin_IMCE_ADD(var19, var12, 15);
    var21 = __builtin_IMCE_RECV(3);
    var22 = __builtin_IMCE_ADD(var21, var14, 15);
    var23 = __builtin_IMCE_RECV(3);
    var24 = __builtin_IMCE_ADD(var23, var16, 15);
    var25 = __builtin_IMCE_RECV(3);
    var26 = __builtin_IMCE_ADD(var25, var18, 15);
    // endgenerate: add


    // generate: min_max_quantize
    __builtin_IMCE_MM_QUANT(var27, 0, 15, 4);
    var28 = __builtin_IMCE_GET_QREG(0);
    __builtin_IMCE_MM_QUANT(var29, 0, 15, 5);
    var30 = __builtin_IMCE_GET_QREG(1);
    __builtin_IMCE_MM_QUANT(var31, 0, 15, 6);
    var32 = __builtin_IMCE_GET_QREG(2);
    __builtin_IMCE_MM_QUANT(var33, 0, 15, 7);
    var34 = __builtin_IMCE_GET_QREG(3);
    // endgenerate: min_max_quantize


    // generate: concat
    var35 = __builtin_IMCE_RECV(2);
    var7 = __builtin_IMCE_OR(var28, var35, 15);
    var36 = __builtin_IMCE_RECV(2);
    var8 = __builtin_IMCE_OR(var30, var36, 15);
    var37 = __builtin_IMCE_RECV(2);
    var9 = __builtin_IMCE_OR(var32, var37, 15);
    var38 = __builtin_IMCE_RECV(2);
    var10 = __builtin_IMCE_OR(var34, var38, 15);
    // endgenerate: concat


    __builtin_IMCE_SEND(0, var7, 2, 0);
    __builtin_IMCE_SEND(0, var8, 2, 0);
    __builtin_IMCE_SEND(0, var9, 2, 0);
    __builtin_IMCE_SEND(0, var10, 2, 0);


    for (int i2 = 0; i2 < 14; i2++) { // generate: inner_loop
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_STEP();

      var3 = __builtin_IMCE_GET_CREG((short)0);
      var4 = __builtin_IMCE_GET_CREG((short)1);
      var5 = __builtin_IMCE_GET_CREG((short)2);
      var6 = __builtin_IMCE_GET_CREG((short)3);


      // generate: div
      var11 = __builtin_IMCE_RECV(1);
      var12 = __builtin_IMCE_DIV(var3, var11, 15);
      var13 = __builtin_IMCE_RECV(1);
      var14 = __builtin_IMCE_DIV(var4, var13, 15);
      var15 = __builtin_IMCE_RECV(1);
      var16 = __builtin_IMCE_DIV(var5, var15, 15);
      var17 = __builtin_IMCE_RECV(1);
      var18 = __builtin_IMCE_DIV(var6, var17, 15);
      // endgenerate: div


      // generate: add
      var19 = __builtin_IMCE_RECV(3);
      var20 = __builtin_IMCE_ADD(var19, var12, 15);
      var21 = __builtin_IMCE_RECV(3);
      var22 = __builtin_IMCE_ADD(var21, var14, 15);
      var23 = __builtin_IMCE_RECV(3);
      var24 = __builtin_IMCE_ADD(var23, var16, 15);
      var25 = __builtin_IMCE_RECV(3);
      var26 = __builtin_IMCE_ADD(var25, var18, 15);
      // endgenerate: add


      // generate: min_max_quantize
      __builtin_IMCE_MM_QUANT(var27, 0, 15, 4);
      var28 = __builtin_IMCE_GET_QREG(0);
      __builtin_IMCE_MM_QUANT(var29, 0, 15, 5);
      var30 = __builtin_IMCE_GET_QREG(1);
      __builtin_IMCE_MM_QUANT(var31, 0, 15, 6);
      var32 = __builtin_IMCE_GET_QREG(2);
      __builtin_IMCE_MM_QUANT(var33, 0, 15, 7);
      var34 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize


      // generate: concat
      var35 = __builtin_IMCE_RECV(2);
      var7 = __builtin_IMCE_OR(var28, var35, 15);
      var36 = __builtin_IMCE_RECV(2);
      var8 = __builtin_IMCE_OR(var30, var36, 15);
      var37 = __builtin_IMCE_RECV(2);
      var9 = __builtin_IMCE_OR(var32, var37, 15);
      var38 = __builtin_IMCE_RECV(2);
      var10 = __builtin_IMCE_OR(var34, var38, 15);
      // endgenerate: concat


      __builtin_IMCE_SEND(0, var7, 2, 0);
      __builtin_IMCE_SEND(0, var8, 2, 0);
      __builtin_IMCE_SEND(0, var9, 2, 0);
      __builtin_IMCE_SEND(0, var10, 2, 0);


    } // endgenerate: inner_loop

    __builtin_IMCE_STEP();

    var3 = __builtin_IMCE_GET_CREG((short)0);
    var4 = __builtin_IMCE_GET_CREG((short)1);
    var5 = __builtin_IMCE_GET_CREG((short)2);
    var6 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var11 = __builtin_IMCE_RECV(1);
    var12 = __builtin_IMCE_DIV(var3, var11, 15);
    var13 = __builtin_IMCE_RECV(1);
    var14 = __builtin_IMCE_DIV(var4, var13, 15);
    var15 = __builtin_IMCE_RECV(1);
    var16 = __builtin_IMCE_DIV(var5, var15, 15);
    var17 = __builtin_IMCE_RECV(1);
    var18 = __builtin_IMCE_DIV(var6, var17, 15);
    // endgenerate: div


    // generate: add
    var19 = __builtin_IMCE_RECV(3);
    var20 = __builtin_IMCE_ADD(var19, var12, 15);
    var21 = __builtin_IMCE_RECV(3);
    var22 = __builtin_IMCE_ADD(var21, var14, 15);
    var23 = __builtin_IMCE_RECV(3);
    var24 = __builtin_IMCE_ADD(var23, var16, 15);
    var25 = __builtin_IMCE_RECV(3);
    var26 = __builtin_IMCE_ADD(var25, var18, 15);
    // endgenerate: add


    // generate: min_max_quantize
    __builtin_IMCE_MM_QUANT(var27, 0, 15, 4);
    var28 = __builtin_IMCE_GET_QREG(0);
    __builtin_IMCE_MM_QUANT(var29, 0, 15, 5);
    var30 = __builtin_IMCE_GET_QREG(1);
    __builtin_IMCE_MM_QUANT(var31, 0, 15, 6);
    var32 = __builtin_IMCE_GET_QREG(2);
    __builtin_IMCE_MM_QUANT(var33, 0, 15, 7);
    var34 = __builtin_IMCE_GET_QREG(3);
    // endgenerate: min_max_quantize


    // generate: concat
    var35 = __builtin_IMCE_RECV(2);
    var7 = __builtin_IMCE_OR(var28, var35, 15);
    var36 = __builtin_IMCE_RECV(2);
    var8 = __builtin_IMCE_OR(var30, var36, 15);
    var37 = __builtin_IMCE_RECV(2);
    var9 = __builtin_IMCE_OR(var32, var37, 15);
    var38 = __builtin_IMCE_RECV(2);
    var10 = __builtin_IMCE_OR(var34, var38, 15);
    // endgenerate: concat


    __builtin_IMCE_SEND(0, var7, 2, 0);
    __builtin_IMCE_SEND(0, var8, 2, 0);
    __builtin_IMCE_SEND(0, var9, 2, 0);
    __builtin_IMCE_SEND(0, var10, 2, 0);


  } // endgenerate: outer_loop
  for (int i1 = 0; i1 < 16; i1++) { // generate: inner_loop

    __builtin_IMCE_STEP();

    var3 = __builtin_IMCE_GET_CREG((short)0);
    var4 = __builtin_IMCE_GET_CREG((short)1);
    var5 = __builtin_IMCE_GET_CREG((short)2);
    var6 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var11 = __builtin_IMCE_RECV(1);
    var12 = __builtin_IMCE_DIV(var3, var11, 15);
    var13 = __builtin_IMCE_RECV(1);
    var14 = __builtin_IMCE_DIV(var4, var13, 15);
    var15 = __builtin_IMCE_RECV(1);
    var16 = __builtin_IMCE_DIV(var5, var15, 15);
    var17 = __builtin_IMCE_RECV(1);
    var18 = __builtin_IMCE_DIV(var6, var17, 15);
    // endgenerate: div


    // generate: add
    var19 = __builtin_IMCE_RECV(3);
    var20 = __builtin_IMCE_ADD(var19, var12, 15);
    var21 = __builtin_IMCE_RECV(3);
    var22 = __builtin_IMCE_ADD(var21, var14, 15);
    var23 = __builtin_IMCE_RECV(3);
    var24 = __builtin_IMCE_ADD(var23, var16, 15);
    var25 = __builtin_IMCE_RECV(3);
    var26 = __builtin_IMCE_ADD(var25, var18, 15);
    // endgenerate: add


    // generate: min_max_quantize
    __builtin_IMCE_MM_QUANT(var27, 0, 15, 4);
    var28 = __builtin_IMCE_GET_QREG(0);
    __builtin_IMCE_MM_QUANT(var29, 0, 15, 5);
    var30 = __builtin_IMCE_GET_QREG(1);
    __builtin_IMCE_MM_QUANT(var31, 0, 15, 6);
    var32 = __builtin_IMCE_GET_QREG(2);
    __builtin_IMCE_MM_QUANT(var33, 0, 15, 7);
    var34 = __builtin_IMCE_GET_QREG(3);
    // endgenerate: min_max_quantize


    // generate: concat
    var35 = __builtin_IMCE_RECV(2);
    var7 = __builtin_IMCE_OR(var28, var35, 15);
    var36 = __builtin_IMCE_RECV(2);
    var8 = __builtin_IMCE_OR(var30, var36, 15);
    var37 = __builtin_IMCE_RECV(2);
    var9 = __builtin_IMCE_OR(var32, var37, 15);
    var38 = __builtin_IMCE_RECV(2);
    var10 = __builtin_IMCE_OR(var34, var38, 15);
    // endgenerate: concat


    __builtin_IMCE_SEND(0, var7, 2, 0);
    __builtin_IMCE_SEND(0, var8, 2, 0);
    __builtin_IMCE_SEND(0, var9, 2, 0);
    __builtin_IMCE_SEND(0, var10, 2, 0);


  } // endgenerate: inner_loop
  // endgenerate: conv exec
}
else if (hid == 0 && wid == 4) {
  // generate: weight write
  for (int i1 = 0; i1 < 8192; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(1);
  } // endgenerate: load_block
  // endgenerate: weight write
  // generate: conv exec
  for (int i1 = 0; i1 < 18; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
  } // endgenerate: load_block
  __builtin_IMCE_STEP();

  var39 = __builtin_IMCE_GET_CREG((short)0);
  var40 = __builtin_IMCE_GET_CREG((short)1);
  var41 = __builtin_IMCE_GET_CREG((short)2);
  var42 = __builtin_IMCE_GET_CREG((short)3);


  // generate: div
  var43 = __builtin_IMCE_RECV(1);
  var44 = __builtin_IMCE_DIV(var39, var43, 15);
  var45 = __builtin_IMCE_RECV(1);
  var46 = __builtin_IMCE_DIV(var40, var45, 15);
  var47 = __builtin_IMCE_RECV(1);
  var48 = __builtin_IMCE_DIV(var41, var47, 15);
  var49 = __builtin_IMCE_RECV(1);
  var50 = __builtin_IMCE_DIV(var42, var49, 15);
  // endgenerate: div


  // generate: add
  var51 = __builtin_IMCE_RECV(2);
  var19 = __builtin_IMCE_ADD(var51, var44, 15);
  var52 = __builtin_IMCE_RECV(2);
  var21 = __builtin_IMCE_ADD(var52, var46, 15);
  var53 = __builtin_IMCE_RECV(2);
  var23 = __builtin_IMCE_ADD(var53, var48, 15);
  var54 = __builtin_IMCE_RECV(2);
  var25 = __builtin_IMCE_ADD(var54, var50, 15);
  // endgenerate: add


  __builtin_IMCE_SEND(0, var19, 3, 0);
  __builtin_IMCE_SEND(0, var21, 3, 0);
  __builtin_IMCE_SEND(0, var23, 3, 0);
  __builtin_IMCE_SEND(0, var25, 3, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: inner_loop
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_STEP();

    var39 = __builtin_IMCE_GET_CREG((short)0);
    var40 = __builtin_IMCE_GET_CREG((short)1);
    var41 = __builtin_IMCE_GET_CREG((short)2);
    var42 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var43 = __builtin_IMCE_RECV(1);
    var44 = __builtin_IMCE_DIV(var39, var43, 15);
    var45 = __builtin_IMCE_RECV(1);
    var46 = __builtin_IMCE_DIV(var40, var45, 15);
    var47 = __builtin_IMCE_RECV(1);
    var48 = __builtin_IMCE_DIV(var41, var47, 15);
    var49 = __builtin_IMCE_RECV(1);
    var50 = __builtin_IMCE_DIV(var42, var49, 15);
    // endgenerate: div


    // generate: add
    var51 = __builtin_IMCE_RECV(2);
    var19 = __builtin_IMCE_ADD(var51, var44, 15);
    var52 = __builtin_IMCE_RECV(2);
    var21 = __builtin_IMCE_ADD(var52, var46, 15);
    var53 = __builtin_IMCE_RECV(2);
    var23 = __builtin_IMCE_ADD(var53, var48, 15);
    var54 = __builtin_IMCE_RECV(2);
    var25 = __builtin_IMCE_ADD(var54, var50, 15);
    // endgenerate: add


    __builtin_IMCE_SEND(0, var19, 3, 0);
    __builtin_IMCE_SEND(0, var21, 3, 0);
    __builtin_IMCE_SEND(0, var23, 3, 0);
    __builtin_IMCE_SEND(0, var25, 3, 0);


  } // endgenerate: inner_loop

  __builtin_IMCE_STEP();

  var39 = __builtin_IMCE_GET_CREG((short)0);
  var40 = __builtin_IMCE_GET_CREG((short)1);
  var41 = __builtin_IMCE_GET_CREG((short)2);
  var42 = __builtin_IMCE_GET_CREG((short)3);


  // generate: div
  var43 = __builtin_IMCE_RECV(1);
  var44 = __builtin_IMCE_DIV(var39, var43, 15);
  var45 = __builtin_IMCE_RECV(1);
  var46 = __builtin_IMCE_DIV(var40, var45, 15);
  var47 = __builtin_IMCE_RECV(1);
  var48 = __builtin_IMCE_DIV(var41, var47, 15);
  var49 = __builtin_IMCE_RECV(1);
  var50 = __builtin_IMCE_DIV(var42, var49, 15);
  // endgenerate: div


  // generate: add
  var51 = __builtin_IMCE_RECV(2);
  var19 = __builtin_IMCE_ADD(var51, var44, 15);
  var52 = __builtin_IMCE_RECV(2);
  var21 = __builtin_IMCE_ADD(var52, var46, 15);
  var53 = __builtin_IMCE_RECV(2);
  var23 = __builtin_IMCE_ADD(var53, var48, 15);
  var54 = __builtin_IMCE_RECV(2);
  var25 = __builtin_IMCE_ADD(var54, var50, 15);
  // endgenerate: add


  __builtin_IMCE_SEND(0, var19, 3, 0);
  __builtin_IMCE_SEND(0, var21, 3, 0);
  __builtin_IMCE_SEND(0, var23, 3, 0);
  __builtin_IMCE_SEND(0, var25, 3, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: outer_loop
    for (int i2 = 0; i2 < 2; i2++) { // generate: load_block
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
    } // endgenerate: load_block
    __builtin_IMCE_STEP();

    var39 = __builtin_IMCE_GET_CREG((short)0);
    var40 = __builtin_IMCE_GET_CREG((short)1);
    var41 = __builtin_IMCE_GET_CREG((short)2);
    var42 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var43 = __builtin_IMCE_RECV(1);
    var44 = __builtin_IMCE_DIV(var39, var43, 15);
    var45 = __builtin_IMCE_RECV(1);
    var46 = __builtin_IMCE_DIV(var40, var45, 15);
    var47 = __builtin_IMCE_RECV(1);
    var48 = __builtin_IMCE_DIV(var41, var47, 15);
    var49 = __builtin_IMCE_RECV(1);
    var50 = __builtin_IMCE_DIV(var42, var49, 15);
    // endgenerate: div


    // generate: add
    var51 = __builtin_IMCE_RECV(2);
    var19 = __builtin_IMCE_ADD(var51, var44, 15);
    var52 = __builtin_IMCE_RECV(2);
    var21 = __builtin_IMCE_ADD(var52, var46, 15);
    var53 = __builtin_IMCE_RECV(2);
    var23 = __builtin_IMCE_ADD(var53, var48, 15);
    var54 = __builtin_IMCE_RECV(2);
    var25 = __builtin_IMCE_ADD(var54, var50, 15);
    // endgenerate: add


    __builtin_IMCE_SEND(0, var19, 3, 0);
    __builtin_IMCE_SEND(0, var21, 3, 0);
    __builtin_IMCE_SEND(0, var23, 3, 0);
    __builtin_IMCE_SEND(0, var25, 3, 0);


    for (int i2 = 0; i2 < 14; i2++) { // generate: inner_loop
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_STEP();

      var39 = __builtin_IMCE_GET_CREG((short)0);
      var40 = __builtin_IMCE_GET_CREG((short)1);
      var41 = __builtin_IMCE_GET_CREG((short)2);
      var42 = __builtin_IMCE_GET_CREG((short)3);


      // generate: div
      var43 = __builtin_IMCE_RECV(1);
      var44 = __builtin_IMCE_DIV(var39, var43, 15);
      var45 = __builtin_IMCE_RECV(1);
      var46 = __builtin_IMCE_DIV(var40, var45, 15);
      var47 = __builtin_IMCE_RECV(1);
      var48 = __builtin_IMCE_DIV(var41, var47, 15);
      var49 = __builtin_IMCE_RECV(1);
      var50 = __builtin_IMCE_DIV(var42, var49, 15);
      // endgenerate: div


      // generate: add
      var51 = __builtin_IMCE_RECV(2);
      var19 = __builtin_IMCE_ADD(var51, var44, 15);
      var52 = __builtin_IMCE_RECV(2);
      var21 = __builtin_IMCE_ADD(var52, var46, 15);
      var53 = __builtin_IMCE_RECV(2);
      var23 = __builtin_IMCE_ADD(var53, var48, 15);
      var54 = __builtin_IMCE_RECV(2);
      var25 = __builtin_IMCE_ADD(var54, var50, 15);
      // endgenerate: add


      __builtin_IMCE_SEND(0, var19, 3, 0);
      __builtin_IMCE_SEND(0, var21, 3, 0);
      __builtin_IMCE_SEND(0, var23, 3, 0);
      __builtin_IMCE_SEND(0, var25, 3, 0);


    } // endgenerate: inner_loop

    __builtin_IMCE_STEP();

    var39 = __builtin_IMCE_GET_CREG((short)0);
    var40 = __builtin_IMCE_GET_CREG((short)1);
    var41 = __builtin_IMCE_GET_CREG((short)2);
    var42 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var43 = __builtin_IMCE_RECV(1);
    var44 = __builtin_IMCE_DIV(var39, var43, 15);
    var45 = __builtin_IMCE_RECV(1);
    var46 = __builtin_IMCE_DIV(var40, var45, 15);
    var47 = __builtin_IMCE_RECV(1);
    var48 = __builtin_IMCE_DIV(var41, var47, 15);
    var49 = __builtin_IMCE_RECV(1);
    var50 = __builtin_IMCE_DIV(var42, var49, 15);
    // endgenerate: div


    // generate: add
    var51 = __builtin_IMCE_RECV(2);
    var19 = __builtin_IMCE_ADD(var51, var44, 15);
    var52 = __builtin_IMCE_RECV(2);
    var21 = __builtin_IMCE_ADD(var52, var46, 15);
    var53 = __builtin_IMCE_RECV(2);
    var23 = __builtin_IMCE_ADD(var53, var48, 15);
    var54 = __builtin_IMCE_RECV(2);
    var25 = __builtin_IMCE_ADD(var54, var50, 15);
    // endgenerate: add


    __builtin_IMCE_SEND(0, var19, 3, 0);
    __builtin_IMCE_SEND(0, var21, 3, 0);
    __builtin_IMCE_SEND(0, var23, 3, 0);
    __builtin_IMCE_SEND(0, var25, 3, 0);


  } // endgenerate: outer_loop
  for (int i1 = 0; i1 < 16; i1++) { // generate: inner_loop

    __builtin_IMCE_STEP();

    var39 = __builtin_IMCE_GET_CREG((short)0);
    var40 = __builtin_IMCE_GET_CREG((short)1);
    var41 = __builtin_IMCE_GET_CREG((short)2);
    var42 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var43 = __builtin_IMCE_RECV(1);
    var44 = __builtin_IMCE_DIV(var39, var43, 15);
    var45 = __builtin_IMCE_RECV(1);
    var46 = __builtin_IMCE_DIV(var40, var45, 15);
    var47 = __builtin_IMCE_RECV(1);
    var48 = __builtin_IMCE_DIV(var41, var47, 15);
    var49 = __builtin_IMCE_RECV(1);
    var50 = __builtin_IMCE_DIV(var42, var49, 15);
    // endgenerate: div


    // generate: add
    var51 = __builtin_IMCE_RECV(2);
    var19 = __builtin_IMCE_ADD(var51, var44, 15);
    var52 = __builtin_IMCE_RECV(2);
    var21 = __builtin_IMCE_ADD(var52, var46, 15);
    var53 = __builtin_IMCE_RECV(2);
    var23 = __builtin_IMCE_ADD(var53, var48, 15);
    var54 = __builtin_IMCE_RECV(2);
    var25 = __builtin_IMCE_ADD(var54, var50, 15);
    // endgenerate: add


    __builtin_IMCE_SEND(0, var19, 3, 0);
    __builtin_IMCE_SEND(0, var21, 3, 0);
    __builtin_IMCE_SEND(0, var23, 3, 0);
    __builtin_IMCE_SEND(0, var25, 3, 0);


  } // endgenerate: inner_loop
  // endgenerate: conv exec
}
else if (hid == 1 && wid == 1) {
  // generate: weight write
  for (int i1 = 0; i1 < 8192; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(1);
  } // endgenerate: load_block
  // endgenerate: weight write
  // generate: conv exec
  for (int i1 = 0; i1 < 18; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
  } // endgenerate: load_block
  __builtin_IMCE_STEP();

  var55 = __builtin_IMCE_GET_CREG((short)0);
  var56 = __builtin_IMCE_GET_CREG((short)1);
  var57 = __builtin_IMCE_GET_CREG((short)2);
  var58 = __builtin_IMCE_GET_CREG((short)3);


  // generate: div
  var59 = __builtin_IMCE_RECV(1);
  var60 = __builtin_IMCE_DIV(var55, var59, 15);
  var61 = __builtin_IMCE_RECV(1);
  var62 = __builtin_IMCE_DIV(var56, var61, 15);
  var63 = __builtin_IMCE_RECV(1);
  var64 = __builtin_IMCE_DIV(var57, var63, 15);
  var65 = __builtin_IMCE_RECV(1);
  var66 = __builtin_IMCE_DIV(var58, var65, 15);
  // endgenerate: div


  // generate: add
  var67 = __builtin_IMCE_RECV(2);
  var51 = __builtin_IMCE_ADD(var67, var60, 15);
  var68 = __builtin_IMCE_RECV(2);
  var52 = __builtin_IMCE_ADD(var68, var62, 15);
  var69 = __builtin_IMCE_RECV(2);
  var53 = __builtin_IMCE_ADD(var69, var64, 15);
  var70 = __builtin_IMCE_RECV(2);
  var54 = __builtin_IMCE_ADD(var70, var66, 15);
  // endgenerate: add


  __builtin_IMCE_SEND(6, var51, 2, 0);
  __builtin_IMCE_SEND(6, var52, 2, 0);
  __builtin_IMCE_SEND(6, var53, 2, 0);
  __builtin_IMCE_SEND(6, var54, 2, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: inner_loop
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_STEP();

    var55 = __builtin_IMCE_GET_CREG((short)0);
    var56 = __builtin_IMCE_GET_CREG((short)1);
    var57 = __builtin_IMCE_GET_CREG((short)2);
    var58 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var59 = __builtin_IMCE_RECV(1);
    var60 = __builtin_IMCE_DIV(var55, var59, 15);
    var61 = __builtin_IMCE_RECV(1);
    var62 = __builtin_IMCE_DIV(var56, var61, 15);
    var63 = __builtin_IMCE_RECV(1);
    var64 = __builtin_IMCE_DIV(var57, var63, 15);
    var65 = __builtin_IMCE_RECV(1);
    var66 = __builtin_IMCE_DIV(var58, var65, 15);
    // endgenerate: div


    // generate: add
    var67 = __builtin_IMCE_RECV(2);
    var51 = __builtin_IMCE_ADD(var67, var60, 15);
    var68 = __builtin_IMCE_RECV(2);
    var52 = __builtin_IMCE_ADD(var68, var62, 15);
    var69 = __builtin_IMCE_RECV(2);
    var53 = __builtin_IMCE_ADD(var69, var64, 15);
    var70 = __builtin_IMCE_RECV(2);
    var54 = __builtin_IMCE_ADD(var70, var66, 15);
    // endgenerate: add


    __builtin_IMCE_SEND(6, var51, 2, 0);
    __builtin_IMCE_SEND(6, var52, 2, 0);
    __builtin_IMCE_SEND(6, var53, 2, 0);
    __builtin_IMCE_SEND(6, var54, 2, 0);


  } // endgenerate: inner_loop

  __builtin_IMCE_STEP();

  var55 = __builtin_IMCE_GET_CREG((short)0);
  var56 = __builtin_IMCE_GET_CREG((short)1);
  var57 = __builtin_IMCE_GET_CREG((short)2);
  var58 = __builtin_IMCE_GET_CREG((short)3);


  // generate: div
  var59 = __builtin_IMCE_RECV(1);
  var60 = __builtin_IMCE_DIV(var55, var59, 15);
  var61 = __builtin_IMCE_RECV(1);
  var62 = __builtin_IMCE_DIV(var56, var61, 15);
  var63 = __builtin_IMCE_RECV(1);
  var64 = __builtin_IMCE_DIV(var57, var63, 15);
  var65 = __builtin_IMCE_RECV(1);
  var66 = __builtin_IMCE_DIV(var58, var65, 15);
  // endgenerate: div


  // generate: add
  var67 = __builtin_IMCE_RECV(2);
  var51 = __builtin_IMCE_ADD(var67, var60, 15);
  var68 = __builtin_IMCE_RECV(2);
  var52 = __builtin_IMCE_ADD(var68, var62, 15);
  var69 = __builtin_IMCE_RECV(2);
  var53 = __builtin_IMCE_ADD(var69, var64, 15);
  var70 = __builtin_IMCE_RECV(2);
  var54 = __builtin_IMCE_ADD(var70, var66, 15);
  // endgenerate: add


  __builtin_IMCE_SEND(6, var51, 2, 0);
  __builtin_IMCE_SEND(6, var52, 2, 0);
  __builtin_IMCE_SEND(6, var53, 2, 0);
  __builtin_IMCE_SEND(6, var54, 2, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: outer_loop
    for (int i2 = 0; i2 < 2; i2++) { // generate: load_block
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
    } // endgenerate: load_block
    __builtin_IMCE_STEP();

    var55 = __builtin_IMCE_GET_CREG((short)0);
    var56 = __builtin_IMCE_GET_CREG((short)1);
    var57 = __builtin_IMCE_GET_CREG((short)2);
    var58 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var59 = __builtin_IMCE_RECV(1);
    var60 = __builtin_IMCE_DIV(var55, var59, 15);
    var61 = __builtin_IMCE_RECV(1);
    var62 = __builtin_IMCE_DIV(var56, var61, 15);
    var63 = __builtin_IMCE_RECV(1);
    var64 = __builtin_IMCE_DIV(var57, var63, 15);
    var65 = __builtin_IMCE_RECV(1);
    var66 = __builtin_IMCE_DIV(var58, var65, 15);
    // endgenerate: div


    // generate: add
    var67 = __builtin_IMCE_RECV(2);
    var51 = __builtin_IMCE_ADD(var67, var60, 15);
    var68 = __builtin_IMCE_RECV(2);
    var52 = __builtin_IMCE_ADD(var68, var62, 15);
    var69 = __builtin_IMCE_RECV(2);
    var53 = __builtin_IMCE_ADD(var69, var64, 15);
    var70 = __builtin_IMCE_RECV(2);
    var54 = __builtin_IMCE_ADD(var70, var66, 15);
    // endgenerate: add


    __builtin_IMCE_SEND(6, var51, 2, 0);
    __builtin_IMCE_SEND(6, var52, 2, 0);
    __builtin_IMCE_SEND(6, var53, 2, 0);
    __builtin_IMCE_SEND(6, var54, 2, 0);


    for (int i2 = 0; i2 < 14; i2++) { // generate: inner_loop
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_STEP();

      var55 = __builtin_IMCE_GET_CREG((short)0);
      var56 = __builtin_IMCE_GET_CREG((short)1);
      var57 = __builtin_IMCE_GET_CREG((short)2);
      var58 = __builtin_IMCE_GET_CREG((short)3);


      // generate: div
      var59 = __builtin_IMCE_RECV(1);
      var60 = __builtin_IMCE_DIV(var55, var59, 15);
      var61 = __builtin_IMCE_RECV(1);
      var62 = __builtin_IMCE_DIV(var56, var61, 15);
      var63 = __builtin_IMCE_RECV(1);
      var64 = __builtin_IMCE_DIV(var57, var63, 15);
      var65 = __builtin_IMCE_RECV(1);
      var66 = __builtin_IMCE_DIV(var58, var65, 15);
      // endgenerate: div


      // generate: add
      var67 = __builtin_IMCE_RECV(2);
      var51 = __builtin_IMCE_ADD(var67, var60, 15);
      var68 = __builtin_IMCE_RECV(2);
      var52 = __builtin_IMCE_ADD(var68, var62, 15);
      var69 = __builtin_IMCE_RECV(2);
      var53 = __builtin_IMCE_ADD(var69, var64, 15);
      var70 = __builtin_IMCE_RECV(2);
      var54 = __builtin_IMCE_ADD(var70, var66, 15);
      // endgenerate: add


      __builtin_IMCE_SEND(6, var51, 2, 0);
      __builtin_IMCE_SEND(6, var52, 2, 0);
      __builtin_IMCE_SEND(6, var53, 2, 0);
      __builtin_IMCE_SEND(6, var54, 2, 0);


    } // endgenerate: inner_loop

    __builtin_IMCE_STEP();

    var55 = __builtin_IMCE_GET_CREG((short)0);
    var56 = __builtin_IMCE_GET_CREG((short)1);
    var57 = __builtin_IMCE_GET_CREG((short)2);
    var58 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var59 = __builtin_IMCE_RECV(1);
    var60 = __builtin_IMCE_DIV(var55, var59, 15);
    var61 = __builtin_IMCE_RECV(1);
    var62 = __builtin_IMCE_DIV(var56, var61, 15);
    var63 = __builtin_IMCE_RECV(1);
    var64 = __builtin_IMCE_DIV(var57, var63, 15);
    var65 = __builtin_IMCE_RECV(1);
    var66 = __builtin_IMCE_DIV(var58, var65, 15);
    // endgenerate: div


    // generate: add
    var67 = __builtin_IMCE_RECV(2);
    var51 = __builtin_IMCE_ADD(var67, var60, 15);
    var68 = __builtin_IMCE_RECV(2);
    var52 = __builtin_IMCE_ADD(var68, var62, 15);
    var69 = __builtin_IMCE_RECV(2);
    var53 = __builtin_IMCE_ADD(var69, var64, 15);
    var70 = __builtin_IMCE_RECV(2);
    var54 = __builtin_IMCE_ADD(var70, var66, 15);
    // endgenerate: add


    __builtin_IMCE_SEND(6, var51, 2, 0);
    __builtin_IMCE_SEND(6, var52, 2, 0);
    __builtin_IMCE_SEND(6, var53, 2, 0);
    __builtin_IMCE_SEND(6, var54, 2, 0);


  } // endgenerate: outer_loop
  for (int i1 = 0; i1 < 16; i1++) { // generate: inner_loop

    __builtin_IMCE_STEP();

    var55 = __builtin_IMCE_GET_CREG((short)0);
    var56 = __builtin_IMCE_GET_CREG((short)1);
    var57 = __builtin_IMCE_GET_CREG((short)2);
    var58 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var59 = __builtin_IMCE_RECV(1);
    var60 = __builtin_IMCE_DIV(var55, var59, 15);
    var61 = __builtin_IMCE_RECV(1);
    var62 = __builtin_IMCE_DIV(var56, var61, 15);
    var63 = __builtin_IMCE_RECV(1);
    var64 = __builtin_IMCE_DIV(var57, var63, 15);
    var65 = __builtin_IMCE_RECV(1);
    var66 = __builtin_IMCE_DIV(var58, var65, 15);
    // endgenerate: div


    // generate: add
    var67 = __builtin_IMCE_RECV(2);
    var51 = __builtin_IMCE_ADD(var67, var60, 15);
    var68 = __builtin_IMCE_RECV(2);
    var52 = __builtin_IMCE_ADD(var68, var62, 15);
    var69 = __builtin_IMCE_RECV(2);
    var53 = __builtin_IMCE_ADD(var69, var64, 15);
    var70 = __builtin_IMCE_RECV(2);
    var54 = __builtin_IMCE_ADD(var70, var66, 15);
    // endgenerate: add


    __builtin_IMCE_SEND(6, var51, 2, 0);
    __builtin_IMCE_SEND(6, var52, 2, 0);
    __builtin_IMCE_SEND(6, var53, 2, 0);
    __builtin_IMCE_SEND(6, var54, 2, 0);


  } // endgenerate: inner_loop
  // endgenerate: conv exec
}
else if (hid == 1 && wid == 2) {
  // generate: weight write
  for (int i1 = 0; i1 < 8192; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(1);
  } // endgenerate: load_block
  // endgenerate: weight write
  // generate: conv exec
  for (int i1 = 0; i1 < 18; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
  } // endgenerate: load_block
  __builtin_IMCE_STEP();

  var71 = __builtin_IMCE_GET_CREG((short)0);
  var72 = __builtin_IMCE_GET_CREG((short)1);
  var73 = __builtin_IMCE_GET_CREG((short)2);
  var74 = __builtin_IMCE_GET_CREG((short)3);


  // generate: div
  var75 = __builtin_IMCE_RECV(1);
  var76 = __builtin_IMCE_DIV(var71, var75, 15);
  var77 = __builtin_IMCE_RECV(1);
  var78 = __builtin_IMCE_DIV(var72, var77, 15);
  var79 = __builtin_IMCE_RECV(1);
  var80 = __builtin_IMCE_DIV(var73, var79, 15);
  var81 = __builtin_IMCE_RECV(1);
  var82 = __builtin_IMCE_DIV(var74, var81, 15);
  // endgenerate: div


  // generate: add
  var83 = __builtin_IMCE_RECV(2);
  var67 = __builtin_IMCE_ADD(var76, var83, 15);
  var84 = __builtin_IMCE_RECV(2);
  var68 = __builtin_IMCE_ADD(var78, var84, 15);
  var85 = __builtin_IMCE_RECV(2);
  var69 = __builtin_IMCE_ADD(var80, var85, 15);
  var86 = __builtin_IMCE_RECV(2);
  var70 = __builtin_IMCE_ADD(var82, var86, 15);
  // endgenerate: add


  __builtin_IMCE_SEND(7, var67, 2, 0);
  __builtin_IMCE_SEND(7, var68, 2, 0);
  __builtin_IMCE_SEND(7, var69, 2, 0);
  __builtin_IMCE_SEND(7, var70, 2, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: inner_loop
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_STEP();

    var71 = __builtin_IMCE_GET_CREG((short)0);
    var72 = __builtin_IMCE_GET_CREG((short)1);
    var73 = __builtin_IMCE_GET_CREG((short)2);
    var74 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var75 = __builtin_IMCE_RECV(1);
    var76 = __builtin_IMCE_DIV(var71, var75, 15);
    var77 = __builtin_IMCE_RECV(1);
    var78 = __builtin_IMCE_DIV(var72, var77, 15);
    var79 = __builtin_IMCE_RECV(1);
    var80 = __builtin_IMCE_DIV(var73, var79, 15);
    var81 = __builtin_IMCE_RECV(1);
    var82 = __builtin_IMCE_DIV(var74, var81, 15);
    // endgenerate: div


    // generate: add
    var83 = __builtin_IMCE_RECV(2);
    var67 = __builtin_IMCE_ADD(var76, var83, 15);
    var84 = __builtin_IMCE_RECV(2);
    var68 = __builtin_IMCE_ADD(var78, var84, 15);
    var85 = __builtin_IMCE_RECV(2);
    var69 = __builtin_IMCE_ADD(var80, var85, 15);
    var86 = __builtin_IMCE_RECV(2);
    var70 = __builtin_IMCE_ADD(var82, var86, 15);
    // endgenerate: add


    __builtin_IMCE_SEND(7, var67, 2, 0);
    __builtin_IMCE_SEND(7, var68, 2, 0);
    __builtin_IMCE_SEND(7, var69, 2, 0);
    __builtin_IMCE_SEND(7, var70, 2, 0);


  } // endgenerate: inner_loop

  __builtin_IMCE_STEP();

  var71 = __builtin_IMCE_GET_CREG((short)0);
  var72 = __builtin_IMCE_GET_CREG((short)1);
  var73 = __builtin_IMCE_GET_CREG((short)2);
  var74 = __builtin_IMCE_GET_CREG((short)3);


  // generate: div
  var75 = __builtin_IMCE_RECV(1);
  var76 = __builtin_IMCE_DIV(var71, var75, 15);
  var77 = __builtin_IMCE_RECV(1);
  var78 = __builtin_IMCE_DIV(var72, var77, 15);
  var79 = __builtin_IMCE_RECV(1);
  var80 = __builtin_IMCE_DIV(var73, var79, 15);
  var81 = __builtin_IMCE_RECV(1);
  var82 = __builtin_IMCE_DIV(var74, var81, 15);
  // endgenerate: div


  // generate: add
  var83 = __builtin_IMCE_RECV(2);
  var67 = __builtin_IMCE_ADD(var76, var83, 15);
  var84 = __builtin_IMCE_RECV(2);
  var68 = __builtin_IMCE_ADD(var78, var84, 15);
  var85 = __builtin_IMCE_RECV(2);
  var69 = __builtin_IMCE_ADD(var80, var85, 15);
  var86 = __builtin_IMCE_RECV(2);
  var70 = __builtin_IMCE_ADD(var82, var86, 15);
  // endgenerate: add


  __builtin_IMCE_SEND(7, var67, 2, 0);
  __builtin_IMCE_SEND(7, var68, 2, 0);
  __builtin_IMCE_SEND(7, var69, 2, 0);
  __builtin_IMCE_SEND(7, var70, 2, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: outer_loop
    for (int i2 = 0; i2 < 2; i2++) { // generate: load_block
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
    } // endgenerate: load_block
    __builtin_IMCE_STEP();

    var71 = __builtin_IMCE_GET_CREG((short)0);
    var72 = __builtin_IMCE_GET_CREG((short)1);
    var73 = __builtin_IMCE_GET_CREG((short)2);
    var74 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var75 = __builtin_IMCE_RECV(1);
    var76 = __builtin_IMCE_DIV(var71, var75, 15);
    var77 = __builtin_IMCE_RECV(1);
    var78 = __builtin_IMCE_DIV(var72, var77, 15);
    var79 = __builtin_IMCE_RECV(1);
    var80 = __builtin_IMCE_DIV(var73, var79, 15);
    var81 = __builtin_IMCE_RECV(1);
    var82 = __builtin_IMCE_DIV(var74, var81, 15);
    // endgenerate: div


    // generate: add
    var83 = __builtin_IMCE_RECV(2);
    var67 = __builtin_IMCE_ADD(var76, var83, 15);
    var84 = __builtin_IMCE_RECV(2);
    var68 = __builtin_IMCE_ADD(var78, var84, 15);
    var85 = __builtin_IMCE_RECV(2);
    var69 = __builtin_IMCE_ADD(var80, var85, 15);
    var86 = __builtin_IMCE_RECV(2);
    var70 = __builtin_IMCE_ADD(var82, var86, 15);
    // endgenerate: add


    __builtin_IMCE_SEND(7, var67, 2, 0);
    __builtin_IMCE_SEND(7, var68, 2, 0);
    __builtin_IMCE_SEND(7, var69, 2, 0);
    __builtin_IMCE_SEND(7, var70, 2, 0);


    for (int i2 = 0; i2 < 14; i2++) { // generate: inner_loop
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_STEP();

      var71 = __builtin_IMCE_GET_CREG((short)0);
      var72 = __builtin_IMCE_GET_CREG((short)1);
      var73 = __builtin_IMCE_GET_CREG((short)2);
      var74 = __builtin_IMCE_GET_CREG((short)3);


      // generate: div
      var75 = __builtin_IMCE_RECV(1);
      var76 = __builtin_IMCE_DIV(var71, var75, 15);
      var77 = __builtin_IMCE_RECV(1);
      var78 = __builtin_IMCE_DIV(var72, var77, 15);
      var79 = __builtin_IMCE_RECV(1);
      var80 = __builtin_IMCE_DIV(var73, var79, 15);
      var81 = __builtin_IMCE_RECV(1);
      var82 = __builtin_IMCE_DIV(var74, var81, 15);
      // endgenerate: div


      // generate: add
      var83 = __builtin_IMCE_RECV(2);
      var67 = __builtin_IMCE_ADD(var76, var83, 15);
      var84 = __builtin_IMCE_RECV(2);
      var68 = __builtin_IMCE_ADD(var78, var84, 15);
      var85 = __builtin_IMCE_RECV(2);
      var69 = __builtin_IMCE_ADD(var80, var85, 15);
      var86 = __builtin_IMCE_RECV(2);
      var70 = __builtin_IMCE_ADD(var82, var86, 15);
      // endgenerate: add


      __builtin_IMCE_SEND(7, var67, 2, 0);
      __builtin_IMCE_SEND(7, var68, 2, 0);
      __builtin_IMCE_SEND(7, var69, 2, 0);
      __builtin_IMCE_SEND(7, var70, 2, 0);


    } // endgenerate: inner_loop

    __builtin_IMCE_STEP();

    var71 = __builtin_IMCE_GET_CREG((short)0);
    var72 = __builtin_IMCE_GET_CREG((short)1);
    var73 = __builtin_IMCE_GET_CREG((short)2);
    var74 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var75 = __builtin_IMCE_RECV(1);
    var76 = __builtin_IMCE_DIV(var71, var75, 15);
    var77 = __builtin_IMCE_RECV(1);
    var78 = __builtin_IMCE_DIV(var72, var77, 15);
    var79 = __builtin_IMCE_RECV(1);
    var80 = __builtin_IMCE_DIV(var73, var79, 15);
    var81 = __builtin_IMCE_RECV(1);
    var82 = __builtin_IMCE_DIV(var74, var81, 15);
    // endgenerate: div


    // generate: add
    var83 = __builtin_IMCE_RECV(2);
    var67 = __builtin_IMCE_ADD(var76, var83, 15);
    var84 = __builtin_IMCE_RECV(2);
    var68 = __builtin_IMCE_ADD(var78, var84, 15);
    var85 = __builtin_IMCE_RECV(2);
    var69 = __builtin_IMCE_ADD(var80, var85, 15);
    var86 = __builtin_IMCE_RECV(2);
    var70 = __builtin_IMCE_ADD(var82, var86, 15);
    // endgenerate: add


    __builtin_IMCE_SEND(7, var67, 2, 0);
    __builtin_IMCE_SEND(7, var68, 2, 0);
    __builtin_IMCE_SEND(7, var69, 2, 0);
    __builtin_IMCE_SEND(7, var70, 2, 0);


  } // endgenerate: outer_loop
  for (int i1 = 0; i1 < 16; i1++) { // generate: inner_loop

    __builtin_IMCE_STEP();

    var71 = __builtin_IMCE_GET_CREG((short)0);
    var72 = __builtin_IMCE_GET_CREG((short)1);
    var73 = __builtin_IMCE_GET_CREG((short)2);
    var74 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var75 = __builtin_IMCE_RECV(1);
    var76 = __builtin_IMCE_DIV(var71, var75, 15);
    var77 = __builtin_IMCE_RECV(1);
    var78 = __builtin_IMCE_DIV(var72, var77, 15);
    var79 = __builtin_IMCE_RECV(1);
    var80 = __builtin_IMCE_DIV(var73, var79, 15);
    var81 = __builtin_IMCE_RECV(1);
    var82 = __builtin_IMCE_DIV(var74, var81, 15);
    // endgenerate: div


    // generate: add
    var83 = __builtin_IMCE_RECV(2);
    var67 = __builtin_IMCE_ADD(var76, var83, 15);
    var84 = __builtin_IMCE_RECV(2);
    var68 = __builtin_IMCE_ADD(var78, var84, 15);
    var85 = __builtin_IMCE_RECV(2);
    var69 = __builtin_IMCE_ADD(var80, var85, 15);
    var86 = __builtin_IMCE_RECV(2);
    var70 = __builtin_IMCE_ADD(var82, var86, 15);
    // endgenerate: add


    __builtin_IMCE_SEND(7, var67, 2, 0);
    __builtin_IMCE_SEND(7, var68, 2, 0);
    __builtin_IMCE_SEND(7, var69, 2, 0);
    __builtin_IMCE_SEND(7, var70, 2, 0);


  } // endgenerate: inner_loop
  // endgenerate: conv exec
}
else if (hid == 1 && wid == 3) {
  // generate: weight write
  for (int i1 = 0; i1 < 8192; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(1);
  } // endgenerate: load_block
  // endgenerate: weight write
  // generate: conv exec
  for (int i1 = 0; i1 < 18; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
  } // endgenerate: load_block
  __builtin_IMCE_STEP();

  var87 = __builtin_IMCE_GET_CREG((short)0);
  var88 = __builtin_IMCE_GET_CREG((short)1);
  var89 = __builtin_IMCE_GET_CREG((short)2);
  var90 = __builtin_IMCE_GET_CREG((short)3);


  // generate: div
  var91 = __builtin_IMCE_RECV(1);
  var83 = __builtin_IMCE_DIV(var87, var91, 15);
  var92 = __builtin_IMCE_RECV(1);
  var84 = __builtin_IMCE_DIV(var88, var92, 15);
  var93 = __builtin_IMCE_RECV(1);
  var85 = __builtin_IMCE_DIV(var89, var93, 15);
  var94 = __builtin_IMCE_RECV(1);
  var86 = __builtin_IMCE_DIV(var90, var94, 15);
  // endgenerate: div


  __builtin_IMCE_SEND(9, var83, 2, 0);
  __builtin_IMCE_SEND(9, var84, 2, 0);
  __builtin_IMCE_SEND(9, var85, 2, 0);
  __builtin_IMCE_SEND(9, var86, 2, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: inner_loop
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_STEP();

    var87 = __builtin_IMCE_GET_CREG((short)0);
    var88 = __builtin_IMCE_GET_CREG((short)1);
    var89 = __builtin_IMCE_GET_CREG((short)2);
    var90 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var91 = __builtin_IMCE_RECV(1);
    var83 = __builtin_IMCE_DIV(var87, var91, 15);
    var92 = __builtin_IMCE_RECV(1);
    var84 = __builtin_IMCE_DIV(var88, var92, 15);
    var93 = __builtin_IMCE_RECV(1);
    var85 = __builtin_IMCE_DIV(var89, var93, 15);
    var94 = __builtin_IMCE_RECV(1);
    var86 = __builtin_IMCE_DIV(var90, var94, 15);
    // endgenerate: div


    __builtin_IMCE_SEND(9, var83, 2, 0);
    __builtin_IMCE_SEND(9, var84, 2, 0);
    __builtin_IMCE_SEND(9, var85, 2, 0);
    __builtin_IMCE_SEND(9, var86, 2, 0);


  } // endgenerate: inner_loop

  __builtin_IMCE_STEP();

  var87 = __builtin_IMCE_GET_CREG((short)0);
  var88 = __builtin_IMCE_GET_CREG((short)1);
  var89 = __builtin_IMCE_GET_CREG((short)2);
  var90 = __builtin_IMCE_GET_CREG((short)3);


  // generate: div
  var91 = __builtin_IMCE_RECV(1);
  var83 = __builtin_IMCE_DIV(var87, var91, 15);
  var92 = __builtin_IMCE_RECV(1);
  var84 = __builtin_IMCE_DIV(var88, var92, 15);
  var93 = __builtin_IMCE_RECV(1);
  var85 = __builtin_IMCE_DIV(var89, var93, 15);
  var94 = __builtin_IMCE_RECV(1);
  var86 = __builtin_IMCE_DIV(var90, var94, 15);
  // endgenerate: div


  __builtin_IMCE_SEND(9, var83, 2, 0);
  __builtin_IMCE_SEND(9, var84, 2, 0);
  __builtin_IMCE_SEND(9, var85, 2, 0);
  __builtin_IMCE_SEND(9, var86, 2, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: outer_loop
    for (int i2 = 0; i2 < 2; i2++) { // generate: load_block
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
    } // endgenerate: load_block
    __builtin_IMCE_STEP();

    var87 = __builtin_IMCE_GET_CREG((short)0);
    var88 = __builtin_IMCE_GET_CREG((short)1);
    var89 = __builtin_IMCE_GET_CREG((short)2);
    var90 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var91 = __builtin_IMCE_RECV(1);
    var83 = __builtin_IMCE_DIV(var87, var91, 15);
    var92 = __builtin_IMCE_RECV(1);
    var84 = __builtin_IMCE_DIV(var88, var92, 15);
    var93 = __builtin_IMCE_RECV(1);
    var85 = __builtin_IMCE_DIV(var89, var93, 15);
    var94 = __builtin_IMCE_RECV(1);
    var86 = __builtin_IMCE_DIV(var90, var94, 15);
    // endgenerate: div


    __builtin_IMCE_SEND(9, var83, 2, 0);
    __builtin_IMCE_SEND(9, var84, 2, 0);
    __builtin_IMCE_SEND(9, var85, 2, 0);
    __builtin_IMCE_SEND(9, var86, 2, 0);


    for (int i2 = 0; i2 < 14; i2++) { // generate: inner_loop
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_STEP();

      var87 = __builtin_IMCE_GET_CREG((short)0);
      var88 = __builtin_IMCE_GET_CREG((short)1);
      var89 = __builtin_IMCE_GET_CREG((short)2);
      var90 = __builtin_IMCE_GET_CREG((short)3);


      // generate: div
      var91 = __builtin_IMCE_RECV(1);
      var83 = __builtin_IMCE_DIV(var87, var91, 15);
      var92 = __builtin_IMCE_RECV(1);
      var84 = __builtin_IMCE_DIV(var88, var92, 15);
      var93 = __builtin_IMCE_RECV(1);
      var85 = __builtin_IMCE_DIV(var89, var93, 15);
      var94 = __builtin_IMCE_RECV(1);
      var86 = __builtin_IMCE_DIV(var90, var94, 15);
      // endgenerate: div


      __builtin_IMCE_SEND(9, var83, 2, 0);
      __builtin_IMCE_SEND(9, var84, 2, 0);
      __builtin_IMCE_SEND(9, var85, 2, 0);
      __builtin_IMCE_SEND(9, var86, 2, 0);


    } // endgenerate: inner_loop

    __builtin_IMCE_STEP();

    var87 = __builtin_IMCE_GET_CREG((short)0);
    var88 = __builtin_IMCE_GET_CREG((short)1);
    var89 = __builtin_IMCE_GET_CREG((short)2);
    var90 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var91 = __builtin_IMCE_RECV(1);
    var83 = __builtin_IMCE_DIV(var87, var91, 15);
    var92 = __builtin_IMCE_RECV(1);
    var84 = __builtin_IMCE_DIV(var88, var92, 15);
    var93 = __builtin_IMCE_RECV(1);
    var85 = __builtin_IMCE_DIV(var89, var93, 15);
    var94 = __builtin_IMCE_RECV(1);
    var86 = __builtin_IMCE_DIV(var90, var94, 15);
    // endgenerate: div


    __builtin_IMCE_SEND(9, var83, 2, 0);
    __builtin_IMCE_SEND(9, var84, 2, 0);
    __builtin_IMCE_SEND(9, var85, 2, 0);
    __builtin_IMCE_SEND(9, var86, 2, 0);


  } // endgenerate: outer_loop
  for (int i1 = 0; i1 < 16; i1++) { // generate: inner_loop

    __builtin_IMCE_STEP();

    var87 = __builtin_IMCE_GET_CREG((short)0);
    var88 = __builtin_IMCE_GET_CREG((short)1);
    var89 = __builtin_IMCE_GET_CREG((short)2);
    var90 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var91 = __builtin_IMCE_RECV(1);
    var83 = __builtin_IMCE_DIV(var87, var91, 15);
    var92 = __builtin_IMCE_RECV(1);
    var84 = __builtin_IMCE_DIV(var88, var92, 15);
    var93 = __builtin_IMCE_RECV(1);
    var85 = __builtin_IMCE_DIV(var89, var93, 15);
    var94 = __builtin_IMCE_RECV(1);
    var86 = __builtin_IMCE_DIV(var90, var94, 15);
    // endgenerate: div


    __builtin_IMCE_SEND(9, var83, 2, 0);
    __builtin_IMCE_SEND(9, var84, 2, 0);
    __builtin_IMCE_SEND(9, var85, 2, 0);
    __builtin_IMCE_SEND(9, var86, 2, 0);


  } // endgenerate: inner_loop
  // endgenerate: conv exec
}
else if (hid == 1 && wid == 4) {
  // generate: weight write
  for (int i1 = 0; i1 < 8192; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(1);
  } // endgenerate: load_block
  // endgenerate: weight write
  // generate: min write
  var95 = __builtin_IMCE_RECV(1);
  // endgenerate: min write
  // generate: max write
  var96 = __builtin_IMCE_RECV(1);
  // endgenerate: max write
  // generate: conv exec
  for (int i1 = 0; i1 < 18; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
  } // endgenerate: load_block
  __builtin_IMCE_STEP();

  var97 = __builtin_IMCE_GET_CREG((short)0);
  var98 = __builtin_IMCE_GET_CREG((short)1);
  var99 = __builtin_IMCE_GET_CREG((short)2);
  var100 = __builtin_IMCE_GET_CREG((short)3);


  // generate: div
  var101 = __builtin_IMCE_RECV(1);
  var102 = __builtin_IMCE_DIV(var97, var101, 15);
  var103 = __builtin_IMCE_RECV(1);
  var104 = __builtin_IMCE_DIV(var98, var103, 15);
  var105 = __builtin_IMCE_RECV(1);
  var106 = __builtin_IMCE_DIV(var99, var105, 15);
  var107 = __builtin_IMCE_RECV(1);
  var108 = __builtin_IMCE_DIV(var100, var107, 15);
  // endgenerate: div


  // generate: add
  var109 = __builtin_IMCE_RECV(2);
  var110 = __builtin_IMCE_ADD(var109, var102, 15);
  var111 = __builtin_IMCE_RECV(2);
  var112 = __builtin_IMCE_ADD(var111, var104, 15);
  var113 = __builtin_IMCE_RECV(2);
  var114 = __builtin_IMCE_ADD(var113, var106, 15);
  var115 = __builtin_IMCE_RECV(2);
  var116 = __builtin_IMCE_ADD(var115, var108, 15);
  // endgenerate: add


  // generate: min_max_quantize
  __builtin_IMCE_MM_QUANT(var117, 0, 15, 4);
  var35 = __builtin_IMCE_GET_QREG(0);
  __builtin_IMCE_MM_QUANT(var118, 0, 15, 5);
  var36 = __builtin_IMCE_GET_QREG(1);
  __builtin_IMCE_MM_QUANT(var119, 0, 15, 6);
  var37 = __builtin_IMCE_GET_QREG(2);
  __builtin_IMCE_MM_QUANT(var120, 0, 15, 7);
  var38 = __builtin_IMCE_GET_QREG(3);
  // endgenerate: min_max_quantize


  __builtin_IMCE_SEND(0, var35, 2, 0);
  __builtin_IMCE_SEND(0, var36, 2, 0);
  __builtin_IMCE_SEND(0, var37, 2, 0);
  __builtin_IMCE_SEND(0, var38, 2, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: inner_loop
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_STEP();

    var97 = __builtin_IMCE_GET_CREG((short)0);
    var98 = __builtin_IMCE_GET_CREG((short)1);
    var99 = __builtin_IMCE_GET_CREG((short)2);
    var100 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var101 = __builtin_IMCE_RECV(1);
    var102 = __builtin_IMCE_DIV(var97, var101, 15);
    var103 = __builtin_IMCE_RECV(1);
    var104 = __builtin_IMCE_DIV(var98, var103, 15);
    var105 = __builtin_IMCE_RECV(1);
    var106 = __builtin_IMCE_DIV(var99, var105, 15);
    var107 = __builtin_IMCE_RECV(1);
    var108 = __builtin_IMCE_DIV(var100, var107, 15);
    // endgenerate: div


    // generate: add
    var109 = __builtin_IMCE_RECV(2);
    var110 = __builtin_IMCE_ADD(var109, var102, 15);
    var111 = __builtin_IMCE_RECV(2);
    var112 = __builtin_IMCE_ADD(var111, var104, 15);
    var113 = __builtin_IMCE_RECV(2);
    var114 = __builtin_IMCE_ADD(var113, var106, 15);
    var115 = __builtin_IMCE_RECV(2);
    var116 = __builtin_IMCE_ADD(var115, var108, 15);
    // endgenerate: add


    // generate: min_max_quantize
    __builtin_IMCE_MM_QUANT(var117, 0, 15, 4);
    var35 = __builtin_IMCE_GET_QREG(0);
    __builtin_IMCE_MM_QUANT(var118, 0, 15, 5);
    var36 = __builtin_IMCE_GET_QREG(1);
    __builtin_IMCE_MM_QUANT(var119, 0, 15, 6);
    var37 = __builtin_IMCE_GET_QREG(2);
    __builtin_IMCE_MM_QUANT(var120, 0, 15, 7);
    var38 = __builtin_IMCE_GET_QREG(3);
    // endgenerate: min_max_quantize


    __builtin_IMCE_SEND(0, var35, 2, 0);
    __builtin_IMCE_SEND(0, var36, 2, 0);
    __builtin_IMCE_SEND(0, var37, 2, 0);
    __builtin_IMCE_SEND(0, var38, 2, 0);


  } // endgenerate: inner_loop

  __builtin_IMCE_STEP();

  var97 = __builtin_IMCE_GET_CREG((short)0);
  var98 = __builtin_IMCE_GET_CREG((short)1);
  var99 = __builtin_IMCE_GET_CREG((short)2);
  var100 = __builtin_IMCE_GET_CREG((short)3);


  // generate: div
  var101 = __builtin_IMCE_RECV(1);
  var102 = __builtin_IMCE_DIV(var97, var101, 15);
  var103 = __builtin_IMCE_RECV(1);
  var104 = __builtin_IMCE_DIV(var98, var103, 15);
  var105 = __builtin_IMCE_RECV(1);
  var106 = __builtin_IMCE_DIV(var99, var105, 15);
  var107 = __builtin_IMCE_RECV(1);
  var108 = __builtin_IMCE_DIV(var100, var107, 15);
  // endgenerate: div


  // generate: add
  var109 = __builtin_IMCE_RECV(2);
  var110 = __builtin_IMCE_ADD(var109, var102, 15);
  var111 = __builtin_IMCE_RECV(2);
  var112 = __builtin_IMCE_ADD(var111, var104, 15);
  var113 = __builtin_IMCE_RECV(2);
  var114 = __builtin_IMCE_ADD(var113, var106, 15);
  var115 = __builtin_IMCE_RECV(2);
  var116 = __builtin_IMCE_ADD(var115, var108, 15);
  // endgenerate: add


  // generate: min_max_quantize
  __builtin_IMCE_MM_QUANT(var117, 0, 15, 4);
  var35 = __builtin_IMCE_GET_QREG(0);
  __builtin_IMCE_MM_QUANT(var118, 0, 15, 5);
  var36 = __builtin_IMCE_GET_QREG(1);
  __builtin_IMCE_MM_QUANT(var119, 0, 15, 6);
  var37 = __builtin_IMCE_GET_QREG(2);
  __builtin_IMCE_MM_QUANT(var120, 0, 15, 7);
  var38 = __builtin_IMCE_GET_QREG(3);
  // endgenerate: min_max_quantize


  __builtin_IMCE_SEND(0, var35, 2, 0);
  __builtin_IMCE_SEND(0, var36, 2, 0);
  __builtin_IMCE_SEND(0, var37, 2, 0);
  __builtin_IMCE_SEND(0, var38, 2, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: outer_loop
    for (int i2 = 0; i2 < 2; i2++) { // generate: load_block
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
    } // endgenerate: load_block
    __builtin_IMCE_STEP();

    var97 = __builtin_IMCE_GET_CREG((short)0);
    var98 = __builtin_IMCE_GET_CREG((short)1);
    var99 = __builtin_IMCE_GET_CREG((short)2);
    var100 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var101 = __builtin_IMCE_RECV(1);
    var102 = __builtin_IMCE_DIV(var97, var101, 15);
    var103 = __builtin_IMCE_RECV(1);
    var104 = __builtin_IMCE_DIV(var98, var103, 15);
    var105 = __builtin_IMCE_RECV(1);
    var106 = __builtin_IMCE_DIV(var99, var105, 15);
    var107 = __builtin_IMCE_RECV(1);
    var108 = __builtin_IMCE_DIV(var100, var107, 15);
    // endgenerate: div


    // generate: add
    var109 = __builtin_IMCE_RECV(2);
    var110 = __builtin_IMCE_ADD(var109, var102, 15);
    var111 = __builtin_IMCE_RECV(2);
    var112 = __builtin_IMCE_ADD(var111, var104, 15);
    var113 = __builtin_IMCE_RECV(2);
    var114 = __builtin_IMCE_ADD(var113, var106, 15);
    var115 = __builtin_IMCE_RECV(2);
    var116 = __builtin_IMCE_ADD(var115, var108, 15);
    // endgenerate: add


    // generate: min_max_quantize
    __builtin_IMCE_MM_QUANT(var117, 0, 15, 4);
    var35 = __builtin_IMCE_GET_QREG(0);
    __builtin_IMCE_MM_QUANT(var118, 0, 15, 5);
    var36 = __builtin_IMCE_GET_QREG(1);
    __builtin_IMCE_MM_QUANT(var119, 0, 15, 6);
    var37 = __builtin_IMCE_GET_QREG(2);
    __builtin_IMCE_MM_QUANT(var120, 0, 15, 7);
    var38 = __builtin_IMCE_GET_QREG(3);
    // endgenerate: min_max_quantize


    __builtin_IMCE_SEND(0, var35, 2, 0);
    __builtin_IMCE_SEND(0, var36, 2, 0);
    __builtin_IMCE_SEND(0, var37, 2, 0);
    __builtin_IMCE_SEND(0, var38, 2, 0);


    for (int i2 = 0; i2 < 14; i2++) { // generate: inner_loop
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_STEP();

      var97 = __builtin_IMCE_GET_CREG((short)0);
      var98 = __builtin_IMCE_GET_CREG((short)1);
      var99 = __builtin_IMCE_GET_CREG((short)2);
      var100 = __builtin_IMCE_GET_CREG((short)3);


      // generate: div
      var101 = __builtin_IMCE_RECV(1);
      var102 = __builtin_IMCE_DIV(var97, var101, 15);
      var103 = __builtin_IMCE_RECV(1);
      var104 = __builtin_IMCE_DIV(var98, var103, 15);
      var105 = __builtin_IMCE_RECV(1);
      var106 = __builtin_IMCE_DIV(var99, var105, 15);
      var107 = __builtin_IMCE_RECV(1);
      var108 = __builtin_IMCE_DIV(var100, var107, 15);
      // endgenerate: div


      // generate: add
      var109 = __builtin_IMCE_RECV(2);
      var110 = __builtin_IMCE_ADD(var109, var102, 15);
      var111 = __builtin_IMCE_RECV(2);
      var112 = __builtin_IMCE_ADD(var111, var104, 15);
      var113 = __builtin_IMCE_RECV(2);
      var114 = __builtin_IMCE_ADD(var113, var106, 15);
      var115 = __builtin_IMCE_RECV(2);
      var116 = __builtin_IMCE_ADD(var115, var108, 15);
      // endgenerate: add


      // generate: min_max_quantize
      __builtin_IMCE_MM_QUANT(var117, 0, 15, 4);
      var35 = __builtin_IMCE_GET_QREG(0);
      __builtin_IMCE_MM_QUANT(var118, 0, 15, 5);
      var36 = __builtin_IMCE_GET_QREG(1);
      __builtin_IMCE_MM_QUANT(var119, 0, 15, 6);
      var37 = __builtin_IMCE_GET_QREG(2);
      __builtin_IMCE_MM_QUANT(var120, 0, 15, 7);
      var38 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize


      __builtin_IMCE_SEND(0, var35, 2, 0);
      __builtin_IMCE_SEND(0, var36, 2, 0);
      __builtin_IMCE_SEND(0, var37, 2, 0);
      __builtin_IMCE_SEND(0, var38, 2, 0);


    } // endgenerate: inner_loop

    __builtin_IMCE_STEP();

    var97 = __builtin_IMCE_GET_CREG((short)0);
    var98 = __builtin_IMCE_GET_CREG((short)1);
    var99 = __builtin_IMCE_GET_CREG((short)2);
    var100 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var101 = __builtin_IMCE_RECV(1);
    var102 = __builtin_IMCE_DIV(var97, var101, 15);
    var103 = __builtin_IMCE_RECV(1);
    var104 = __builtin_IMCE_DIV(var98, var103, 15);
    var105 = __builtin_IMCE_RECV(1);
    var106 = __builtin_IMCE_DIV(var99, var105, 15);
    var107 = __builtin_IMCE_RECV(1);
    var108 = __builtin_IMCE_DIV(var100, var107, 15);
    // endgenerate: div


    // generate: add
    var109 = __builtin_IMCE_RECV(2);
    var110 = __builtin_IMCE_ADD(var109, var102, 15);
    var111 = __builtin_IMCE_RECV(2);
    var112 = __builtin_IMCE_ADD(var111, var104, 15);
    var113 = __builtin_IMCE_RECV(2);
    var114 = __builtin_IMCE_ADD(var113, var106, 15);
    var115 = __builtin_IMCE_RECV(2);
    var116 = __builtin_IMCE_ADD(var115, var108, 15);
    // endgenerate: add


    // generate: min_max_quantize
    __builtin_IMCE_MM_QUANT(var117, 0, 15, 4);
    var35 = __builtin_IMCE_GET_QREG(0);
    __builtin_IMCE_MM_QUANT(var118, 0, 15, 5);
    var36 = __builtin_IMCE_GET_QREG(1);
    __builtin_IMCE_MM_QUANT(var119, 0, 15, 6);
    var37 = __builtin_IMCE_GET_QREG(2);
    __builtin_IMCE_MM_QUANT(var120, 0, 15, 7);
    var38 = __builtin_IMCE_GET_QREG(3);
    // endgenerate: min_max_quantize


    __builtin_IMCE_SEND(0, var35, 2, 0);
    __builtin_IMCE_SEND(0, var36, 2, 0);
    __builtin_IMCE_SEND(0, var37, 2, 0);
    __builtin_IMCE_SEND(0, var38, 2, 0);


  } // endgenerate: outer_loop
  for (int i1 = 0; i1 < 16; i1++) { // generate: inner_loop

    __builtin_IMCE_STEP();

    var97 = __builtin_IMCE_GET_CREG((short)0);
    var98 = __builtin_IMCE_GET_CREG((short)1);
    var99 = __builtin_IMCE_GET_CREG((short)2);
    var100 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var101 = __builtin_IMCE_RECV(1);
    var102 = __builtin_IMCE_DIV(var97, var101, 15);
    var103 = __builtin_IMCE_RECV(1);
    var104 = __builtin_IMCE_DIV(var98, var103, 15);
    var105 = __builtin_IMCE_RECV(1);
    var106 = __builtin_IMCE_DIV(var99, var105, 15);
    var107 = __builtin_IMCE_RECV(1);
    var108 = __builtin_IMCE_DIV(var100, var107, 15);
    // endgenerate: div


    // generate: add
    var109 = __builtin_IMCE_RECV(2);
    var110 = __builtin_IMCE_ADD(var109, var102, 15);
    var111 = __builtin_IMCE_RECV(2);
    var112 = __builtin_IMCE_ADD(var111, var104, 15);
    var113 = __builtin_IMCE_RECV(2);
    var114 = __builtin_IMCE_ADD(var113, var106, 15);
    var115 = __builtin_IMCE_RECV(2);
    var116 = __builtin_IMCE_ADD(var115, var108, 15);
    // endgenerate: add


    // generate: min_max_quantize
    __builtin_IMCE_MM_QUANT(var117, 0, 15, 4);
    var35 = __builtin_IMCE_GET_QREG(0);
    __builtin_IMCE_MM_QUANT(var118, 0, 15, 5);
    var36 = __builtin_IMCE_GET_QREG(1);
    __builtin_IMCE_MM_QUANT(var119, 0, 15, 6);
    var37 = __builtin_IMCE_GET_QREG(2);
    __builtin_IMCE_MM_QUANT(var120, 0, 15, 7);
    var38 = __builtin_IMCE_GET_QREG(3);
    // endgenerate: min_max_quantize


    __builtin_IMCE_SEND(0, var35, 2, 0);
    __builtin_IMCE_SEND(0, var36, 2, 0);
    __builtin_IMCE_SEND(0, var37, 2, 0);
    __builtin_IMCE_SEND(0, var38, 2, 0);


  } // endgenerate: inner_loop
  // endgenerate: conv exec
}
else if (hid == 2 && wid == 1) {
  // generate: weight write
  for (int i1 = 0; i1 < 8192; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(1);
  } // endgenerate: load_block
  // endgenerate: weight write
  // generate: conv exec
  for (int i1 = 0; i1 < 18; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
  } // endgenerate: load_block
  __builtin_IMCE_STEP();

  var121 = __builtin_IMCE_GET_CREG((short)0);
  var122 = __builtin_IMCE_GET_CREG((short)1);
  var123 = __builtin_IMCE_GET_CREG((short)2);
  var124 = __builtin_IMCE_GET_CREG((short)3);


  // generate: div
  var125 = __builtin_IMCE_RECV(1);
  var126 = __builtin_IMCE_DIV(var121, var125, 15);
  var127 = __builtin_IMCE_RECV(1);
  var128 = __builtin_IMCE_DIV(var122, var127, 15);
  var129 = __builtin_IMCE_RECV(1);
  var130 = __builtin_IMCE_DIV(var123, var129, 15);
  var131 = __builtin_IMCE_RECV(1);
  var132 = __builtin_IMCE_DIV(var124, var131, 15);
  // endgenerate: div


  // generate: add
  var133 = __builtin_IMCE_RECV(2);
  var109 = __builtin_IMCE_ADD(var133, var126, 15);
  var134 = __builtin_IMCE_RECV(2);
  var111 = __builtin_IMCE_ADD(var134, var128, 15);
  var135 = __builtin_IMCE_RECV(2);
  var113 = __builtin_IMCE_ADD(var135, var130, 15);
  var136 = __builtin_IMCE_RECV(2);
  var115 = __builtin_IMCE_ADD(var136, var132, 15);
  // endgenerate: add


  __builtin_IMCE_SEND(0, var109, 2, 0);
  __builtin_IMCE_SEND(0, var111, 2, 0);
  __builtin_IMCE_SEND(0, var113, 2, 0);
  __builtin_IMCE_SEND(0, var115, 2, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: inner_loop
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_STEP();

    var121 = __builtin_IMCE_GET_CREG((short)0);
    var122 = __builtin_IMCE_GET_CREG((short)1);
    var123 = __builtin_IMCE_GET_CREG((short)2);
    var124 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var125 = __builtin_IMCE_RECV(1);
    var126 = __builtin_IMCE_DIV(var121, var125, 15);
    var127 = __builtin_IMCE_RECV(1);
    var128 = __builtin_IMCE_DIV(var122, var127, 15);
    var129 = __builtin_IMCE_RECV(1);
    var130 = __builtin_IMCE_DIV(var123, var129, 15);
    var131 = __builtin_IMCE_RECV(1);
    var132 = __builtin_IMCE_DIV(var124, var131, 15);
    // endgenerate: div


    // generate: add
    var133 = __builtin_IMCE_RECV(2);
    var109 = __builtin_IMCE_ADD(var133, var126, 15);
    var134 = __builtin_IMCE_RECV(2);
    var111 = __builtin_IMCE_ADD(var134, var128, 15);
    var135 = __builtin_IMCE_RECV(2);
    var113 = __builtin_IMCE_ADD(var135, var130, 15);
    var136 = __builtin_IMCE_RECV(2);
    var115 = __builtin_IMCE_ADD(var136, var132, 15);
    // endgenerate: add


    __builtin_IMCE_SEND(0, var109, 2, 0);
    __builtin_IMCE_SEND(0, var111, 2, 0);
    __builtin_IMCE_SEND(0, var113, 2, 0);
    __builtin_IMCE_SEND(0, var115, 2, 0);


  } // endgenerate: inner_loop

  __builtin_IMCE_STEP();

  var121 = __builtin_IMCE_GET_CREG((short)0);
  var122 = __builtin_IMCE_GET_CREG((short)1);
  var123 = __builtin_IMCE_GET_CREG((short)2);
  var124 = __builtin_IMCE_GET_CREG((short)3);


  // generate: div
  var125 = __builtin_IMCE_RECV(1);
  var126 = __builtin_IMCE_DIV(var121, var125, 15);
  var127 = __builtin_IMCE_RECV(1);
  var128 = __builtin_IMCE_DIV(var122, var127, 15);
  var129 = __builtin_IMCE_RECV(1);
  var130 = __builtin_IMCE_DIV(var123, var129, 15);
  var131 = __builtin_IMCE_RECV(1);
  var132 = __builtin_IMCE_DIV(var124, var131, 15);
  // endgenerate: div


  // generate: add
  var133 = __builtin_IMCE_RECV(2);
  var109 = __builtin_IMCE_ADD(var133, var126, 15);
  var134 = __builtin_IMCE_RECV(2);
  var111 = __builtin_IMCE_ADD(var134, var128, 15);
  var135 = __builtin_IMCE_RECV(2);
  var113 = __builtin_IMCE_ADD(var135, var130, 15);
  var136 = __builtin_IMCE_RECV(2);
  var115 = __builtin_IMCE_ADD(var136, var132, 15);
  // endgenerate: add


  __builtin_IMCE_SEND(0, var109, 2, 0);
  __builtin_IMCE_SEND(0, var111, 2, 0);
  __builtin_IMCE_SEND(0, var113, 2, 0);
  __builtin_IMCE_SEND(0, var115, 2, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: outer_loop
    for (int i2 = 0; i2 < 2; i2++) { // generate: load_block
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
    } // endgenerate: load_block
    __builtin_IMCE_STEP();

    var121 = __builtin_IMCE_GET_CREG((short)0);
    var122 = __builtin_IMCE_GET_CREG((short)1);
    var123 = __builtin_IMCE_GET_CREG((short)2);
    var124 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var125 = __builtin_IMCE_RECV(1);
    var126 = __builtin_IMCE_DIV(var121, var125, 15);
    var127 = __builtin_IMCE_RECV(1);
    var128 = __builtin_IMCE_DIV(var122, var127, 15);
    var129 = __builtin_IMCE_RECV(1);
    var130 = __builtin_IMCE_DIV(var123, var129, 15);
    var131 = __builtin_IMCE_RECV(1);
    var132 = __builtin_IMCE_DIV(var124, var131, 15);
    // endgenerate: div


    // generate: add
    var133 = __builtin_IMCE_RECV(2);
    var109 = __builtin_IMCE_ADD(var133, var126, 15);
    var134 = __builtin_IMCE_RECV(2);
    var111 = __builtin_IMCE_ADD(var134, var128, 15);
    var135 = __builtin_IMCE_RECV(2);
    var113 = __builtin_IMCE_ADD(var135, var130, 15);
    var136 = __builtin_IMCE_RECV(2);
    var115 = __builtin_IMCE_ADD(var136, var132, 15);
    // endgenerate: add


    __builtin_IMCE_SEND(0, var109, 2, 0);
    __builtin_IMCE_SEND(0, var111, 2, 0);
    __builtin_IMCE_SEND(0, var113, 2, 0);
    __builtin_IMCE_SEND(0, var115, 2, 0);


    for (int i2 = 0; i2 < 14; i2++) { // generate: inner_loop
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_STEP();

      var121 = __builtin_IMCE_GET_CREG((short)0);
      var122 = __builtin_IMCE_GET_CREG((short)1);
      var123 = __builtin_IMCE_GET_CREG((short)2);
      var124 = __builtin_IMCE_GET_CREG((short)3);


      // generate: div
      var125 = __builtin_IMCE_RECV(1);
      var126 = __builtin_IMCE_DIV(var121, var125, 15);
      var127 = __builtin_IMCE_RECV(1);
      var128 = __builtin_IMCE_DIV(var122, var127, 15);
      var129 = __builtin_IMCE_RECV(1);
      var130 = __builtin_IMCE_DIV(var123, var129, 15);
      var131 = __builtin_IMCE_RECV(1);
      var132 = __builtin_IMCE_DIV(var124, var131, 15);
      // endgenerate: div


      // generate: add
      var133 = __builtin_IMCE_RECV(2);
      var109 = __builtin_IMCE_ADD(var133, var126, 15);
      var134 = __builtin_IMCE_RECV(2);
      var111 = __builtin_IMCE_ADD(var134, var128, 15);
      var135 = __builtin_IMCE_RECV(2);
      var113 = __builtin_IMCE_ADD(var135, var130, 15);
      var136 = __builtin_IMCE_RECV(2);
      var115 = __builtin_IMCE_ADD(var136, var132, 15);
      // endgenerate: add


      __builtin_IMCE_SEND(0, var109, 2, 0);
      __builtin_IMCE_SEND(0, var111, 2, 0);
      __builtin_IMCE_SEND(0, var113, 2, 0);
      __builtin_IMCE_SEND(0, var115, 2, 0);


    } // endgenerate: inner_loop

    __builtin_IMCE_STEP();

    var121 = __builtin_IMCE_GET_CREG((short)0);
    var122 = __builtin_IMCE_GET_CREG((short)1);
    var123 = __builtin_IMCE_GET_CREG((short)2);
    var124 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var125 = __builtin_IMCE_RECV(1);
    var126 = __builtin_IMCE_DIV(var121, var125, 15);
    var127 = __builtin_IMCE_RECV(1);
    var128 = __builtin_IMCE_DIV(var122, var127, 15);
    var129 = __builtin_IMCE_RECV(1);
    var130 = __builtin_IMCE_DIV(var123, var129, 15);
    var131 = __builtin_IMCE_RECV(1);
    var132 = __builtin_IMCE_DIV(var124, var131, 15);
    // endgenerate: div


    // generate: add
    var133 = __builtin_IMCE_RECV(2);
    var109 = __builtin_IMCE_ADD(var133, var126, 15);
    var134 = __builtin_IMCE_RECV(2);
    var111 = __builtin_IMCE_ADD(var134, var128, 15);
    var135 = __builtin_IMCE_RECV(2);
    var113 = __builtin_IMCE_ADD(var135, var130, 15);
    var136 = __builtin_IMCE_RECV(2);
    var115 = __builtin_IMCE_ADD(var136, var132, 15);
    // endgenerate: add


    __builtin_IMCE_SEND(0, var109, 2, 0);
    __builtin_IMCE_SEND(0, var111, 2, 0);
    __builtin_IMCE_SEND(0, var113, 2, 0);
    __builtin_IMCE_SEND(0, var115, 2, 0);


  } // endgenerate: outer_loop
  for (int i1 = 0; i1 < 16; i1++) { // generate: inner_loop

    __builtin_IMCE_STEP();

    var121 = __builtin_IMCE_GET_CREG((short)0);
    var122 = __builtin_IMCE_GET_CREG((short)1);
    var123 = __builtin_IMCE_GET_CREG((short)2);
    var124 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var125 = __builtin_IMCE_RECV(1);
    var126 = __builtin_IMCE_DIV(var121, var125, 15);
    var127 = __builtin_IMCE_RECV(1);
    var128 = __builtin_IMCE_DIV(var122, var127, 15);
    var129 = __builtin_IMCE_RECV(1);
    var130 = __builtin_IMCE_DIV(var123, var129, 15);
    var131 = __builtin_IMCE_RECV(1);
    var132 = __builtin_IMCE_DIV(var124, var131, 15);
    // endgenerate: div


    // generate: add
    var133 = __builtin_IMCE_RECV(2);
    var109 = __builtin_IMCE_ADD(var133, var126, 15);
    var134 = __builtin_IMCE_RECV(2);
    var111 = __builtin_IMCE_ADD(var134, var128, 15);
    var135 = __builtin_IMCE_RECV(2);
    var113 = __builtin_IMCE_ADD(var135, var130, 15);
    var136 = __builtin_IMCE_RECV(2);
    var115 = __builtin_IMCE_ADD(var136, var132, 15);
    // endgenerate: add


    __builtin_IMCE_SEND(0, var109, 2, 0);
    __builtin_IMCE_SEND(0, var111, 2, 0);
    __builtin_IMCE_SEND(0, var113, 2, 0);
    __builtin_IMCE_SEND(0, var115, 2, 0);


  } // endgenerate: inner_loop
  // endgenerate: conv exec
}
else if (hid == 2 && wid == 2) {
  // generate: weight write
  for (int i1 = 0; i1 < 8192; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(1);
  } // endgenerate: load_block
  // endgenerate: weight write
  // generate: conv exec
  for (int i1 = 0; i1 < 18; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
  } // endgenerate: load_block
  __builtin_IMCE_STEP();

  var137 = __builtin_IMCE_GET_CREG((short)0);
  var138 = __builtin_IMCE_GET_CREG((short)1);
  var139 = __builtin_IMCE_GET_CREG((short)2);
  var140 = __builtin_IMCE_GET_CREG((short)3);


  // generate: div
  var141 = __builtin_IMCE_RECV(1);
  var142 = __builtin_IMCE_DIV(var137, var141, 15);
  var143 = __builtin_IMCE_RECV(1);
  var144 = __builtin_IMCE_DIV(var138, var143, 15);
  var145 = __builtin_IMCE_RECV(1);
  var146 = __builtin_IMCE_DIV(var139, var145, 15);
  var147 = __builtin_IMCE_RECV(1);
  var148 = __builtin_IMCE_DIV(var140, var147, 15);
  // endgenerate: div


  // generate: add
  var149 = __builtin_IMCE_RECV(2);
  var133 = __builtin_IMCE_ADD(var149, var142, 15);
  var150 = __builtin_IMCE_RECV(2);
  var134 = __builtin_IMCE_ADD(var150, var144, 15);
  var151 = __builtin_IMCE_RECV(2);
  var135 = __builtin_IMCE_ADD(var151, var146, 15);
  var152 = __builtin_IMCE_RECV(2);
  var136 = __builtin_IMCE_ADD(var152, var148, 15);
  // endgenerate: add


  __builtin_IMCE_SEND(1, var133, 2, 0);
  __builtin_IMCE_SEND(1, var134, 2, 0);
  __builtin_IMCE_SEND(1, var135, 2, 0);
  __builtin_IMCE_SEND(1, var136, 2, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: inner_loop
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_STEP();

    var137 = __builtin_IMCE_GET_CREG((short)0);
    var138 = __builtin_IMCE_GET_CREG((short)1);
    var139 = __builtin_IMCE_GET_CREG((short)2);
    var140 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var141 = __builtin_IMCE_RECV(1);
    var142 = __builtin_IMCE_DIV(var137, var141, 15);
    var143 = __builtin_IMCE_RECV(1);
    var144 = __builtin_IMCE_DIV(var138, var143, 15);
    var145 = __builtin_IMCE_RECV(1);
    var146 = __builtin_IMCE_DIV(var139, var145, 15);
    var147 = __builtin_IMCE_RECV(1);
    var148 = __builtin_IMCE_DIV(var140, var147, 15);
    // endgenerate: div


    // generate: add
    var149 = __builtin_IMCE_RECV(2);
    var133 = __builtin_IMCE_ADD(var149, var142, 15);
    var150 = __builtin_IMCE_RECV(2);
    var134 = __builtin_IMCE_ADD(var150, var144, 15);
    var151 = __builtin_IMCE_RECV(2);
    var135 = __builtin_IMCE_ADD(var151, var146, 15);
    var152 = __builtin_IMCE_RECV(2);
    var136 = __builtin_IMCE_ADD(var152, var148, 15);
    // endgenerate: add


    __builtin_IMCE_SEND(1, var133, 2, 0);
    __builtin_IMCE_SEND(1, var134, 2, 0);
    __builtin_IMCE_SEND(1, var135, 2, 0);
    __builtin_IMCE_SEND(1, var136, 2, 0);


  } // endgenerate: inner_loop

  __builtin_IMCE_STEP();

  var137 = __builtin_IMCE_GET_CREG((short)0);
  var138 = __builtin_IMCE_GET_CREG((short)1);
  var139 = __builtin_IMCE_GET_CREG((short)2);
  var140 = __builtin_IMCE_GET_CREG((short)3);


  // generate: div
  var141 = __builtin_IMCE_RECV(1);
  var142 = __builtin_IMCE_DIV(var137, var141, 15);
  var143 = __builtin_IMCE_RECV(1);
  var144 = __builtin_IMCE_DIV(var138, var143, 15);
  var145 = __builtin_IMCE_RECV(1);
  var146 = __builtin_IMCE_DIV(var139, var145, 15);
  var147 = __builtin_IMCE_RECV(1);
  var148 = __builtin_IMCE_DIV(var140, var147, 15);
  // endgenerate: div


  // generate: add
  var149 = __builtin_IMCE_RECV(2);
  var133 = __builtin_IMCE_ADD(var149, var142, 15);
  var150 = __builtin_IMCE_RECV(2);
  var134 = __builtin_IMCE_ADD(var150, var144, 15);
  var151 = __builtin_IMCE_RECV(2);
  var135 = __builtin_IMCE_ADD(var151, var146, 15);
  var152 = __builtin_IMCE_RECV(2);
  var136 = __builtin_IMCE_ADD(var152, var148, 15);
  // endgenerate: add


  __builtin_IMCE_SEND(1, var133, 2, 0);
  __builtin_IMCE_SEND(1, var134, 2, 0);
  __builtin_IMCE_SEND(1, var135, 2, 0);
  __builtin_IMCE_SEND(1, var136, 2, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: outer_loop
    for (int i2 = 0; i2 < 2; i2++) { // generate: load_block
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
    } // endgenerate: load_block
    __builtin_IMCE_STEP();

    var137 = __builtin_IMCE_GET_CREG((short)0);
    var138 = __builtin_IMCE_GET_CREG((short)1);
    var139 = __builtin_IMCE_GET_CREG((short)2);
    var140 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var141 = __builtin_IMCE_RECV(1);
    var142 = __builtin_IMCE_DIV(var137, var141, 15);
    var143 = __builtin_IMCE_RECV(1);
    var144 = __builtin_IMCE_DIV(var138, var143, 15);
    var145 = __builtin_IMCE_RECV(1);
    var146 = __builtin_IMCE_DIV(var139, var145, 15);
    var147 = __builtin_IMCE_RECV(1);
    var148 = __builtin_IMCE_DIV(var140, var147, 15);
    // endgenerate: div


    // generate: add
    var149 = __builtin_IMCE_RECV(2);
    var133 = __builtin_IMCE_ADD(var149, var142, 15);
    var150 = __builtin_IMCE_RECV(2);
    var134 = __builtin_IMCE_ADD(var150, var144, 15);
    var151 = __builtin_IMCE_RECV(2);
    var135 = __builtin_IMCE_ADD(var151, var146, 15);
    var152 = __builtin_IMCE_RECV(2);
    var136 = __builtin_IMCE_ADD(var152, var148, 15);
    // endgenerate: add


    __builtin_IMCE_SEND(1, var133, 2, 0);
    __builtin_IMCE_SEND(1, var134, 2, 0);
    __builtin_IMCE_SEND(1, var135, 2, 0);
    __builtin_IMCE_SEND(1, var136, 2, 0);


    for (int i2 = 0; i2 < 14; i2++) { // generate: inner_loop
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_STEP();

      var137 = __builtin_IMCE_GET_CREG((short)0);
      var138 = __builtin_IMCE_GET_CREG((short)1);
      var139 = __builtin_IMCE_GET_CREG((short)2);
      var140 = __builtin_IMCE_GET_CREG((short)3);


      // generate: div
      var141 = __builtin_IMCE_RECV(1);
      var142 = __builtin_IMCE_DIV(var137, var141, 15);
      var143 = __builtin_IMCE_RECV(1);
      var144 = __builtin_IMCE_DIV(var138, var143, 15);
      var145 = __builtin_IMCE_RECV(1);
      var146 = __builtin_IMCE_DIV(var139, var145, 15);
      var147 = __builtin_IMCE_RECV(1);
      var148 = __builtin_IMCE_DIV(var140, var147, 15);
      // endgenerate: div


      // generate: add
      var149 = __builtin_IMCE_RECV(2);
      var133 = __builtin_IMCE_ADD(var149, var142, 15);
      var150 = __builtin_IMCE_RECV(2);
      var134 = __builtin_IMCE_ADD(var150, var144, 15);
      var151 = __builtin_IMCE_RECV(2);
      var135 = __builtin_IMCE_ADD(var151, var146, 15);
      var152 = __builtin_IMCE_RECV(2);
      var136 = __builtin_IMCE_ADD(var152, var148, 15);
      // endgenerate: add


      __builtin_IMCE_SEND(1, var133, 2, 0);
      __builtin_IMCE_SEND(1, var134, 2, 0);
      __builtin_IMCE_SEND(1, var135, 2, 0);
      __builtin_IMCE_SEND(1, var136, 2, 0);


    } // endgenerate: inner_loop

    __builtin_IMCE_STEP();

    var137 = __builtin_IMCE_GET_CREG((short)0);
    var138 = __builtin_IMCE_GET_CREG((short)1);
    var139 = __builtin_IMCE_GET_CREG((short)2);
    var140 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var141 = __builtin_IMCE_RECV(1);
    var142 = __builtin_IMCE_DIV(var137, var141, 15);
    var143 = __builtin_IMCE_RECV(1);
    var144 = __builtin_IMCE_DIV(var138, var143, 15);
    var145 = __builtin_IMCE_RECV(1);
    var146 = __builtin_IMCE_DIV(var139, var145, 15);
    var147 = __builtin_IMCE_RECV(1);
    var148 = __builtin_IMCE_DIV(var140, var147, 15);
    // endgenerate: div


    // generate: add
    var149 = __builtin_IMCE_RECV(2);
    var133 = __builtin_IMCE_ADD(var149, var142, 15);
    var150 = __builtin_IMCE_RECV(2);
    var134 = __builtin_IMCE_ADD(var150, var144, 15);
    var151 = __builtin_IMCE_RECV(2);
    var135 = __builtin_IMCE_ADD(var151, var146, 15);
    var152 = __builtin_IMCE_RECV(2);
    var136 = __builtin_IMCE_ADD(var152, var148, 15);
    // endgenerate: add


    __builtin_IMCE_SEND(1, var133, 2, 0);
    __builtin_IMCE_SEND(1, var134, 2, 0);
    __builtin_IMCE_SEND(1, var135, 2, 0);
    __builtin_IMCE_SEND(1, var136, 2, 0);


  } // endgenerate: outer_loop
  for (int i1 = 0; i1 < 16; i1++) { // generate: inner_loop

    __builtin_IMCE_STEP();

    var137 = __builtin_IMCE_GET_CREG((short)0);
    var138 = __builtin_IMCE_GET_CREG((short)1);
    var139 = __builtin_IMCE_GET_CREG((short)2);
    var140 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var141 = __builtin_IMCE_RECV(1);
    var142 = __builtin_IMCE_DIV(var137, var141, 15);
    var143 = __builtin_IMCE_RECV(1);
    var144 = __builtin_IMCE_DIV(var138, var143, 15);
    var145 = __builtin_IMCE_RECV(1);
    var146 = __builtin_IMCE_DIV(var139, var145, 15);
    var147 = __builtin_IMCE_RECV(1);
    var148 = __builtin_IMCE_DIV(var140, var147, 15);
    // endgenerate: div


    // generate: add
    var149 = __builtin_IMCE_RECV(2);
    var133 = __builtin_IMCE_ADD(var149, var142, 15);
    var150 = __builtin_IMCE_RECV(2);
    var134 = __builtin_IMCE_ADD(var150, var144, 15);
    var151 = __builtin_IMCE_RECV(2);
    var135 = __builtin_IMCE_ADD(var151, var146, 15);
    var152 = __builtin_IMCE_RECV(2);
    var136 = __builtin_IMCE_ADD(var152, var148, 15);
    // endgenerate: add


    __builtin_IMCE_SEND(1, var133, 2, 0);
    __builtin_IMCE_SEND(1, var134, 2, 0);
    __builtin_IMCE_SEND(1, var135, 2, 0);
    __builtin_IMCE_SEND(1, var136, 2, 0);


  } // endgenerate: inner_loop
  // endgenerate: conv exec
}
else if (hid == 2 && wid == 3) {
  // generate: weight write
  for (int i1 = 0; i1 < 8192; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(1);
  } // endgenerate: load_block
  // endgenerate: weight write
  // generate: conv exec
  for (int i1 = 0; i1 < 18; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
  } // endgenerate: load_block
  __builtin_IMCE_STEP();

  var153 = __builtin_IMCE_GET_CREG((short)0);
  var154 = __builtin_IMCE_GET_CREG((short)1);
  var155 = __builtin_IMCE_GET_CREG((short)2);
  var156 = __builtin_IMCE_GET_CREG((short)3);


  // generate: div
  var157 = __builtin_IMCE_RECV(1);
  var158 = __builtin_IMCE_DIV(var153, var157, 15);
  var159 = __builtin_IMCE_RECV(1);
  var160 = __builtin_IMCE_DIV(var154, var159, 15);
  var161 = __builtin_IMCE_RECV(1);
  var162 = __builtin_IMCE_DIV(var155, var161, 15);
  var163 = __builtin_IMCE_RECV(1);
  var164 = __builtin_IMCE_DIV(var156, var163, 15);
  // endgenerate: div


  // generate: add
  var165 = __builtin_IMCE_RECV(2);
  var149 = __builtin_IMCE_ADD(var158, var165, 15);
  var166 = __builtin_IMCE_RECV(2);
  var150 = __builtin_IMCE_ADD(var160, var166, 15);
  var167 = __builtin_IMCE_RECV(2);
  var151 = __builtin_IMCE_ADD(var162, var167, 15);
  var168 = __builtin_IMCE_RECV(2);
  var152 = __builtin_IMCE_ADD(var164, var168, 15);
  // endgenerate: add


  __builtin_IMCE_SEND(1, var149, 2, 0);
  __builtin_IMCE_SEND(1, var150, 2, 0);
  __builtin_IMCE_SEND(1, var151, 2, 0);
  __builtin_IMCE_SEND(1, var152, 2, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: inner_loop
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_STEP();

    var153 = __builtin_IMCE_GET_CREG((short)0);
    var154 = __builtin_IMCE_GET_CREG((short)1);
    var155 = __builtin_IMCE_GET_CREG((short)2);
    var156 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var157 = __builtin_IMCE_RECV(1);
    var158 = __builtin_IMCE_DIV(var153, var157, 15);
    var159 = __builtin_IMCE_RECV(1);
    var160 = __builtin_IMCE_DIV(var154, var159, 15);
    var161 = __builtin_IMCE_RECV(1);
    var162 = __builtin_IMCE_DIV(var155, var161, 15);
    var163 = __builtin_IMCE_RECV(1);
    var164 = __builtin_IMCE_DIV(var156, var163, 15);
    // endgenerate: div


    // generate: add
    var165 = __builtin_IMCE_RECV(2);
    var149 = __builtin_IMCE_ADD(var158, var165, 15);
    var166 = __builtin_IMCE_RECV(2);
    var150 = __builtin_IMCE_ADD(var160, var166, 15);
    var167 = __builtin_IMCE_RECV(2);
    var151 = __builtin_IMCE_ADD(var162, var167, 15);
    var168 = __builtin_IMCE_RECV(2);
    var152 = __builtin_IMCE_ADD(var164, var168, 15);
    // endgenerate: add


    __builtin_IMCE_SEND(1, var149, 2, 0);
    __builtin_IMCE_SEND(1, var150, 2, 0);
    __builtin_IMCE_SEND(1, var151, 2, 0);
    __builtin_IMCE_SEND(1, var152, 2, 0);


  } // endgenerate: inner_loop

  __builtin_IMCE_STEP();

  var153 = __builtin_IMCE_GET_CREG((short)0);
  var154 = __builtin_IMCE_GET_CREG((short)1);
  var155 = __builtin_IMCE_GET_CREG((short)2);
  var156 = __builtin_IMCE_GET_CREG((short)3);


  // generate: div
  var157 = __builtin_IMCE_RECV(1);
  var158 = __builtin_IMCE_DIV(var153, var157, 15);
  var159 = __builtin_IMCE_RECV(1);
  var160 = __builtin_IMCE_DIV(var154, var159, 15);
  var161 = __builtin_IMCE_RECV(1);
  var162 = __builtin_IMCE_DIV(var155, var161, 15);
  var163 = __builtin_IMCE_RECV(1);
  var164 = __builtin_IMCE_DIV(var156, var163, 15);
  // endgenerate: div


  // generate: add
  var165 = __builtin_IMCE_RECV(2);
  var149 = __builtin_IMCE_ADD(var158, var165, 15);
  var166 = __builtin_IMCE_RECV(2);
  var150 = __builtin_IMCE_ADD(var160, var166, 15);
  var167 = __builtin_IMCE_RECV(2);
  var151 = __builtin_IMCE_ADD(var162, var167, 15);
  var168 = __builtin_IMCE_RECV(2);
  var152 = __builtin_IMCE_ADD(var164, var168, 15);
  // endgenerate: add


  __builtin_IMCE_SEND(1, var149, 2, 0);
  __builtin_IMCE_SEND(1, var150, 2, 0);
  __builtin_IMCE_SEND(1, var151, 2, 0);
  __builtin_IMCE_SEND(1, var152, 2, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: outer_loop
    for (int i2 = 0; i2 < 2; i2++) { // generate: load_block
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
    } // endgenerate: load_block
    __builtin_IMCE_STEP();

    var153 = __builtin_IMCE_GET_CREG((short)0);
    var154 = __builtin_IMCE_GET_CREG((short)1);
    var155 = __builtin_IMCE_GET_CREG((short)2);
    var156 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var157 = __builtin_IMCE_RECV(1);
    var158 = __builtin_IMCE_DIV(var153, var157, 15);
    var159 = __builtin_IMCE_RECV(1);
    var160 = __builtin_IMCE_DIV(var154, var159, 15);
    var161 = __builtin_IMCE_RECV(1);
    var162 = __builtin_IMCE_DIV(var155, var161, 15);
    var163 = __builtin_IMCE_RECV(1);
    var164 = __builtin_IMCE_DIV(var156, var163, 15);
    // endgenerate: div


    // generate: add
    var165 = __builtin_IMCE_RECV(2);
    var149 = __builtin_IMCE_ADD(var158, var165, 15);
    var166 = __builtin_IMCE_RECV(2);
    var150 = __builtin_IMCE_ADD(var160, var166, 15);
    var167 = __builtin_IMCE_RECV(2);
    var151 = __builtin_IMCE_ADD(var162, var167, 15);
    var168 = __builtin_IMCE_RECV(2);
    var152 = __builtin_IMCE_ADD(var164, var168, 15);
    // endgenerate: add


    __builtin_IMCE_SEND(1, var149, 2, 0);
    __builtin_IMCE_SEND(1, var150, 2, 0);
    __builtin_IMCE_SEND(1, var151, 2, 0);
    __builtin_IMCE_SEND(1, var152, 2, 0);


    for (int i2 = 0; i2 < 14; i2++) { // generate: inner_loop
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_STEP();

      var153 = __builtin_IMCE_GET_CREG((short)0);
      var154 = __builtin_IMCE_GET_CREG((short)1);
      var155 = __builtin_IMCE_GET_CREG((short)2);
      var156 = __builtin_IMCE_GET_CREG((short)3);


      // generate: div
      var157 = __builtin_IMCE_RECV(1);
      var158 = __builtin_IMCE_DIV(var153, var157, 15);
      var159 = __builtin_IMCE_RECV(1);
      var160 = __builtin_IMCE_DIV(var154, var159, 15);
      var161 = __builtin_IMCE_RECV(1);
      var162 = __builtin_IMCE_DIV(var155, var161, 15);
      var163 = __builtin_IMCE_RECV(1);
      var164 = __builtin_IMCE_DIV(var156, var163, 15);
      // endgenerate: div


      // generate: add
      var165 = __builtin_IMCE_RECV(2);
      var149 = __builtin_IMCE_ADD(var158, var165, 15);
      var166 = __builtin_IMCE_RECV(2);
      var150 = __builtin_IMCE_ADD(var160, var166, 15);
      var167 = __builtin_IMCE_RECV(2);
      var151 = __builtin_IMCE_ADD(var162, var167, 15);
      var168 = __builtin_IMCE_RECV(2);
      var152 = __builtin_IMCE_ADD(var164, var168, 15);
      // endgenerate: add


      __builtin_IMCE_SEND(1, var149, 2, 0);
      __builtin_IMCE_SEND(1, var150, 2, 0);
      __builtin_IMCE_SEND(1, var151, 2, 0);
      __builtin_IMCE_SEND(1, var152, 2, 0);


    } // endgenerate: inner_loop

    __builtin_IMCE_STEP();

    var153 = __builtin_IMCE_GET_CREG((short)0);
    var154 = __builtin_IMCE_GET_CREG((short)1);
    var155 = __builtin_IMCE_GET_CREG((short)2);
    var156 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var157 = __builtin_IMCE_RECV(1);
    var158 = __builtin_IMCE_DIV(var153, var157, 15);
    var159 = __builtin_IMCE_RECV(1);
    var160 = __builtin_IMCE_DIV(var154, var159, 15);
    var161 = __builtin_IMCE_RECV(1);
    var162 = __builtin_IMCE_DIV(var155, var161, 15);
    var163 = __builtin_IMCE_RECV(1);
    var164 = __builtin_IMCE_DIV(var156, var163, 15);
    // endgenerate: div


    // generate: add
    var165 = __builtin_IMCE_RECV(2);
    var149 = __builtin_IMCE_ADD(var158, var165, 15);
    var166 = __builtin_IMCE_RECV(2);
    var150 = __builtin_IMCE_ADD(var160, var166, 15);
    var167 = __builtin_IMCE_RECV(2);
    var151 = __builtin_IMCE_ADD(var162, var167, 15);
    var168 = __builtin_IMCE_RECV(2);
    var152 = __builtin_IMCE_ADD(var164, var168, 15);
    // endgenerate: add


    __builtin_IMCE_SEND(1, var149, 2, 0);
    __builtin_IMCE_SEND(1, var150, 2, 0);
    __builtin_IMCE_SEND(1, var151, 2, 0);
    __builtin_IMCE_SEND(1, var152, 2, 0);


  } // endgenerate: outer_loop
  for (int i1 = 0; i1 < 16; i1++) { // generate: inner_loop

    __builtin_IMCE_STEP();

    var153 = __builtin_IMCE_GET_CREG((short)0);
    var154 = __builtin_IMCE_GET_CREG((short)1);
    var155 = __builtin_IMCE_GET_CREG((short)2);
    var156 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var157 = __builtin_IMCE_RECV(1);
    var158 = __builtin_IMCE_DIV(var153, var157, 15);
    var159 = __builtin_IMCE_RECV(1);
    var160 = __builtin_IMCE_DIV(var154, var159, 15);
    var161 = __builtin_IMCE_RECV(1);
    var162 = __builtin_IMCE_DIV(var155, var161, 15);
    var163 = __builtin_IMCE_RECV(1);
    var164 = __builtin_IMCE_DIV(var156, var163, 15);
    // endgenerate: div


    // generate: add
    var165 = __builtin_IMCE_RECV(2);
    var149 = __builtin_IMCE_ADD(var158, var165, 15);
    var166 = __builtin_IMCE_RECV(2);
    var150 = __builtin_IMCE_ADD(var160, var166, 15);
    var167 = __builtin_IMCE_RECV(2);
    var151 = __builtin_IMCE_ADD(var162, var167, 15);
    var168 = __builtin_IMCE_RECV(2);
    var152 = __builtin_IMCE_ADD(var164, var168, 15);
    // endgenerate: add


    __builtin_IMCE_SEND(1, var149, 2, 0);
    __builtin_IMCE_SEND(1, var150, 2, 0);
    __builtin_IMCE_SEND(1, var151, 2, 0);
    __builtin_IMCE_SEND(1, var152, 2, 0);


  } // endgenerate: inner_loop
  // endgenerate: conv exec
}
else if (hid == 2 && wid == 4) {
  // generate: weight write
  for (int i1 = 0; i1 < 8192; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(1);
  } // endgenerate: load_block
  // endgenerate: weight write
  // generate: conv exec
  for (int i1 = 0; i1 < 18; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
  } // endgenerate: load_block
  __builtin_IMCE_STEP();

  var169 = __builtin_IMCE_GET_CREG((short)0);
  var170 = __builtin_IMCE_GET_CREG((short)1);
  var171 = __builtin_IMCE_GET_CREG((short)2);
  var172 = __builtin_IMCE_GET_CREG((short)3);


  // generate: div
  var173 = __builtin_IMCE_RECV(1);
  var165 = __builtin_IMCE_DIV(var169, var173, 15);
  var174 = __builtin_IMCE_RECV(1);
  var166 = __builtin_IMCE_DIV(var170, var174, 15);
  var175 = __builtin_IMCE_RECV(1);
  var167 = __builtin_IMCE_DIV(var171, var175, 15);
  var176 = __builtin_IMCE_RECV(1);
  var168 = __builtin_IMCE_DIV(var172, var176, 15);
  // endgenerate: div


  __builtin_IMCE_SEND(2, var165, 2, 0);
  __builtin_IMCE_SEND(2, var166, 2, 0);
  __builtin_IMCE_SEND(2, var167, 2, 0);
  __builtin_IMCE_SEND(2, var168, 2, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: inner_loop
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_STEP();

    var169 = __builtin_IMCE_GET_CREG((short)0);
    var170 = __builtin_IMCE_GET_CREG((short)1);
    var171 = __builtin_IMCE_GET_CREG((short)2);
    var172 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var173 = __builtin_IMCE_RECV(1);
    var165 = __builtin_IMCE_DIV(var169, var173, 15);
    var174 = __builtin_IMCE_RECV(1);
    var166 = __builtin_IMCE_DIV(var170, var174, 15);
    var175 = __builtin_IMCE_RECV(1);
    var167 = __builtin_IMCE_DIV(var171, var175, 15);
    var176 = __builtin_IMCE_RECV(1);
    var168 = __builtin_IMCE_DIV(var172, var176, 15);
    // endgenerate: div


    __builtin_IMCE_SEND(2, var165, 2, 0);
    __builtin_IMCE_SEND(2, var166, 2, 0);
    __builtin_IMCE_SEND(2, var167, 2, 0);
    __builtin_IMCE_SEND(2, var168, 2, 0);


  } // endgenerate: inner_loop

  __builtin_IMCE_STEP();

  var169 = __builtin_IMCE_GET_CREG((short)0);
  var170 = __builtin_IMCE_GET_CREG((short)1);
  var171 = __builtin_IMCE_GET_CREG((short)2);
  var172 = __builtin_IMCE_GET_CREG((short)3);


  // generate: div
  var173 = __builtin_IMCE_RECV(1);
  var165 = __builtin_IMCE_DIV(var169, var173, 15);
  var174 = __builtin_IMCE_RECV(1);
  var166 = __builtin_IMCE_DIV(var170, var174, 15);
  var175 = __builtin_IMCE_RECV(1);
  var167 = __builtin_IMCE_DIV(var171, var175, 15);
  var176 = __builtin_IMCE_RECV(1);
  var168 = __builtin_IMCE_DIV(var172, var176, 15);
  // endgenerate: div


  __builtin_IMCE_SEND(2, var165, 2, 0);
  __builtin_IMCE_SEND(2, var166, 2, 0);
  __builtin_IMCE_SEND(2, var167, 2, 0);
  __builtin_IMCE_SEND(2, var168, 2, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: outer_loop
    for (int i2 = 0; i2 < 2; i2++) { // generate: load_block
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
    } // endgenerate: load_block
    __builtin_IMCE_STEP();

    var169 = __builtin_IMCE_GET_CREG((short)0);
    var170 = __builtin_IMCE_GET_CREG((short)1);
    var171 = __builtin_IMCE_GET_CREG((short)2);
    var172 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var173 = __builtin_IMCE_RECV(1);
    var165 = __builtin_IMCE_DIV(var169, var173, 15);
    var174 = __builtin_IMCE_RECV(1);
    var166 = __builtin_IMCE_DIV(var170, var174, 15);
    var175 = __builtin_IMCE_RECV(1);
    var167 = __builtin_IMCE_DIV(var171, var175, 15);
    var176 = __builtin_IMCE_RECV(1);
    var168 = __builtin_IMCE_DIV(var172, var176, 15);
    // endgenerate: div


    __builtin_IMCE_SEND(2, var165, 2, 0);
    __builtin_IMCE_SEND(2, var166, 2, 0);
    __builtin_IMCE_SEND(2, var167, 2, 0);
    __builtin_IMCE_SEND(2, var168, 2, 0);


    for (int i2 = 0; i2 < 14; i2++) { // generate: inner_loop
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_STEP();

      var169 = __builtin_IMCE_GET_CREG((short)0);
      var170 = __builtin_IMCE_GET_CREG((short)1);
      var171 = __builtin_IMCE_GET_CREG((short)2);
      var172 = __builtin_IMCE_GET_CREG((short)3);


      // generate: div
      var173 = __builtin_IMCE_RECV(1);
      var165 = __builtin_IMCE_DIV(var169, var173, 15);
      var174 = __builtin_IMCE_RECV(1);
      var166 = __builtin_IMCE_DIV(var170, var174, 15);
      var175 = __builtin_IMCE_RECV(1);
      var167 = __builtin_IMCE_DIV(var171, var175, 15);
      var176 = __builtin_IMCE_RECV(1);
      var168 = __builtin_IMCE_DIV(var172, var176, 15);
      // endgenerate: div


      __builtin_IMCE_SEND(2, var165, 2, 0);
      __builtin_IMCE_SEND(2, var166, 2, 0);
      __builtin_IMCE_SEND(2, var167, 2, 0);
      __builtin_IMCE_SEND(2, var168, 2, 0);


    } // endgenerate: inner_loop

    __builtin_IMCE_STEP();

    var169 = __builtin_IMCE_GET_CREG((short)0);
    var170 = __builtin_IMCE_GET_CREG((short)1);
    var171 = __builtin_IMCE_GET_CREG((short)2);
    var172 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var173 = __builtin_IMCE_RECV(1);
    var165 = __builtin_IMCE_DIV(var169, var173, 15);
    var174 = __builtin_IMCE_RECV(1);
    var166 = __builtin_IMCE_DIV(var170, var174, 15);
    var175 = __builtin_IMCE_RECV(1);
    var167 = __builtin_IMCE_DIV(var171, var175, 15);
    var176 = __builtin_IMCE_RECV(1);
    var168 = __builtin_IMCE_DIV(var172, var176, 15);
    // endgenerate: div


    __builtin_IMCE_SEND(2, var165, 2, 0);
    __builtin_IMCE_SEND(2, var166, 2, 0);
    __builtin_IMCE_SEND(2, var167, 2, 0);
    __builtin_IMCE_SEND(2, var168, 2, 0);


  } // endgenerate: outer_loop
  for (int i1 = 0; i1 < 16; i1++) { // generate: inner_loop

    __builtin_IMCE_STEP();

    var169 = __builtin_IMCE_GET_CREG((short)0);
    var170 = __builtin_IMCE_GET_CREG((short)1);
    var171 = __builtin_IMCE_GET_CREG((short)2);
    var172 = __builtin_IMCE_GET_CREG((short)3);


    // generate: div
    var173 = __builtin_IMCE_RECV(1);
    var165 = __builtin_IMCE_DIV(var169, var173, 15);
    var174 = __builtin_IMCE_RECV(1);
    var166 = __builtin_IMCE_DIV(var170, var174, 15);
    var175 = __builtin_IMCE_RECV(1);
    var167 = __builtin_IMCE_DIV(var171, var175, 15);
    var176 = __builtin_IMCE_RECV(1);
    var168 = __builtin_IMCE_DIV(var172, var176, 15);
    // endgenerate: div


    __builtin_IMCE_SEND(2, var165, 2, 0);
    __builtin_IMCE_SEND(2, var166, 2, 0);
    __builtin_IMCE_SEND(2, var167, 2, 0);
    __builtin_IMCE_SEND(2, var168, 2, 0);


  } // endgenerate: inner_loop
  // endgenerate: conv exec
}
else if (hid == 3 && wid == 1) {
  // generate: weight write
  for (int i1 = 0; i1 < 8192; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(1);
  } // endgenerate: load_block
  // endgenerate: weight write
  // generate: min write
  var177 = __builtin_IMCE_RECV(1);
  // endgenerate: min write
  // generate: max write
  var178 = __builtin_IMCE_RECV(1);
  // endgenerate: max write
  // generate: conv exec
  for (int i1 = 0; i1 < 18; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
  } // endgenerate: load_block
  __builtin_IMCE_STEP();

  var179 = __builtin_IMCE_GET_CREG((short)0);
  var180 = __builtin_IMCE_GET_CREG((short)1);
  var181 = __builtin_IMCE_GET_CREG((short)2);
  var182 = __builtin_IMCE_GET_CREG((short)3);


  // generate: add
  var187 = __builtin_IMCE_RECV(3);
  var188 = __builtin_IMCE_ADD(var179, var187, 15);
  var189 = __builtin_IMCE_RECV(3);
  var190 = __builtin_IMCE_ADD(var180, var189, 15);
  var191 = __builtin_IMCE_RECV(3);
  var192 = __builtin_IMCE_ADD(var181, var191, 15);
  var193 = __builtin_IMCE_RECV(3);
  var194 = __builtin_IMCE_ADD(var182, var193, 15);
  // endgenerate: add


  // generate: min_max_quantize
  __builtin_IMCE_MM_QUANT(var195, 0, 15, 4);
  var196 = __builtin_IMCE_GET_QREG(0);
  __builtin_IMCE_MM_QUANT(var197, 0, 15, 5);
  var198 = __builtin_IMCE_GET_QREG(1);
  __builtin_IMCE_MM_QUANT(var199, 0, 15, 6);
  var200 = __builtin_IMCE_GET_QREG(2);
  __builtin_IMCE_MM_QUANT(var201, 0, 15, 7);
  var202 = __builtin_IMCE_GET_QREG(3);
  // endgenerate: min_max_quantize


  // generate: concat
  var204 = __builtin_IMCE_RECV(2);
  var203 = __builtin_IMCE_OR(var196, var204, 15);
  var206 = __builtin_IMCE_RECV(2);
  var205 = __builtin_IMCE_OR(var198, var206, 15);
  var208 = __builtin_IMCE_RECV(2);
  var207 = __builtin_IMCE_OR(var200, var208, 15);
  var210 = __builtin_IMCE_RECV(2);
  var209 = __builtin_IMCE_OR(var202, var210, 15);
  // endgenerate: concat


  // generate: split
  // endgenerate: split


  __builtin_IMCE_SEND(0, var183, 0, 0);
  __builtin_IMCE_SEND(0, var184, 0, 0);
  __builtin_IMCE_SEND(0, var185, 0, 0);
  __builtin_IMCE_SEND(0, var186, 0, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: inner_loop
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_STEP();

    var179 = __builtin_IMCE_GET_CREG((short)0);
    var180 = __builtin_IMCE_GET_CREG((short)1);
    var181 = __builtin_IMCE_GET_CREG((short)2);
    var182 = __builtin_IMCE_GET_CREG((short)3);


    // generate: add
    var187 = __builtin_IMCE_RECV(3);
    var188 = __builtin_IMCE_ADD(var179, var187, 15);
    var189 = __builtin_IMCE_RECV(3);
    var190 = __builtin_IMCE_ADD(var180, var189, 15);
    var191 = __builtin_IMCE_RECV(3);
    var192 = __builtin_IMCE_ADD(var181, var191, 15);
    var193 = __builtin_IMCE_RECV(3);
    var194 = __builtin_IMCE_ADD(var182, var193, 15);
    // endgenerate: add


    // generate: min_max_quantize
    __builtin_IMCE_MM_QUANT(var195, 0, 15, 4);
    var196 = __builtin_IMCE_GET_QREG(0);
    __builtin_IMCE_MM_QUANT(var197, 0, 15, 5);
    var198 = __builtin_IMCE_GET_QREG(1);
    __builtin_IMCE_MM_QUANT(var199, 0, 15, 6);
    var200 = __builtin_IMCE_GET_QREG(2);
    __builtin_IMCE_MM_QUANT(var201, 0, 15, 7);
    var202 = __builtin_IMCE_GET_QREG(3);
    // endgenerate: min_max_quantize


    // generate: concat
    var204 = __builtin_IMCE_RECV(2);
    var203 = __builtin_IMCE_OR(var196, var204, 15);
    var206 = __builtin_IMCE_RECV(2);
    var205 = __builtin_IMCE_OR(var198, var206, 15);
    var208 = __builtin_IMCE_RECV(2);
    var207 = __builtin_IMCE_OR(var200, var208, 15);
    var210 = __builtin_IMCE_RECV(2);
    var209 = __builtin_IMCE_OR(var202, var210, 15);
    // endgenerate: concat


    // generate: split
    // endgenerate: split


    __builtin_IMCE_SEND(0, var183, 0, 0);
    __builtin_IMCE_SEND(0, var184, 0, 0);
    __builtin_IMCE_SEND(0, var185, 0, 0);
    __builtin_IMCE_SEND(0, var186, 0, 0);


  } // endgenerate: inner_loop

  __builtin_IMCE_STEP();

  var179 = __builtin_IMCE_GET_CREG((short)0);
  var180 = __builtin_IMCE_GET_CREG((short)1);
  var181 = __builtin_IMCE_GET_CREG((short)2);
  var182 = __builtin_IMCE_GET_CREG((short)3);


  // generate: add
  var187 = __builtin_IMCE_RECV(3);
  var188 = __builtin_IMCE_ADD(var179, var187, 15);
  var189 = __builtin_IMCE_RECV(3);
  var190 = __builtin_IMCE_ADD(var180, var189, 15);
  var191 = __builtin_IMCE_RECV(3);
  var192 = __builtin_IMCE_ADD(var181, var191, 15);
  var193 = __builtin_IMCE_RECV(3);
  var194 = __builtin_IMCE_ADD(var182, var193, 15);
  // endgenerate: add


  // generate: min_max_quantize
  __builtin_IMCE_MM_QUANT(var195, 0, 15, 4);
  var196 = __builtin_IMCE_GET_QREG(0);
  __builtin_IMCE_MM_QUANT(var197, 0, 15, 5);
  var198 = __builtin_IMCE_GET_QREG(1);
  __builtin_IMCE_MM_QUANT(var199, 0, 15, 6);
  var200 = __builtin_IMCE_GET_QREG(2);
  __builtin_IMCE_MM_QUANT(var201, 0, 15, 7);
  var202 = __builtin_IMCE_GET_QREG(3);
  // endgenerate: min_max_quantize


  // generate: concat
  var204 = __builtin_IMCE_RECV(2);
  var203 = __builtin_IMCE_OR(var196, var204, 15);
  var206 = __builtin_IMCE_RECV(2);
  var205 = __builtin_IMCE_OR(var198, var206, 15);
  var208 = __builtin_IMCE_RECV(2);
  var207 = __builtin_IMCE_OR(var200, var208, 15);
  var210 = __builtin_IMCE_RECV(2);
  var209 = __builtin_IMCE_OR(var202, var210, 15);
  // endgenerate: concat


  // generate: split
  // endgenerate: split


  __builtin_IMCE_SEND(0, var183, 0, 0);
  __builtin_IMCE_SEND(0, var184, 0, 0);
  __builtin_IMCE_SEND(0, var185, 0, 0);
  __builtin_IMCE_SEND(0, var186, 0, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: outer_loop
    for (int i2 = 0; i2 < 2; i2++) { // generate: load_block
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
    } // endgenerate: load_block
    __builtin_IMCE_STEP();

    var179 = __builtin_IMCE_GET_CREG((short)0);
    var180 = __builtin_IMCE_GET_CREG((short)1);
    var181 = __builtin_IMCE_GET_CREG((short)2);
    var182 = __builtin_IMCE_GET_CREG((short)3);


    // generate: add
    var187 = __builtin_IMCE_RECV(3);
    var188 = __builtin_IMCE_ADD(var179, var187, 15);
    var189 = __builtin_IMCE_RECV(3);
    var190 = __builtin_IMCE_ADD(var180, var189, 15);
    var191 = __builtin_IMCE_RECV(3);
    var192 = __builtin_IMCE_ADD(var181, var191, 15);
    var193 = __builtin_IMCE_RECV(3);
    var194 = __builtin_IMCE_ADD(var182, var193, 15);
    // endgenerate: add


    // generate: min_max_quantize
    __builtin_IMCE_MM_QUANT(var195, 0, 15, 4);
    var196 = __builtin_IMCE_GET_QREG(0);
    __builtin_IMCE_MM_QUANT(var197, 0, 15, 5);
    var198 = __builtin_IMCE_GET_QREG(1);
    __builtin_IMCE_MM_QUANT(var199, 0, 15, 6);
    var200 = __builtin_IMCE_GET_QREG(2);
    __builtin_IMCE_MM_QUANT(var201, 0, 15, 7);
    var202 = __builtin_IMCE_GET_QREG(3);
    // endgenerate: min_max_quantize


    // generate: concat
    var204 = __builtin_IMCE_RECV(2);
    var203 = __builtin_IMCE_OR(var196, var204, 15);
    var206 = __builtin_IMCE_RECV(2);
    var205 = __builtin_IMCE_OR(var198, var206, 15);
    var208 = __builtin_IMCE_RECV(2);
    var207 = __builtin_IMCE_OR(var200, var208, 15);
    var210 = __builtin_IMCE_RECV(2);
    var209 = __builtin_IMCE_OR(var202, var210, 15);
    // endgenerate: concat


    // generate: split
    // endgenerate: split


    __builtin_IMCE_SEND(0, var183, 0, 0);
    __builtin_IMCE_SEND(0, var184, 0, 0);
    __builtin_IMCE_SEND(0, var185, 0, 0);
    __builtin_IMCE_SEND(0, var186, 0, 0);


    for (int i2 = 0; i2 < 14; i2++) { // generate: inner_loop
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_STEP();

      var179 = __builtin_IMCE_GET_CREG((short)0);
      var180 = __builtin_IMCE_GET_CREG((short)1);
      var181 = __builtin_IMCE_GET_CREG((short)2);
      var182 = __builtin_IMCE_GET_CREG((short)3);


      // generate: add
      var187 = __builtin_IMCE_RECV(3);
      var188 = __builtin_IMCE_ADD(var179, var187, 15);
      var189 = __builtin_IMCE_RECV(3);
      var190 = __builtin_IMCE_ADD(var180, var189, 15);
      var191 = __builtin_IMCE_RECV(3);
      var192 = __builtin_IMCE_ADD(var181, var191, 15);
      var193 = __builtin_IMCE_RECV(3);
      var194 = __builtin_IMCE_ADD(var182, var193, 15);
      // endgenerate: add


      // generate: min_max_quantize
      __builtin_IMCE_MM_QUANT(var195, 0, 15, 4);
      var196 = __builtin_IMCE_GET_QREG(0);
      __builtin_IMCE_MM_QUANT(var197, 0, 15, 5);
      var198 = __builtin_IMCE_GET_QREG(1);
      __builtin_IMCE_MM_QUANT(var199, 0, 15, 6);
      var200 = __builtin_IMCE_GET_QREG(2);
      __builtin_IMCE_MM_QUANT(var201, 0, 15, 7);
      var202 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize


      // generate: concat
      var204 = __builtin_IMCE_RECV(2);
      var203 = __builtin_IMCE_OR(var196, var204, 15);
      var206 = __builtin_IMCE_RECV(2);
      var205 = __builtin_IMCE_OR(var198, var206, 15);
      var208 = __builtin_IMCE_RECV(2);
      var207 = __builtin_IMCE_OR(var200, var208, 15);
      var210 = __builtin_IMCE_RECV(2);
      var209 = __builtin_IMCE_OR(var202, var210, 15);
      // endgenerate: concat


      // generate: split
      // endgenerate: split


      __builtin_IMCE_SEND(0, var183, 0, 0);
      __builtin_IMCE_SEND(0, var184, 0, 0);
      __builtin_IMCE_SEND(0, var185, 0, 0);
      __builtin_IMCE_SEND(0, var186, 0, 0);


    } // endgenerate: inner_loop

    __builtin_IMCE_STEP();

    var179 = __builtin_IMCE_GET_CREG((short)0);
    var180 = __builtin_IMCE_GET_CREG((short)1);
    var181 = __builtin_IMCE_GET_CREG((short)2);
    var182 = __builtin_IMCE_GET_CREG((short)3);


    // generate: add
    var187 = __builtin_IMCE_RECV(3);
    var188 = __builtin_IMCE_ADD(var179, var187, 15);
    var189 = __builtin_IMCE_RECV(3);
    var190 = __builtin_IMCE_ADD(var180, var189, 15);
    var191 = __builtin_IMCE_RECV(3);
    var192 = __builtin_IMCE_ADD(var181, var191, 15);
    var193 = __builtin_IMCE_RECV(3);
    var194 = __builtin_IMCE_ADD(var182, var193, 15);
    // endgenerate: add


    // generate: min_max_quantize
    __builtin_IMCE_MM_QUANT(var195, 0, 15, 4);
    var196 = __builtin_IMCE_GET_QREG(0);
    __builtin_IMCE_MM_QUANT(var197, 0, 15, 5);
    var198 = __builtin_IMCE_GET_QREG(1);
    __builtin_IMCE_MM_QUANT(var199, 0, 15, 6);
    var200 = __builtin_IMCE_GET_QREG(2);
    __builtin_IMCE_MM_QUANT(var201, 0, 15, 7);
    var202 = __builtin_IMCE_GET_QREG(3);
    // endgenerate: min_max_quantize


    // generate: concat
    var204 = __builtin_IMCE_RECV(2);
    var203 = __builtin_IMCE_OR(var196, var204, 15);
    var206 = __builtin_IMCE_RECV(2);
    var205 = __builtin_IMCE_OR(var198, var206, 15);
    var208 = __builtin_IMCE_RECV(2);
    var207 = __builtin_IMCE_OR(var200, var208, 15);
    var210 = __builtin_IMCE_RECV(2);
    var209 = __builtin_IMCE_OR(var202, var210, 15);
    // endgenerate: concat


    // generate: split
    // endgenerate: split


    __builtin_IMCE_SEND(0, var183, 0, 0);
    __builtin_IMCE_SEND(0, var184, 0, 0);
    __builtin_IMCE_SEND(0, var185, 0, 0);
    __builtin_IMCE_SEND(0, var186, 0, 0);


  } // endgenerate: outer_loop
  for (int i1 = 0; i1 < 16; i1++) { // generate: inner_loop

    __builtin_IMCE_STEP();

    var179 = __builtin_IMCE_GET_CREG((short)0);
    var180 = __builtin_IMCE_GET_CREG((short)1);
    var181 = __builtin_IMCE_GET_CREG((short)2);
    var182 = __builtin_IMCE_GET_CREG((short)3);


    // generate: add
    var187 = __builtin_IMCE_RECV(3);
    var188 = __builtin_IMCE_ADD(var179, var187, 15);
    var189 = __builtin_IMCE_RECV(3);
    var190 = __builtin_IMCE_ADD(var180, var189, 15);
    var191 = __builtin_IMCE_RECV(3);
    var192 = __builtin_IMCE_ADD(var181, var191, 15);
    var193 = __builtin_IMCE_RECV(3);
    var194 = __builtin_IMCE_ADD(var182, var193, 15);
    // endgenerate: add


    // generate: min_max_quantize
    __builtin_IMCE_MM_QUANT(var195, 0, 15, 4);
    var196 = __builtin_IMCE_GET_QREG(0);
    __builtin_IMCE_MM_QUANT(var197, 0, 15, 5);
    var198 = __builtin_IMCE_GET_QREG(1);
    __builtin_IMCE_MM_QUANT(var199, 0, 15, 6);
    var200 = __builtin_IMCE_GET_QREG(2);
    __builtin_IMCE_MM_QUANT(var201, 0, 15, 7);
    var202 = __builtin_IMCE_GET_QREG(3);
    // endgenerate: min_max_quantize


    // generate: concat
    var204 = __builtin_IMCE_RECV(2);
    var203 = __builtin_IMCE_OR(var196, var204, 15);
    var206 = __builtin_IMCE_RECV(2);
    var205 = __builtin_IMCE_OR(var198, var206, 15);
    var208 = __builtin_IMCE_RECV(2);
    var207 = __builtin_IMCE_OR(var200, var208, 15);
    var210 = __builtin_IMCE_RECV(2);
    var209 = __builtin_IMCE_OR(var202, var210, 15);
    // endgenerate: concat


    // generate: split
    // endgenerate: split


    __builtin_IMCE_SEND(0, var183, 0, 0);
    __builtin_IMCE_SEND(0, var184, 0, 0);
    __builtin_IMCE_SEND(0, var185, 0, 0);
    __builtin_IMCE_SEND(0, var186, 0, 0);


  } // endgenerate: inner_loop
  // endgenerate: conv exec
}
else if (hid == 3 && wid == 2) {
  // generate: weight write
  for (int i1 = 0; i1 < 8192; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(1);
  } // endgenerate: load_block
  // endgenerate: weight write
  // generate: conv exec
  for (int i1 = 0; i1 < 18; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
  } // endgenerate: load_block
  __builtin_IMCE_STEP();

  var187 = __builtin_IMCE_GET_CREG((short)0);
  var189 = __builtin_IMCE_GET_CREG((short)1);
  var191 = __builtin_IMCE_GET_CREG((short)2);
  var193 = __builtin_IMCE_GET_CREG((short)3);


  __builtin_IMCE_SEND(8, var187, 3, 0);
  __builtin_IMCE_SEND(8, var189, 3, 0);
  __builtin_IMCE_SEND(8, var191, 3, 0);
  __builtin_IMCE_SEND(8, var193, 3, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: inner_loop
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_STEP();

    var187 = __builtin_IMCE_GET_CREG((short)0);
    var189 = __builtin_IMCE_GET_CREG((short)1);
    var191 = __builtin_IMCE_GET_CREG((short)2);
    var193 = __builtin_IMCE_GET_CREG((short)3);


    __builtin_IMCE_SEND(8, var187, 3, 0);
    __builtin_IMCE_SEND(8, var189, 3, 0);
    __builtin_IMCE_SEND(8, var191, 3, 0);
    __builtin_IMCE_SEND(8, var193, 3, 0);


  } // endgenerate: inner_loop

  __builtin_IMCE_STEP();

  var187 = __builtin_IMCE_GET_CREG((short)0);
  var189 = __builtin_IMCE_GET_CREG((short)1);
  var191 = __builtin_IMCE_GET_CREG((short)2);
  var193 = __builtin_IMCE_GET_CREG((short)3);


  __builtin_IMCE_SEND(8, var187, 3, 0);
  __builtin_IMCE_SEND(8, var189, 3, 0);
  __builtin_IMCE_SEND(8, var191, 3, 0);
  __builtin_IMCE_SEND(8, var193, 3, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: outer_loop
    for (int i2 = 0; i2 < 2; i2++) { // generate: load_block
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
    } // endgenerate: load_block
    __builtin_IMCE_STEP();

    var187 = __builtin_IMCE_GET_CREG((short)0);
    var189 = __builtin_IMCE_GET_CREG((short)1);
    var191 = __builtin_IMCE_GET_CREG((short)2);
    var193 = __builtin_IMCE_GET_CREG((short)3);


    __builtin_IMCE_SEND(8, var187, 3, 0);
    __builtin_IMCE_SEND(8, var189, 3, 0);
    __builtin_IMCE_SEND(8, var191, 3, 0);
    __builtin_IMCE_SEND(8, var193, 3, 0);


    for (int i2 = 0; i2 < 14; i2++) { // generate: inner_loop
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_STEP();

      var187 = __builtin_IMCE_GET_CREG((short)0);
      var189 = __builtin_IMCE_GET_CREG((short)1);
      var191 = __builtin_IMCE_GET_CREG((short)2);
      var193 = __builtin_IMCE_GET_CREG((short)3);


      __builtin_IMCE_SEND(8, var187, 3, 0);
      __builtin_IMCE_SEND(8, var189, 3, 0);
      __builtin_IMCE_SEND(8, var191, 3, 0);
      __builtin_IMCE_SEND(8, var193, 3, 0);


    } // endgenerate: inner_loop

    __builtin_IMCE_STEP();

    var187 = __builtin_IMCE_GET_CREG((short)0);
    var189 = __builtin_IMCE_GET_CREG((short)1);
    var191 = __builtin_IMCE_GET_CREG((short)2);
    var193 = __builtin_IMCE_GET_CREG((short)3);


    __builtin_IMCE_SEND(8, var187, 3, 0);
    __builtin_IMCE_SEND(8, var189, 3, 0);
    __builtin_IMCE_SEND(8, var191, 3, 0);
    __builtin_IMCE_SEND(8, var193, 3, 0);


  } // endgenerate: outer_loop
  for (int i1 = 0; i1 < 16; i1++) { // generate: inner_loop

    __builtin_IMCE_STEP();

    var187 = __builtin_IMCE_GET_CREG((short)0);
    var189 = __builtin_IMCE_GET_CREG((short)1);
    var191 = __builtin_IMCE_GET_CREG((short)2);
    var193 = __builtin_IMCE_GET_CREG((short)3);


    __builtin_IMCE_SEND(8, var187, 3, 0);
    __builtin_IMCE_SEND(8, var189, 3, 0);
    __builtin_IMCE_SEND(8, var191, 3, 0);
    __builtin_IMCE_SEND(8, var193, 3, 0);


  } // endgenerate: inner_loop
  // endgenerate: conv exec
}
else if (hid == 3 && wid == 3) {
  // generate: weight write
  for (int i1 = 0; i1 < 8192; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(1);
  } // endgenerate: load_block
  // endgenerate: weight write
  // generate: min write
  var211 = __builtin_IMCE_RECV(1);
  // endgenerate: min write
  // generate: max write
  var212 = __builtin_IMCE_RECV(1);
  // endgenerate: max write
  // generate: conv exec
  for (int i1 = 0; i1 < 18; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
  } // endgenerate: load_block
  __builtin_IMCE_STEP();

  var213 = __builtin_IMCE_GET_CREG((short)0);
  var214 = __builtin_IMCE_GET_CREG((short)1);
  var215 = __builtin_IMCE_GET_CREG((short)2);
  var216 = __builtin_IMCE_GET_CREG((short)3);


  // generate: add
  var217 = __builtin_IMCE_RECV(2);
  var218 = __builtin_IMCE_ADD(var213, var217, 15);
  var219 = __builtin_IMCE_RECV(2);
  var220 = __builtin_IMCE_ADD(var214, var219, 15);
  var221 = __builtin_IMCE_RECV(2);
  var222 = __builtin_IMCE_ADD(var215, var221, 15);
  var223 = __builtin_IMCE_RECV(2);
  var224 = __builtin_IMCE_ADD(var216, var223, 15);
  // endgenerate: add


  // generate: min_max_quantize
  __builtin_IMCE_MM_QUANT(var225, 0, 15, 0);
  var204 = __builtin_IMCE_GET_QREG(0);
  __builtin_IMCE_MM_QUANT(var226, 0, 15, 1);
  var206 = __builtin_IMCE_GET_QREG(1);
  __builtin_IMCE_MM_QUANT(var227, 0, 15, 2);
  var208 = __builtin_IMCE_GET_QREG(2);
  __builtin_IMCE_MM_QUANT(var228, 0, 15, 3);
  var210 = __builtin_IMCE_GET_QREG(3);
  // endgenerate: min_max_quantize


  __builtin_IMCE_SEND(1, var204, 2, 0);
  __builtin_IMCE_SEND(1, var206, 2, 0);
  __builtin_IMCE_SEND(1, var208, 2, 0);
  __builtin_IMCE_SEND(1, var210, 2, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: inner_loop
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_STEP();

    var213 = __builtin_IMCE_GET_CREG((short)0);
    var214 = __builtin_IMCE_GET_CREG((short)1);
    var215 = __builtin_IMCE_GET_CREG((short)2);
    var216 = __builtin_IMCE_GET_CREG((short)3);


    // generate: add
    var217 = __builtin_IMCE_RECV(2);
    var218 = __builtin_IMCE_ADD(var213, var217, 15);
    var219 = __builtin_IMCE_RECV(2);
    var220 = __builtin_IMCE_ADD(var214, var219, 15);
    var221 = __builtin_IMCE_RECV(2);
    var222 = __builtin_IMCE_ADD(var215, var221, 15);
    var223 = __builtin_IMCE_RECV(2);
    var224 = __builtin_IMCE_ADD(var216, var223, 15);
    // endgenerate: add


    // generate: min_max_quantize
    __builtin_IMCE_MM_QUANT(var225, 0, 15, 0);
    var204 = __builtin_IMCE_GET_QREG(0);
    __builtin_IMCE_MM_QUANT(var226, 0, 15, 1);
    var206 = __builtin_IMCE_GET_QREG(1);
    __builtin_IMCE_MM_QUANT(var227, 0, 15, 2);
    var208 = __builtin_IMCE_GET_QREG(2);
    __builtin_IMCE_MM_QUANT(var228, 0, 15, 3);
    var210 = __builtin_IMCE_GET_QREG(3);
    // endgenerate: min_max_quantize


    __builtin_IMCE_SEND(1, var204, 2, 0);
    __builtin_IMCE_SEND(1, var206, 2, 0);
    __builtin_IMCE_SEND(1, var208, 2, 0);
    __builtin_IMCE_SEND(1, var210, 2, 0);


  } // endgenerate: inner_loop

  __builtin_IMCE_STEP();

  var213 = __builtin_IMCE_GET_CREG((short)0);
  var214 = __builtin_IMCE_GET_CREG((short)1);
  var215 = __builtin_IMCE_GET_CREG((short)2);
  var216 = __builtin_IMCE_GET_CREG((short)3);


  // generate: add
  var217 = __builtin_IMCE_RECV(2);
  var218 = __builtin_IMCE_ADD(var213, var217, 15);
  var219 = __builtin_IMCE_RECV(2);
  var220 = __builtin_IMCE_ADD(var214, var219, 15);
  var221 = __builtin_IMCE_RECV(2);
  var222 = __builtin_IMCE_ADD(var215, var221, 15);
  var223 = __builtin_IMCE_RECV(2);
  var224 = __builtin_IMCE_ADD(var216, var223, 15);
  // endgenerate: add


  // generate: min_max_quantize
  __builtin_IMCE_MM_QUANT(var225, 0, 15, 0);
  var204 = __builtin_IMCE_GET_QREG(0);
  __builtin_IMCE_MM_QUANT(var226, 0, 15, 1);
  var206 = __builtin_IMCE_GET_QREG(1);
  __builtin_IMCE_MM_QUANT(var227, 0, 15, 2);
  var208 = __builtin_IMCE_GET_QREG(2);
  __builtin_IMCE_MM_QUANT(var228, 0, 15, 3);
  var210 = __builtin_IMCE_GET_QREG(3);
  // endgenerate: min_max_quantize


  __builtin_IMCE_SEND(1, var204, 2, 0);
  __builtin_IMCE_SEND(1, var206, 2, 0);
  __builtin_IMCE_SEND(1, var208, 2, 0);
  __builtin_IMCE_SEND(1, var210, 2, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: outer_loop
    for (int i2 = 0; i2 < 2; i2++) { // generate: load_block
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
    } // endgenerate: load_block
    __builtin_IMCE_STEP();

    var213 = __builtin_IMCE_GET_CREG((short)0);
    var214 = __builtin_IMCE_GET_CREG((short)1);
    var215 = __builtin_IMCE_GET_CREG((short)2);
    var216 = __builtin_IMCE_GET_CREG((short)3);


    // generate: add
    var217 = __builtin_IMCE_RECV(2);
    var218 = __builtin_IMCE_ADD(var213, var217, 15);
    var219 = __builtin_IMCE_RECV(2);
    var220 = __builtin_IMCE_ADD(var214, var219, 15);
    var221 = __builtin_IMCE_RECV(2);
    var222 = __builtin_IMCE_ADD(var215, var221, 15);
    var223 = __builtin_IMCE_RECV(2);
    var224 = __builtin_IMCE_ADD(var216, var223, 15);
    // endgenerate: add


    // generate: min_max_quantize
    __builtin_IMCE_MM_QUANT(var225, 0, 15, 0);
    var204 = __builtin_IMCE_GET_QREG(0);
    __builtin_IMCE_MM_QUANT(var226, 0, 15, 1);
    var206 = __builtin_IMCE_GET_QREG(1);
    __builtin_IMCE_MM_QUANT(var227, 0, 15, 2);
    var208 = __builtin_IMCE_GET_QREG(2);
    __builtin_IMCE_MM_QUANT(var228, 0, 15, 3);
    var210 = __builtin_IMCE_GET_QREG(3);
    // endgenerate: min_max_quantize


    __builtin_IMCE_SEND(1, var204, 2, 0);
    __builtin_IMCE_SEND(1, var206, 2, 0);
    __builtin_IMCE_SEND(1, var208, 2, 0);
    __builtin_IMCE_SEND(1, var210, 2, 0);


    for (int i2 = 0; i2 < 14; i2++) { // generate: inner_loop
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_STEP();

      var213 = __builtin_IMCE_GET_CREG((short)0);
      var214 = __builtin_IMCE_GET_CREG((short)1);
      var215 = __builtin_IMCE_GET_CREG((short)2);
      var216 = __builtin_IMCE_GET_CREG((short)3);


      // generate: add
      var217 = __builtin_IMCE_RECV(2);
      var218 = __builtin_IMCE_ADD(var213, var217, 15);
      var219 = __builtin_IMCE_RECV(2);
      var220 = __builtin_IMCE_ADD(var214, var219, 15);
      var221 = __builtin_IMCE_RECV(2);
      var222 = __builtin_IMCE_ADD(var215, var221, 15);
      var223 = __builtin_IMCE_RECV(2);
      var224 = __builtin_IMCE_ADD(var216, var223, 15);
      // endgenerate: add


      // generate: min_max_quantize
      __builtin_IMCE_MM_QUANT(var225, 0, 15, 0);
      var204 = __builtin_IMCE_GET_QREG(0);
      __builtin_IMCE_MM_QUANT(var226, 0, 15, 1);
      var206 = __builtin_IMCE_GET_QREG(1);
      __builtin_IMCE_MM_QUANT(var227, 0, 15, 2);
      var208 = __builtin_IMCE_GET_QREG(2);
      __builtin_IMCE_MM_QUANT(var228, 0, 15, 3);
      var210 = __builtin_IMCE_GET_QREG(3);
      // endgenerate: min_max_quantize


      __builtin_IMCE_SEND(1, var204, 2, 0);
      __builtin_IMCE_SEND(1, var206, 2, 0);
      __builtin_IMCE_SEND(1, var208, 2, 0);
      __builtin_IMCE_SEND(1, var210, 2, 0);


    } // endgenerate: inner_loop

    __builtin_IMCE_STEP();

    var213 = __builtin_IMCE_GET_CREG((short)0);
    var214 = __builtin_IMCE_GET_CREG((short)1);
    var215 = __builtin_IMCE_GET_CREG((short)2);
    var216 = __builtin_IMCE_GET_CREG((short)3);


    // generate: add
    var217 = __builtin_IMCE_RECV(2);
    var218 = __builtin_IMCE_ADD(var213, var217, 15);
    var219 = __builtin_IMCE_RECV(2);
    var220 = __builtin_IMCE_ADD(var214, var219, 15);
    var221 = __builtin_IMCE_RECV(2);
    var222 = __builtin_IMCE_ADD(var215, var221, 15);
    var223 = __builtin_IMCE_RECV(2);
    var224 = __builtin_IMCE_ADD(var216, var223, 15);
    // endgenerate: add


    // generate: min_max_quantize
    __builtin_IMCE_MM_QUANT(var225, 0, 15, 0);
    var204 = __builtin_IMCE_GET_QREG(0);
    __builtin_IMCE_MM_QUANT(var226, 0, 15, 1);
    var206 = __builtin_IMCE_GET_QREG(1);
    __builtin_IMCE_MM_QUANT(var227, 0, 15, 2);
    var208 = __builtin_IMCE_GET_QREG(2);
    __builtin_IMCE_MM_QUANT(var228, 0, 15, 3);
    var210 = __builtin_IMCE_GET_QREG(3);
    // endgenerate: min_max_quantize


    __builtin_IMCE_SEND(1, var204, 2, 0);
    __builtin_IMCE_SEND(1, var206, 2, 0);
    __builtin_IMCE_SEND(1, var208, 2, 0);
    __builtin_IMCE_SEND(1, var210, 2, 0);


  } // endgenerate: outer_loop
  for (int i1 = 0; i1 < 16; i1++) { // generate: inner_loop

    __builtin_IMCE_STEP();

    var213 = __builtin_IMCE_GET_CREG((short)0);
    var214 = __builtin_IMCE_GET_CREG((short)1);
    var215 = __builtin_IMCE_GET_CREG((short)2);
    var216 = __builtin_IMCE_GET_CREG((short)3);


    // generate: add
    var217 = __builtin_IMCE_RECV(2);
    var218 = __builtin_IMCE_ADD(var213, var217, 15);
    var219 = __builtin_IMCE_RECV(2);
    var220 = __builtin_IMCE_ADD(var214, var219, 15);
    var221 = __builtin_IMCE_RECV(2);
    var222 = __builtin_IMCE_ADD(var215, var221, 15);
    var223 = __builtin_IMCE_RECV(2);
    var224 = __builtin_IMCE_ADD(var216, var223, 15);
    // endgenerate: add


    // generate: min_max_quantize
    __builtin_IMCE_MM_QUANT(var225, 0, 15, 0);
    var204 = __builtin_IMCE_GET_QREG(0);
    __builtin_IMCE_MM_QUANT(var226, 0, 15, 1);
    var206 = __builtin_IMCE_GET_QREG(1);
    __builtin_IMCE_MM_QUANT(var227, 0, 15, 2);
    var208 = __builtin_IMCE_GET_QREG(2);
    __builtin_IMCE_MM_QUANT(var228, 0, 15, 3);
    var210 = __builtin_IMCE_GET_QREG(3);
    // endgenerate: min_max_quantize


    __builtin_IMCE_SEND(1, var204, 2, 0);
    __builtin_IMCE_SEND(1, var206, 2, 0);
    __builtin_IMCE_SEND(1, var208, 2, 0);
    __builtin_IMCE_SEND(1, var210, 2, 0);


  } // endgenerate: inner_loop
  // endgenerate: conv exec
}
else if (hid == 3 && wid == 4) {
  // generate: weight write
  for (int i1 = 0; i1 < 8192; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(1);
  } // endgenerate: load_block
  // endgenerate: weight write
  // generate: conv exec
  for (int i1 = 0; i1 < 18; i1++) { // generate: load_block
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
  } // endgenerate: load_block
  __builtin_IMCE_STEP();

  var217 = __builtin_IMCE_GET_CREG((short)0);
  var219 = __builtin_IMCE_GET_CREG((short)1);
  var221 = __builtin_IMCE_GET_CREG((short)2);
  var223 = __builtin_IMCE_GET_CREG((short)3);


  __builtin_IMCE_SEND(1, var217, 2, 0);
  __builtin_IMCE_SEND(1, var219, 2, 0);
  __builtin_IMCE_SEND(1, var221, 2, 0);
  __builtin_IMCE_SEND(1, var223, 2, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: inner_loop
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_LOAD_LB(0);
    __builtin_IMCE_STEP();

    var217 = __builtin_IMCE_GET_CREG((short)0);
    var219 = __builtin_IMCE_GET_CREG((short)1);
    var221 = __builtin_IMCE_GET_CREG((short)2);
    var223 = __builtin_IMCE_GET_CREG((short)3);


    __builtin_IMCE_SEND(1, var217, 2, 0);
    __builtin_IMCE_SEND(1, var219, 2, 0);
    __builtin_IMCE_SEND(1, var221, 2, 0);
    __builtin_IMCE_SEND(1, var223, 2, 0);


  } // endgenerate: inner_loop

  __builtin_IMCE_STEP();

  var217 = __builtin_IMCE_GET_CREG((short)0);
  var219 = __builtin_IMCE_GET_CREG((short)1);
  var221 = __builtin_IMCE_GET_CREG((short)2);
  var223 = __builtin_IMCE_GET_CREG((short)3);


  __builtin_IMCE_SEND(1, var217, 2, 0);
  __builtin_IMCE_SEND(1, var219, 2, 0);
  __builtin_IMCE_SEND(1, var221, 2, 0);
  __builtin_IMCE_SEND(1, var223, 2, 0);


  for (int i1 = 0; i1 < 14; i1++) { // generate: outer_loop
    for (int i2 = 0; i2 < 2; i2++) { // generate: load_block
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
    } // endgenerate: load_block
    __builtin_IMCE_STEP();

    var217 = __builtin_IMCE_GET_CREG((short)0);
    var219 = __builtin_IMCE_GET_CREG((short)1);
    var221 = __builtin_IMCE_GET_CREG((short)2);
    var223 = __builtin_IMCE_GET_CREG((short)3);


    __builtin_IMCE_SEND(1, var217, 2, 0);
    __builtin_IMCE_SEND(1, var219, 2, 0);
    __builtin_IMCE_SEND(1, var221, 2, 0);
    __builtin_IMCE_SEND(1, var223, 2, 0);


    for (int i2 = 0; i2 < 14; i2++) { // generate: inner_loop
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_STEP();

      var217 = __builtin_IMCE_GET_CREG((short)0);
      var219 = __builtin_IMCE_GET_CREG((short)1);
      var221 = __builtin_IMCE_GET_CREG((short)2);
      var223 = __builtin_IMCE_GET_CREG((short)3);


      __builtin_IMCE_SEND(1, var217, 2, 0);
      __builtin_IMCE_SEND(1, var219, 2, 0);
      __builtin_IMCE_SEND(1, var221, 2, 0);
      __builtin_IMCE_SEND(1, var223, 2, 0);


    } // endgenerate: inner_loop

    __builtin_IMCE_STEP();

    var217 = __builtin_IMCE_GET_CREG((short)0);
    var219 = __builtin_IMCE_GET_CREG((short)1);
    var221 = __builtin_IMCE_GET_CREG((short)2);
    var223 = __builtin_IMCE_GET_CREG((short)3);


    __builtin_IMCE_SEND(1, var217, 2, 0);
    __builtin_IMCE_SEND(1, var219, 2, 0);
    __builtin_IMCE_SEND(1, var221, 2, 0);
    __builtin_IMCE_SEND(1, var223, 2, 0);


  } // endgenerate: outer_loop
  for (int i1 = 0; i1 < 16; i1++) { // generate: inner_loop

    __builtin_IMCE_STEP();

    var217 = __builtin_IMCE_GET_CREG((short)0);
    var219 = __builtin_IMCE_GET_CREG((short)1);
    var221 = __builtin_IMCE_GET_CREG((short)2);
    var223 = __builtin_IMCE_GET_CREG((short)3);


    __builtin_IMCE_SEND(1, var217, 2, 0);
    __builtin_IMCE_SEND(1, var219, 2, 0);
    __builtin_IMCE_SEND(1, var221, 2, 0);
    __builtin_IMCE_SEND(1, var223, 2, 0);


  } // endgenerate: inner_loop
  // endgenerate: conv exec
}
}
