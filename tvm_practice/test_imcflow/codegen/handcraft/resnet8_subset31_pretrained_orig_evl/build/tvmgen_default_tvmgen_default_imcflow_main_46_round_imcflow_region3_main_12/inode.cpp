#include "../common_decl.h"
void tvmgen_default_tvmgen_default_imcflow_main_46_round_imcflow_region3_main_12() {
  int hid = __builtin_INODE_GET_CORE_HID();
  int wid = 0;
  int var1; // policy_table_start_address
  int var2; // imem_start_address
  int var3; // imcu_start_address
  int var4; // send_data_base_address
  int* var5; // sm61_d85lhs_send_data_base_address
  int var6; // sm61_d85lhs_tile_loop_count
  int* var7; // sm62_d85rhs_send_data_base_address
  int var8; // sm62_d85rhs_tile_loop_count
  int* var9; // s104_101_d105func_out1_split1_recv_data_base_address
  int var10; // s104_101_d105func_out1_split1_tile_loop_count
  int var11; // recv_data_base_address
  int* var12; // s97_79_d105func_out0_split0_recv_data_base_address
  int var13; // s97_79_d105func_out0_split0_tile_loop_count
  if (hid == 0 && wid == 0) { // inode_0_0
    // generate: clear flag before policy update
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: clear flag before policy update
    // generate: policy update
    var1 = 0;
    for (int i1 = 0; i1 < 13; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 0);
    } // endgenerate
    var1 = 416;
    for (int i1 = 0; i1 < 13; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 832;
    for (int i1 = 0; i1 < 8; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 2);
    } // endgenerate
    var1 = 1088;
    __builtin_INODE_PU(var1, 0, 0, 3);
    __builtin_INODE_PU(var1, 32, 1, 3);
    __builtin_INODE_PU(var1, 64, 2, 3);
    __builtin_INODE_PU(var1, 96, 3, 3);
    var1 = 1216;
    __builtin_INODE_PU(var1, 0, 0, 4);
    __builtin_INODE_PU(var1, 32, 1, 4);
    // endgenerate: policy update
    // generate: sync all inodes
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(10, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: sync all inodes
    // generate: halt for slave inodes
    __builtin_INODE_HALT();
    // endgenerate: halt for slave inodes
    // generate: imem write: imce_0_1
    var2 = 1280;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 3);
    // endgenerate
    // endgenerate: imem write: imce_0_1
    // generate: imem write: imce_0_2
    var2 = 1312;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 4);
    // endgenerate
    // endgenerate: imem write: imce_0_2
    // generate: imem write: imce_0_3
    var2 = 1344;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 5);
    // endgenerate
    // endgenerate: imem write: imce_0_3
    // generate: imem write: imce_0_4
    var2 = 1376;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 6);
    // endgenerate
    // endgenerate: imem write: imce_0_4

    // generate: sync before compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(10, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: sync before compute enable
    // generate: wait all imce compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(10, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: wait all imce compute enable
    // generate: sync all inodes
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(10, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: sync all inodes
    // generate: halt for slave inodes
    __builtin_INODE_HALT();
    // endgenerate: halt for slave inodes
    // generate: send - TensorEdge((-61, odata), (85, lhs)), inode_0_0 -> imce_3_4
    var5 = (int*)(16384);
    var4 = 0;
    var6 = var5[0];
    __asm__ volatile("nop");
    for (int i1 = 0; i1 < var6; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 1, 2);

    } // endgenerate
    // endgenerate: send - TensorEdge((-61, odata), (85, lhs)), inode_0_0 -> imce_3_4
    // generate: sync all inodes
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(10, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: sync all inodes
    // generate: halt for slave inodes
    __builtin_INODE_HALT();
    // endgenerate: halt for slave inodes
  }
  else if (hid == 1 && wid == 0) { // inode_1_0
    // generate: clear flag before policy update
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: clear flag before policy update
    // generate: policy update
    var1 = 8800;
    for (int i1 = 0; i1 < 20; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 0);
    } // endgenerate
    var1 = 9440;
    for (int i1 = 0; i1 < 18; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 10016;
    for (int i1 = 0; i1 < 16; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 2);
    } // endgenerate
    var1 = 10528;
    for (int i1 = 0; i1 < 14; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 3);
    } // endgenerate
    var1 = 10976;
    for (int i1 = 0; i1 < 6; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 4);
    } // endgenerate
    // endgenerate: policy update
    // generate: sync all inodes
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(10, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: sync all inodes
    // generate: halt for slave inodes
    __builtin_INODE_HALT();
    // endgenerate: halt for slave inodes
    // generate: imem write: imce_1_1
    var2 = 11168;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 4);
    // endgenerate
    // endgenerate: imem write: imce_1_1
    // generate: imem write: imce_1_2
    var2 = 11200;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 5);
    // endgenerate
    // endgenerate: imem write: imce_1_2
    // generate: imem write: imce_1_3
    var2 = 11232;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 141; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 6);
    } // endgenerate
    // endgenerate: imem write: imce_1_3
    // generate: imem write: imce_1_4
    var2 = 15744;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 16; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 7);
    } // endgenerate
    // endgenerate: imem write: imce_1_4
    // generate: imcu write
    var3 = 0;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 256; i1++) { // generate
      __builtin_INODE_WR_IMCU(var3 + i1*32, 0, 17);
    } // endgenerate
    // endgenerate: imcu write
    // generate: sync before compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(10, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: sync before compute enable
    // generate: imce_1_3 compute
    __builtin_INODE_IMCE_COMPUTE(0, 6);
    // endgenerate: imce_1_3 compute
    // generate: imce_1_4 compute
    __builtin_INODE_IMCE_COMPUTE(0, 7);
    // endgenerate: imce_1_4 compute
    // generate: wait all imce compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(10, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: wait all imce compute enable
    // generate: send - TensorEdge(((104, -90), config), ((104, 98), config)), inode_1_0 -> imce_1_3
    var4 = 8320;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 18, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((104, -90), config), ((104, 98), config)), inode_1_0 -> imce_1_3
    // generate: send - TensorEdge(((104, -91), fused_scale), ((104, 99), fused_scale)), inode_1_0 -> imce_1_3
    var4 = 8352;
    for (int i1 = 0; i1 < 4; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 10, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge(((104, -91), fused_scale), ((104, 99), fused_scale)), inode_1_0 -> imce_1_3
    // generate: send - TensorEdge(((104, -92), fused_bias), ((104, 99), fused_bias)), inode_1_0 -> imce_1_3
    var4 = 8480;
    for (int i1 = 0; i1 < 4; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 16, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge(((104, -92), fused_bias), ((104, 99), fused_bias)), inode_1_0 -> imce_1_3
    // generate: send - TensorEdge(((104, -88), odata), ((104, 100), lhs)), inode_1_0 -> imce_1_3
    var4 = 8192;
    for (int i1 = 0; i1 < 4; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 3, 3);

    } // endgenerate
    // endgenerate: send - TensorEdge(((104, -88), odata), ((104, 100), lhs)), inode_1_0 -> imce_1_3
    // generate: send - TensorEdge(((104, -93), odata), ((104, 101), rhs)), inode_1_0 -> imce_1_3
    var4 = 8608;
    for (int i1 = 0; i1 < 4; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 2, 2);

    } // endgenerate
    // endgenerate: send - TensorEdge(((104, -93), odata), ((104, 101), rhs)), inode_1_0 -> imce_1_3
    // generate: send - TensorEdge((-94, min), (103, min)), inode_1_0 -> imce_1_4
    var4 = 8736;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 11, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-94, min), (103, min)), inode_1_0 -> imce_1_4
    // generate: send - TensorEdge((-95, max), (103, max)), inode_1_0 -> imce_1_4
    var4 = 8768;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 19, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-95, max), (103, max)), inode_1_0 -> imce_1_4
    // generate: sync all inodes
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(10, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: sync all inodes
    // generate: halt for slave inodes
    __builtin_INODE_HALT();
    // endgenerate: halt for slave inodes
    // generate: send - TensorEdge((-62, odata), (85, rhs)), inode_1_0 -> imce_3_4
    var7 = (int*)(16384);
    var4 = 0;
    var8 = var7[0];
    __asm__ volatile("nop");
    for (int i1 = 0; i1 < var8; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 1, 3);

    } // endgenerate
    // endgenerate: send - TensorEdge((-62, odata), (85, rhs)), inode_1_0 -> imce_3_4
    // generate: sync all inodes
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(10, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: sync all inodes
    // generate: halt for slave inodes
    __builtin_INODE_HALT();
    // endgenerate: halt for slave inodes
  }
  else if (hid == 2 && wid == 0) { // inode_2_0
    // generate: clear flag before policy update
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: clear flag before policy update
    // generate: policy update
    var1 = 24992;
    for (int i1 = 0; i1 < 23; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 0);
    } // endgenerate
    var1 = 25728;
    for (int i1 = 0; i1 < 24; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 26496;
    for (int i1 = 0; i1 < 18; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 2);
    } // endgenerate
    var1 = 27072;
    for (int i1 = 0; i1 < 12; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 3);
    } // endgenerate
    var1 = 27456;
    for (int i1 = 0; i1 < 8; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 4);
    } // endgenerate
    // endgenerate: policy update
    // generate: sync all inodes
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: sync all inodes
    // generate: halt for slave inodes
    __builtin_INODE_HALT();
    // endgenerate: halt for slave inodes
    // generate: imem write: imce_2_1
    var2 = 27712;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 192; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 3);
    } // endgenerate
    // endgenerate: imem write: imce_2_1
    // generate: imem write: imce_2_2
    var2 = 33856;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 128; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 4);
    } // endgenerate
    // endgenerate: imem write: imce_2_2
    // generate: imem write: imce_2_3
    var2 = 37952;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 70; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 5);
    } // endgenerate
    // endgenerate: imem write: imce_2_3
    // generate: imem write: imce_2_4
    var2 = 40192;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 20; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 6);
    } // endgenerate
    // endgenerate: imem write: imce_2_4
    // generate: imcu write
    var3 = 0;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 256; i1++) { // generate
      __builtin_INODE_WR_IMCU(var3 + i1*32, 0, 14);
    } // endgenerate
    var3 = 8192;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 256; i1++) { // generate
      __builtin_INODE_WR_IMCU(var3 + i1*32, 0, 11);
    } // endgenerate
    var3 = 16384;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 256; i1++) { // generate
      __builtin_INODE_WR_IMCU(var3 + i1*32, 0, 12);
    } // endgenerate
    // endgenerate: imcu write
    // generate: sync before compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: sync before compute enable
    // generate: imce_2_1 compute
    __builtin_INODE_IMCE_COMPUTE(0, 3);
    // endgenerate: imce_2_1 compute
    // generate: imce_2_2 compute
    __builtin_INODE_IMCE_COMPUTE(0, 4);
    // endgenerate: imce_2_2 compute
    // generate: imce_2_3 compute
    __builtin_INODE_IMCE_COMPUTE(0, 5);
    // endgenerate: imce_2_3 compute
    // generate: imce_2_4 compute
    __builtin_INODE_IMCE_COMPUTE(0, 6);
    // endgenerate: imce_2_4 compute
    // generate: wait all imce compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: wait all imce compute enable
    // generate: send - TensorEdge(((97, -66), config), ((97, 77), config)), inode_2_0 -> imce_2_1
    var4 = 24576;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 15, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((97, -66), config), ((97, 77), config)), inode_2_0 -> imce_2_1
    // generate: send - TensorEdge(((97, -67), fused_scale), ((97, 79), fused_scale)), inode_2_0 -> imce_2_1
    var4 = 24608;
    for (int i1 = 0; i1 < 4; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 7, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge(((97, -67), fused_scale), ((97, 79), fused_scale)), inode_2_0 -> imce_2_1
    // generate: send - TensorEdge(((97, -68), fused_bias), ((97, 79), fused_bias)), inode_2_0 -> imce_2_1
    var4 = 24736;
    for (int i1 = 0; i1 < 4; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 13, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge(((97, -68), fused_bias), ((97, 79), fused_bias)), inode_2_0 -> imce_2_1
    // generate: send - TensorEdge(((96, -84), config), ((96, 92), config)), inode_2_0 -> imce_2_2
    var4 = 24928;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 21, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((96, -84), config), ((96, 92), config)), inode_2_0 -> imce_2_2
    // generate: send - TensorEdge((-86, config), (95, config)), inode_2_0 -> imce_2_3
    var4 = 24960;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 22, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-86, config), (95, config)), inode_2_0 -> imce_2_3
    // generate: send - TensorEdge((-79, min), (90, min)), inode_2_0 -> imce_2_4
    var4 = 24864;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 8, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-79, min), (90, min)), inode_2_0 -> imce_2_4
    // generate: send - TensorEdge((-80, max), (90, max)), inode_2_0 -> imce_2_4
    var4 = 24896;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 16, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-80, max), (90, max)), inode_2_0 -> imce_2_4
    // generate: sync all inodes
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: sync all inodes
    // generate: halt for slave inodes
    __builtin_INODE_HALT();
    // endgenerate: halt for slave inodes
    // generate: recv: TensorID(105, func_out1)
    var9 = (int*)(8192);
    var11 = 0;
    var10 = var9[0];
    __asm__ volatile("nop");
    for (int i1 = 0; i1 < var10; i1++) { // generate
      __builtin_INODE_RECV(var11 + i1*32, 0, 0, 2);

    } // endgenerate
    // endgenerate: recv: TensorID(105, func_out1)
    // generate: sync all inodes
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: sync all inodes
    // generate: halt for slave inodes
    __builtin_INODE_HALT();
    // endgenerate: halt for slave inodes
  }
  else if (hid == 3 && wid == 0) { // inode_3_0
    // generate: clear flag before policy update
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: clear flag before policy update
    // generate: policy update
    var1 = 16768;
    for (int i1 = 0; i1 < 16; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 0);
    } // endgenerate
    var1 = 17280;
    for (int i1 = 0; i1 < 17; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 17824;
    for (int i1 = 0; i1 < 13; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 2);
    } // endgenerate
    var1 = 18240;
    for (int i1 = 0; i1 < 9; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 3);
    } // endgenerate
    var1 = 18528;
    __builtin_INODE_PU(var1, 0, 0, 4);
    __builtin_INODE_PU(var1, 32, 1, 4);
    __builtin_INODE_PU(var1, 64, 2, 4);
    __builtin_INODE_PU(var1, 96, 3, 4);
    __builtin_INODE_PU(var1, 128, 4, 4);
    // endgenerate: policy update
    // generate: sync all inodes
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(10, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: sync all inodes
    // generate: done and intrt for master inode
    __builtin_INODE_DONE();
    __builtin_INODE_INTRT(0);
    // endgenerate: done and intrt for master inode
    // generate: halt for master inode after done and intrt
    __builtin_INODE_HALT();
    // endgenerate: halt for master inode after done and intrt
    // generate: imem write: imce_3_1
    var2 = 18688;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 74; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 2);
    } // endgenerate
    // endgenerate: imem write: imce_3_1
    // generate: imem write: imce_3_2
    var2 = 21056;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 32; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 3);
    } // endgenerate
    // endgenerate: imem write: imce_3_2
    // generate: imem write: imce_3_3
    var2 = 22080;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 16; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 4);
    } // endgenerate
    // endgenerate: imem write: imce_3_3
    // generate: imem write: imce_3_4
    var2 = 22592;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 9; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 5);
    } // endgenerate
    // endgenerate: imem write: imce_3_4
    // generate: imcu write
    var3 = 0;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 256; i1++) { // generate
      __builtin_INODE_WR_IMCU(var3 + i1*32, 0, 12);
    } // endgenerate
    var3 = 8192;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 256; i1++) { // generate
      __builtin_INODE_WR_IMCU(var3 + i1*32, 0, 9);
    } // endgenerate
    // endgenerate: imcu write
    // generate: sync before compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(10, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: sync before compute enable
    // generate: imce_3_1 compute
    __builtin_INODE_IMCE_COMPUTE(0, 2);
    // endgenerate: imce_3_1 compute
    // generate: imce_3_2 compute
    __builtin_INODE_IMCE_COMPUTE(0, 3);
    // endgenerate: imce_3_2 compute
    // generate: imce_3_3 compute
    __builtin_INODE_IMCE_COMPUTE(0, 4);
    // endgenerate: imce_3_3 compute
    // generate: imce_3_4 compute
    __builtin_INODE_IMCE_COMPUTE(0, 5);
    // endgenerate: imce_3_4 compute
    // generate: wait all imce compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(10, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: wait all imce compute enable
    // generate: send - TensorEdge(((89, -72), config), ((89, 81), config)), inode_3_0 -> imce_3_1
    var4 = 16384;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 13, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((89, -72), config), ((89, 81), config)), inode_3_0 -> imce_3_1
    // generate: send - TensorEdge(((89, -73), fused_scale), ((89, 83), fused_scale)), inode_3_0 -> imce_3_1
    var4 = 16416;
    for (int i1 = 0; i1 < 4; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 7, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge(((89, -73), fused_scale), ((89, 83), fused_scale)), inode_3_0 -> imce_3_1
    // generate: send - TensorEdge(((89, -74), fused_bias), ((89, 83), fused_bias)), inode_3_0 -> imce_3_1
    var4 = 16544;
    for (int i1 = 0; i1 < 4; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 11, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge(((89, -74), fused_bias), ((89, 83), fused_bias)), inode_3_0 -> imce_3_1
    // generate: send - TensorEdge((-78, config), (88, config)), inode_3_0 -> imce_3_2
    var4 = 16736;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 15, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-78, config), (88, config)), inode_3_0 -> imce_3_2
    // generate: send - TensorEdge((-75, min), (86, min)), inode_3_0 -> imce_3_3
    var4 = 16672;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 8, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-75, min), (86, min)), inode_3_0 -> imce_3_3
    // generate: send - TensorEdge((-76, max), (86, max)), inode_3_0 -> imce_3_3
    var4 = 16704;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 14, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-76, max), (86, max)), inode_3_0 -> imce_3_3
    // generate: sync all inodes
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(10, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: sync all inodes
    // generate: done and intrt for master inode
    __builtin_INODE_DONE();
    __builtin_INODE_INTRT(0);
    // endgenerate: done and intrt for master inode
    // generate: halt for master inode after done and intrt
    __builtin_INODE_HALT();
    // endgenerate: halt for master inode after done and intrt
    // generate: recv: TensorID(105, func_out0)
    var12 = (int*)(8192);
    var11 = 0;
    var13 = var12[0];
    __asm__ volatile("nop");
    for (int i1 = 0; i1 < var13; i1++) { // generate
      __builtin_INODE_RECV(var11 + i1*32, 0, 0, 2);

    } // endgenerate
    // endgenerate: recv: TensorID(105, func_out0)
    // generate: sync all inodes
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(10, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: sync all inodes
    // generate: done and intrt for master inode
    __builtin_INODE_DONE();
    __builtin_INODE_INTRT(0);
    // endgenerate: done and intrt for master inode
    // generate: halt for master inode after done and intrt
    __builtin_INODE_HALT();
    // endgenerate: halt for master inode after done and intrt
  }
}
