#include "../common_decl.h"
void tvmgen_default_tvmgen_default_imcflow_main_46_round_imcflow_region3_main_20() {
  int hid = __builtin_INODE_GET_CORE_HID();
  int wid = 0;
  int var1; // policy_table_start_address
  int var2; // imem_start_address
  int var3; // imcu_start_address
  int var4; // send_data_base_address
  int* var5; // sm61_d84lhs_send_data_base_address
  int var6; // sm61_d84lhs_tile_loop_count
  int* var7; // sm62_d84rhs_send_data_base_address
  int var8; // sm62_d84rhs_tile_loop_count
  int* var9; // s104_100_d105func_out1_split1_recv_data_base_address
  int var10; // s104_100_d105func_out1_split1_tile_loop_count
  int var11; // recv_data_base_address
  int* var12; // s97_d105func_out0_split0_recv_data_base_address
  int var13; // s97_d105func_out0_split0_tile_loop_count
  if (hid == 0 && wid == 0) { // inode_0_0
    // generate: clear flag before policy update
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: clear flag before policy update
    // generate: policy update
    var1 = 128;
    for (int i1 = 0; i1 < 10; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 0);
    } // endgenerate
    var1 = 448;
    for (int i1 = 0; i1 < 12; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 832;
    for (int i1 = 0; i1 < 10; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 2);
    } // endgenerate
    var1 = 1152;
    for (int i1 = 0; i1 < 7; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 3);
    } // endgenerate
    var1 = 1376;
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
    var2 = 1440;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 22; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 6);
    } // endgenerate
    // endgenerate: imem write: imce_0_1
    // generate: imem write: imce_0_2
    var2 = 2144;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 29; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 7);
    } // endgenerate
    // endgenerate: imem write: imce_0_2
    // generate: imem write: imce_0_3
    var2 = 3072;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 29; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 8);
    } // endgenerate
    // endgenerate: imem write: imce_0_3
    // generate: imem write: imce_0_4
    var2 = 4000;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 9);
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
    // generate: imce_0_1 compute
    __builtin_INODE_IMCE_COMPUTE(0, 6);
    // endgenerate: imce_0_1 compute
    // generate: imce_0_2 compute
    __builtin_INODE_IMCE_COMPUTE(0, 7);
    // endgenerate: imce_0_2 compute
    // generate: imce_0_3 compute
    __builtin_INODE_IMCE_COMPUTE(0, 8);
    // endgenerate: imce_0_3 compute
    // generate: wait all imce compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(10, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: wait all imce compute enable
    // generate: send - TensorEdge((-93, min), (102, min)), inode_0_0 -> imce_0_1
    var4 = 64;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 4, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-93, min), (102, min)), inode_0_0 -> imce_0_1
    // generate: send - TensorEdge((-94, max), (102, max)), inode_0_0 -> imce_0_1
    var4 = 96;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 5, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-94, max), (102, max)), inode_0_0 -> imce_0_1
    // generate: send - TensorEdge((-76, min), (85, min)), inode_0_0 -> imce_0_3
    var4 = 0;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 1, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-76, min), (85, min)), inode_0_0 -> imce_0_3
    // generate: send - TensorEdge((-77, max), (85, max)), inode_0_0 -> imce_0_3
    var4 = 32;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 2, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-77, max), (85, max)), inode_0_0 -> imce_0_3
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
    // generate: send - TensorEdge((-61, odata), (84, lhs)), inode_0_0 -> imce_0_2
    var5 = (int*)(16384);
    var4 = 0;
    var6 = var5[0];
    __asm__ volatile("nop");
    for (int i1 = 0; i1 < var6; i1++) { // generate
      __builtin_INODE_STANDBY(2, 1);
      __builtin_INODE_SET_FLAG(1);
      __builtin_INODE_STANDBY(2, 0);
      __builtin_INODE_SET_FLAG(0);
      __builtin_INODE_SEND(var4 + i1*32, 0, 3, 2);

    } // endgenerate
    // endgenerate: send - TensorEdge((-61, odata), (84, lhs)), inode_0_0 -> imce_0_2
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
    var1 = 24672;
    for (int i1 = 0; i1 < 12; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 0);
    } // endgenerate
    var1 = 25056;
    for (int i1 = 0; i1 < 14; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 25504;
    for (int i1 = 0; i1 < 12; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 2);
    } // endgenerate
    var1 = 25888;
    for (int i1 = 0; i1 < 7; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 3);
    } // endgenerate
    var1 = 26112;
    __builtin_INODE_PU(var1, 0, 0, 4);
    __builtin_INODE_PU(var1, 32, 1, 4);
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
    var2 = 26176;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 78; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 8);
    } // endgenerate
    // endgenerate: imem write: imce_1_1
    // generate: imem write: imce_1_2
    var2 = 28672;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 60; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 9);
    } // endgenerate
    // endgenerate: imem write: imce_1_2
    // generate: imem write: imce_1_3
    var2 = 30592;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 42; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 10);
    } // endgenerate
    // endgenerate: imem write: imce_1_3
    // generate: imem write: imce_1_4
    var2 = 31936;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 11);
    // endgenerate
    // endgenerate: imem write: imce_1_4
    // generate: imcu write
    var3 = 0;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 256; i1++) { // generate
      __builtin_INODE_WR_IMCU(var3 + i1*32, 0, 1);
    } // endgenerate
    var3 = 8192;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 256; i1++) { // generate
      __builtin_INODE_WR_IMCU(var3 + i1*32, 0, 4);
    } // endgenerate
    var3 = 16384;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 256; i1++) { // generate
      __builtin_INODE_WR_IMCU(var3 + i1*32, 0, 6);
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
    // generate: imce_1_1 compute
    __builtin_INODE_IMCE_COMPUTE(0, 8);
    // endgenerate: imce_1_1 compute
    // generate: imce_1_2 compute
    __builtin_INODE_IMCE_COMPUTE(0, 9);
    // endgenerate: imce_1_2 compute
    // generate: imce_1_3 compute
    __builtin_INODE_IMCE_COMPUTE(0, 10);
    // endgenerate: imce_1_3 compute
    // generate: wait all imce compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(10, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: wait all imce compute enable
    // generate: send - TensorEdge((-96, config), (103, config)), inode_1_0 -> imce_1_1
    var4 = 24640;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 7, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-96, config), (103, config)), inode_1_0 -> imce_1_1
    // generate: send - TensorEdge(((88, -75), config), ((88, 81), config)), inode_1_0 -> imce_1_2
    var4 = 24576;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 2, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((88, -75), config), ((88, 81), config)), inode_1_0 -> imce_1_2
    // generate: send - TensorEdge((-79, config), (87, config)), inode_1_0 -> imce_1_3
    var4 = 24608;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 5, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-79, config), (87, config)), inode_1_0 -> imce_1_3
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
    // generate: send - TensorEdge((-62, odata), (84, rhs)), inode_1_0 -> imce_0_2
    var7 = (int*)(16384);
    var4 = 0;
    var8 = var7[0];
    __asm__ volatile("nop");
    for (int i1 = 0; i1 < var8; i1++) { // generate
      __builtin_INODE_STANDBY(2, 1);
      __builtin_INODE_SET_FLAG(1);
      __builtin_INODE_STANDBY(2, 0);
      __builtin_INODE_SET_FLAG(0);
      __builtin_INODE_SEND(var4 + i1*32, 0, 3, 3);

    } // endgenerate
    // endgenerate: send - TensorEdge((-62, odata), (84, rhs)), inode_1_0 -> imce_0_2
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
    var1 = 9056;
    for (int i1 = 0; i1 < 16; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 0);
    } // endgenerate
    var1 = 9568;
    for (int i1 = 0; i1 < 17; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 10112;
    for (int i1 = 0; i1 < 12; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 2);
    } // endgenerate
    var1 = 10496;
    for (int i1 = 0; i1 < 7; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 3);
    } // endgenerate
    var1 = 10720;
    __builtin_INODE_PU(var1, 0, 0, 4);
    __builtin_INODE_PU(var1, 32, 1, 4);
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
    var2 = 10784;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 122; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 12);
    } // endgenerate
    // endgenerate: imem write: imce_2_1
    // generate: imem write: imce_2_2
    var2 = 14688;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 78; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 13);
    } // endgenerate
    // endgenerate: imem write: imce_2_2
    // generate: imem write: imce_2_3
    var2 = 17184;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 95; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 14);
    } // endgenerate
    // endgenerate: imem write: imce_2_3
    // generate: imem write: imce_2_4
    var2 = 20224;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 15);
    // endgenerate
    // endgenerate: imem write: imce_2_4
    // generate: imcu write
    var3 = 0;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 256; i1++) { // generate
      __builtin_INODE_WR_IMCU(var3 + i1*32, 0, 6);
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
    __builtin_INODE_IMCE_COMPUTE(0, 12);
    // endgenerate: imce_2_1 compute
    // generate: imce_2_2 compute
    __builtin_INODE_IMCE_COMPUTE(0, 13);
    // endgenerate: imce_2_2 compute
    // generate: imce_2_3 compute
    __builtin_INODE_IMCE_COMPUTE(0, 14);
    // endgenerate: imce_2_3 compute
    // generate: wait all imce compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: wait all imce compute enable
    // generate: send - TensorEdge(((104, -91), fused_scale), ((104, 98), fused_scale)), inode_2_0 -> imce_2_1
    var4 = 8672;
    for (int i1 = 0; i1 < 4; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 9, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge(((104, -91), fused_scale), ((104, 98), fused_scale)), inode_2_0 -> imce_2_1
    // generate: send - TensorEdge(((104, -92), fused_bias), ((104, 98), fused_bias)), inode_2_0 -> imce_2_1
    var4 = 8800;
    for (int i1 = 0; i1 < 4; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 10, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge(((104, -92), fused_bias), ((104, 98), fused_bias)), inode_2_0 -> imce_2_1
    // generate: send - TensorEdge(((104, -90), scale), ((104, 99), rhs)), inode_2_0 -> imce_2_1
    var4 = 8544;
    for (int i1 = 0; i1 < 4; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 8, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge(((104, -90), scale), ((104, 99), rhs)), inode_2_0 -> imce_2_1
    // generate: send - TensorEdge((-97, scale), ((104, 100), rhs)), inode_2_0 -> imce_2_1
    var4 = 8928;
    for (int i1 = 0; i1 < 4; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 11, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge((-97, scale), ((104, 100), rhs)), inode_2_0 -> imce_2_1
    // generate: send - TensorEdge(((89, -68), fused_scale), ((89, 78), fused_scale)), inode_2_0 -> imce_2_2
    var4 = 8192;
    for (int i1 = 0; i1 < 4; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 4, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge(((89, -68), fused_scale), ((89, 78), fused_scale)), inode_2_0 -> imce_2_2
    // generate: send - TensorEdge(((89, -69), fused_bias), ((89, 78), fused_bias)), inode_2_0 -> imce_2_2
    var4 = 8320;
    for (int i1 = 0; i1 < 4; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 5, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge(((89, -69), fused_bias), ((89, 78), fused_bias)), inode_2_0 -> imce_2_2
    // generate: send - TensorEdge(((89, -70), min), ((89, 79), min)), inode_2_0 -> imce_2_2
    var4 = 8448;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 2, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((89, -70), min), ((89, 79), min)), inode_2_0 -> imce_2_2
    // generate: send - TensorEdge(((89, -71), max), ((89, 79), max)), inode_2_0 -> imce_2_2
    var4 = 8480;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 3, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((89, -71), max), ((89, 79), max)), inode_2_0 -> imce_2_2
    // generate: send - TensorEdge((-85, config), (94, config)), inode_2_0 -> imce_2_3
    var4 = 8512;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 7, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-85, config), (94, config)), inode_2_0 -> imce_2_3
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
    var1 = 16704;
    for (int i1 = 0; i1 < 12; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 0);
    } // endgenerate
    var1 = 17088;
    for (int i1 = 0; i1 < 13; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 17504;
    for (int i1 = 0; i1 < 11; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 2);
    } // endgenerate
    var1 = 17856;
    for (int i1 = 0; i1 < 8; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 3);
    } // endgenerate
    var1 = 18112;
    __builtin_INODE_PU(var1, 0, 0, 4);
    __builtin_INODE_PU(var1, 32, 1, 4);
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
    var2 = 18176;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 70; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 8);
    } // endgenerate
    // endgenerate: imem write: imce_3_1
    // generate: imem write: imce_3_2
    var2 = 20416;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 145; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 9);
    } // endgenerate
    // endgenerate: imem write: imce_3_2
    // generate: imem write: imce_3_3
    var2 = 25056;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 152; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 10);
    } // endgenerate
    // endgenerate: imem write: imce_3_3
    // generate: imem write: imce_3_4
    var2 = 29920;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 11);
    // endgenerate
    // endgenerate: imem write: imce_3_4
    // generate: imcu write
    var3 = 0;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 256; i1++) { // generate
      __builtin_INODE_WR_IMCU(var3 + i1*32, 0, 4);
    } // endgenerate
    var3 = 8192;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 256; i1++) { // generate
      __builtin_INODE_WR_IMCU(var3 + i1*32, 0, 6);
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
    __builtin_INODE_IMCE_COMPUTE(0, 8);
    // endgenerate: imce_3_1 compute
    // generate: imce_3_2 compute
    __builtin_INODE_IMCE_COMPUTE(0, 9);
    // endgenerate: imce_3_2 compute
    // generate: imce_3_3 compute
    __builtin_INODE_IMCE_COMPUTE(0, 10);
    // endgenerate: imce_3_3 compute
    // generate: wait all imce compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(10, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: wait all imce compute enable
    // generate: send - TensorEdge((-86, fused_scale), (97, fused_scale)), inode_3_0 -> imce_3_1
    var4 = 16448;
    for (int i1 = 0; i1 < 4; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 2, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge((-86, fused_scale), (97, fused_scale)), inode_3_0 -> imce_3_1
    // generate: send - TensorEdge((-87, fused_bias), (97, fused_bias)), inode_3_0 -> imce_3_1
    var4 = 16576;
    for (int i1 = 0; i1 < 4; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 3, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge((-87, fused_bias), (97, fused_bias)), inode_3_0 -> imce_3_1
    // generate: send - TensorEdge(((96, -66), config), ((96, 75), config)), inode_3_0 -> imce_3_2
    var4 = 16384;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 5, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((96, -66), config), ((96, 75), config)), inode_3_0 -> imce_3_2
    // generate: send - TensorEdge(((95, -83), config), ((95, 91), config)), inode_3_0 -> imce_3_3
    var4 = 16416;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 7, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((95, -83), config), ((95, 91), config)), inode_3_0 -> imce_3_3
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
