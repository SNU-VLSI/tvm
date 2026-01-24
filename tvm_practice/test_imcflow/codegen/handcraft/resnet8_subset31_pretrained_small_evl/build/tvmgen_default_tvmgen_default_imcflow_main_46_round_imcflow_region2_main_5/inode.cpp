#include "../common_decl.h"
void tvmgen_default_tvmgen_default_imcflow_main_46_round_imcflow_region2_main_5() {
  int hid = __builtin_INODE_GET_CORE_HID();
  int wid = 0;
  int var1; // policy_table_start_address
  int var2; // imem_start_address
  int var3; // imcu_start_address
  int var4; // send_data_base_address
  int* var5; // sm31_d56lhs_send_data_base_address
  int var6; // sm31_d56lhs_tile_loop_count
  int* var7; // sm32_d56rhs_send_data_base_address
  int var8; // sm32_d56rhs_tile_loop_count
  int* var9; // s62_51_d70func_out0_split0_recv_data_base_address
  int var10; // s62_51_d70func_out0_split0_tile_loop_count
  int var11; // recv_data_base_address
  int* var12; // s69_66_d70func_out1_split1_recv_data_base_address
  int var13; // s69_66_d70func_out1_split1_tile_loop_count
  if (hid == 0 && wid == 0) { // inode_0_0
    // generate: clear flag before policy update
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: clear flag before policy update
    // generate: policy update
    var1 = 0;
    for (int i1 = 0; i1 < 6; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 0);
    } // endgenerate
    var1 = 192;
    for (int i1 = 0; i1 < 7; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 416;
    for (int i1 = 0; i1 < 6; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 2);
    } // endgenerate
    var1 = 608;
    for (int i1 = 0; i1 < 6; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 3);
    } // endgenerate
    var1 = 800;
    __builtin_INODE_PU(var1, 0, 0, 4);
    __builtin_INODE_PU(var1, 32, 1, 4);
    __builtin_INODE_PU(var1, 64, 2, 4);
    __builtin_INODE_PU(var1, 96, 3, 4);
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
    var2 = 928;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 2);
    // endgenerate
    // endgenerate: imem write: imce_0_1
    // generate: imem write: imce_0_2
    var2 = 960;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 3);
    // endgenerate
    // endgenerate: imem write: imce_0_2
    // generate: imem write: imce_0_3
    var2 = 992;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 4);
    // endgenerate
    // endgenerate: imem write: imce_0_3
    // generate: imem write: imce_0_4
    var2 = 1024;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 5);
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
    // generate: send - TensorEdge((-31, odata), (56, lhs)), inode_0_0 -> imce_3_4
    var5 = (int*)(2048);
    var4 = 0;
    var6 = var5[0];
    __asm__ volatile("nop");
    for (int i1 = 0; i1 < var6; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 1, 2);

    } // endgenerate
    // endgenerate: send - TensorEdge((-31, odata), (56, lhs)), inode_0_0 -> imce_3_4
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
    var1 = 0;
    for (int i1 = 0; i1 < 12; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 0);
    } // endgenerate
    var1 = 384;
    for (int i1 = 0; i1 < 13; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 800;
    for (int i1 = 0; i1 < 12; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 2);
    } // endgenerate
    var1 = 1184;
    for (int i1 = 0; i1 < 12; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 3);
    } // endgenerate
    var1 = 1568;
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
    var2 = 1760;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 2);
    // endgenerate
    // endgenerate: imem write: imce_1_1
    // generate: imem write: imce_1_2
    var2 = 1792;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 3);
    // endgenerate
    // endgenerate: imem write: imce_1_2
    // generate: imem write: imce_1_3
    var2 = 1824;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 4);
    // endgenerate
    // endgenerate: imem write: imce_1_3
    // generate: imem write: imce_1_4
    var2 = 1856;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 5);
    // endgenerate
    // endgenerate: imem write: imce_1_4

    // generate: sync before compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(10, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: sync before compute enable
    // generate: wait all imce compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(10, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: wait all imce compute enable
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
    // generate: send - TensorEdge((-32, odata), (56, rhs)), inode_1_0 -> imce_3_4
    var7 = (int*)(2048);
    var4 = 0;
    var8 = var7[0];
    __asm__ volatile("nop");
    for (int i1 = 0; i1 < var8; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 1, 3);

    } // endgenerate
    // endgenerate: send - TensorEdge((-32, odata), (56, rhs)), inode_1_0 -> imce_3_4
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
    var1 = 25120;
    for (int i1 = 0; i1 < 22; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 0);
    } // endgenerate
    var1 = 25824;
    for (int i1 = 0; i1 < 22; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 26528;
    for (int i1 = 0; i1 < 16; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 2);
    } // endgenerate
    var1 = 27040;
    for (int i1 = 0; i1 < 13; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 3);
    } // endgenerate
    var1 = 27456;
    for (int i1 = 0; i1 < 7; i1++) { // generate
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
    var2 = 27680;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 133; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 4);
    } // endgenerate
    // endgenerate: imem write: imce_2_1
    // generate: imem write: imce_2_2
    var2 = 31936;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 14; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 5);
    } // endgenerate
    // endgenerate: imem write: imce_2_2
    // generate: imem write: imce_2_3
    var2 = 32384;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 188; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 6);
    } // endgenerate
    // endgenerate: imem write: imce_2_3
    // generate: imem write: imce_2_4
    var2 = 38400;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 70; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 7);
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
      __builtin_INODE_WR_IMCU(var3 + i1*32, 0, 10);
    } // endgenerate
    var3 = 16384;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 256; i1++) { // generate
      __builtin_INODE_WR_IMCU(var3 + i1*32, 0, 19);
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
    __builtin_INODE_IMCE_COMPUTE(0, 4);
    // endgenerate: imce_2_1 compute
    // generate: imce_2_2 compute
    __builtin_INODE_IMCE_COMPUTE(0, 5);
    // endgenerate: imce_2_2 compute
    // generate: imce_2_3 compute
    __builtin_INODE_IMCE_COMPUTE(0, 6);
    // endgenerate: imce_2_3 compute
    // generate: imce_2_4 compute
    __builtin_INODE_IMCE_COMPUTE(0, 7);
    // endgenerate: imce_2_4 compute
    // generate: wait all imce compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: wait all imce compute enable
    // generate: send - TensorEdge(((69, -53), config), ((69, 63), config)), inode_2_0 -> imce_2_1
    var4 = 24832;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 20, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((69, -53), config), ((69, 63), config)), inode_2_0 -> imce_2_1
    // generate: send - TensorEdge(((69, -54), fused_scale), ((69, 64), fused_scale)), inode_2_0 -> imce_2_1
    var4 = 24864;
    for (int i1 = 0; i1 < 2; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 11, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge(((69, -54), fused_scale), ((69, 64), fused_scale)), inode_2_0 -> imce_2_1
    // generate: send - TensorEdge(((69, -55), fused_bias), ((69, 64), fused_bias)), inode_2_0 -> imce_2_1
    var4 = 24928;
    for (int i1 = 0; i1 < 2; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 18, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge(((69, -55), fused_bias), ((69, 64), fused_bias)), inode_2_0 -> imce_2_1
    // generate: send - TensorEdge(((69, -51), odata), ((69, 65), lhs)), inode_2_0 -> imce_2_1
    var4 = 24768;
    for (int i1 = 0; i1 < 2; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 3, 3);

    } // endgenerate
    // endgenerate: send - TensorEdge(((69, -51), odata), ((69, 65), lhs)), inode_2_0 -> imce_2_1
    // generate: send - TensorEdge(((69, -56), odata), ((69, 66), rhs)), inode_2_0 -> imce_2_1
    var4 = 24992;
    for (int i1 = 0; i1 < 2; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 2, 2);

    } // endgenerate
    // endgenerate: send - TensorEdge(((69, -56), odata), ((69, 66), rhs)), inode_2_0 -> imce_2_1
    // generate: send - TensorEdge((-57, min), (68, min)), inode_2_0 -> imce_2_2
    var4 = 25056;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 12, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-57, min), (68, min)), inode_2_0 -> imce_2_2
    // generate: send - TensorEdge((-58, max), (68, max)), inode_2_0 -> imce_2_2
    var4 = 25088;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 21, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-58, max), (68, max)), inode_2_0 -> imce_2_2
    // generate: send - TensorEdge(((62, -36), config), ((62, 49), config)), inode_2_0 -> imce_2_3
    var4 = 24576;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 15, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((62, -36), config), ((62, 49), config)), inode_2_0 -> imce_2_3
    // generate: send - TensorEdge(((62, -37), fused_scale), ((62, 51), fused_scale)), inode_2_0 -> imce_2_3
    var4 = 24608;
    for (int i1 = 0; i1 < 2; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 8, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge(((62, -37), fused_scale), ((62, 51), fused_scale)), inode_2_0 -> imce_2_3
    // generate: send - TensorEdge(((62, -38), fused_bias), ((62, 51), fused_bias)), inode_2_0 -> imce_2_3
    var4 = 24672;
    for (int i1 = 0; i1 < 2; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 13, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge(((62, -38), fused_bias), ((62, 51), fused_bias)), inode_2_0 -> imce_2_3
    // generate: send - TensorEdge((-49, config), (61, config)), inode_2_0 -> imce_2_4
    var4 = 24736;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 17, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-49, config), (61, config)), inode_2_0 -> imce_2_4
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
    // generate: recv: TensorID(70, func_out1)
    var12 = (int*)(2048);
    var11 = 0;
    var13 = var12[0];
    __asm__ volatile("nop");
    for (int i1 = 0; i1 < var13; i1++) { // generate
      __builtin_INODE_RECV(var11 + i1*32, 0, 0, 2);

    } // endgenerate
    // endgenerate: recv: TensorID(70, func_out1)
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
    var1 = 8480;
    for (int i1 = 0; i1 < 15; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 0);
    } // endgenerate
    var1 = 8960;
    for (int i1 = 0; i1 < 17; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 9504;
    for (int i1 = 0; i1 < 14; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 2);
    } // endgenerate
    var1 = 9952;
    for (int i1 = 0; i1 < 8; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 3);
    } // endgenerate
    var1 = 10208;
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
    var2 = 10368;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 18; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 3);
    } // endgenerate
    // endgenerate: imem write: imce_3_1
    // generate: imem write: imce_3_2
    var2 = 10944;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 54; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 4);
    } // endgenerate
    // endgenerate: imem write: imce_3_2
    // generate: imem write: imce_3_3
    var2 = 12672;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 14; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 5);
    } // endgenerate
    // endgenerate: imem write: imce_3_3
    // generate: imem write: imce_3_4
    var2 = 13120;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 9; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 6);
    } // endgenerate
    // endgenerate: imem write: imce_3_4
    // generate: imcu write
    var3 = 0;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 256; i1++) { // generate
      __builtin_INODE_WR_IMCU(var3 + i1*32, 0, 12);
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
    __builtin_INODE_IMCE_COMPUTE(0, 3);
    // endgenerate: imce_3_1 compute
    // generate: imce_3_2 compute
    __builtin_INODE_IMCE_COMPUTE(0, 4);
    // endgenerate: imce_3_2 compute
    // generate: imce_3_3 compute
    __builtin_INODE_IMCE_COMPUTE(0, 5);
    // endgenerate: imce_3_3 compute
    // generate: imce_3_4 compute
    __builtin_INODE_IMCE_COMPUTE(0, 6);
    // endgenerate: imce_3_4 compute
    // generate: wait all imce compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(10, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: wait all imce compute enable
    // generate: send - TensorEdge((-46, min), (59, min)), inode_3_0 -> imce_3_1
    var4 = 8416;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 7, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-46, min), (59, min)), inode_3_0 -> imce_3_1
    // generate: send - TensorEdge((-47, max), (59, max)), inode_3_0 -> imce_3_1
    var4 = 8448;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 10, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-47, max), (59, max)), inode_3_0 -> imce_3_1
    // generate: send - TensorEdge(((58, -41), config), ((58, 53), config)), inode_3_0 -> imce_3_2
    var4 = 8192;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 13, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((58, -41), config), ((58, 53), config)), inode_3_0 -> imce_3_2
    // generate: send - TensorEdge(((58, -42), fused_scale), ((58, 54), fused_scale)), inode_3_0 -> imce_3_2
    var4 = 8224;
    for (int i1 = 0; i1 < 2; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 8, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge(((58, -42), fused_scale), ((58, 54), fused_scale)), inode_3_0 -> imce_3_2
    // generate: send - TensorEdge(((58, -43), fused_bias), ((58, 54), fused_bias)), inode_3_0 -> imce_3_2
    var4 = 8288;
    for (int i1 = 0; i1 < 2; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 11, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge(((58, -43), fused_bias), ((58, 54), fused_bias)), inode_3_0 -> imce_3_2
    // generate: send - TensorEdge((-44, min), (57, min)), inode_3_0 -> imce_3_3
    var4 = 8352;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 9, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-44, min), (57, min)), inode_3_0 -> imce_3_3
    // generate: send - TensorEdge((-45, max), (57, max)), inode_3_0 -> imce_3_3
    var4 = 8384;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 14, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-45, max), (57, max)), inode_3_0 -> imce_3_3
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
    // generate: recv: TensorID(70, func_out0)
    var9 = (int*)(2048);
    var11 = 0;
    var10 = var9[0];
    __asm__ volatile("nop");
    for (int i1 = 0; i1 < var10; i1++) { // generate
      __builtin_INODE_RECV(var11 + i1*32, 0, 0, 2);

    } // endgenerate
    // endgenerate: recv: TensorID(70, func_out0)
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
