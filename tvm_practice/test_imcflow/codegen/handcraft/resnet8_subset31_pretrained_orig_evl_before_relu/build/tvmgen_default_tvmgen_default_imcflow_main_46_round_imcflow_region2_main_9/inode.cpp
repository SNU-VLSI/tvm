#include "../common_decl.h"
void tvmgen_default_tvmgen_default_imcflow_main_46_round_imcflow_region2_main_9() {
  int hid = __builtin_INODE_GET_CORE_HID();
  int wid = 0;
  int var1; // policy_table_start_address
  int var2; // imem_start_address
  int var3; // imcu_start_address
  int var4; // send_data_base_address
  int* var5; // sm30_d53lhs_send_data_base_address
  int var6; // sm30_d53lhs_tile_loop_count
  int* var7; // sm31_d53rhs_send_data_base_address
  int var8; // sm31_d53rhs_tile_loop_count
  int* var9; // s67_63_d68func_out1_split1_recv_data_base_address
  int var10; // s67_63_d68func_out1_split1_tile_loop_count
  int var11; // recv_data_base_address
  int* var12; // s60_d68func_out0_split0_recv_data_base_address
  int var13; // s60_d68func_out0_split0_tile_loop_count
  if (hid == 0 && wid == 0) { // inode_0_0
    // generate: clear flag before policy update
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: clear flag before policy update
    // generate: policy update
    var1 = 8544;
    for (int i1 = 0; i1 < 15; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 0);
    } // endgenerate
    var1 = 9024;
    for (int i1 = 0; i1 < 16; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 9536;
    for (int i1 = 0; i1 < 11; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 2);
    } // endgenerate
    var1 = 9888;
    for (int i1 = 0; i1 < 8; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 3);
    } // endgenerate
    var1 = 10144;
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
    var2 = 10208;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 66; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 11);
    } // endgenerate
    // endgenerate: imem write: imce_0_1
    // generate: imem write: imce_0_2
    var2 = 12320;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 78; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 12);
    } // endgenerate
    // endgenerate: imem write: imce_0_2
    // generate: imem write: imce_0_3
    var2 = 14816;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 20; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 13);
    } // endgenerate
    // endgenerate: imem write: imce_0_3
    // generate: imem write: imce_0_4
    var2 = 15456;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 14);
    // endgenerate
    // endgenerate: imem write: imce_0_4
    // generate: imcu write
    var3 = 0;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 256; i1++) { // generate
      __builtin_INODE_WR_IMCU(var3 + i1*32, 0, 7);
    } // endgenerate
    // endgenerate: imcu write
    // generate: sync before compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(10, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: sync before compute enable
    // generate: imce_0_1 compute
    __builtin_INODE_IMCE_COMPUTE(0, 11);
    // endgenerate: imce_0_1 compute
    // generate: imce_0_2 compute
    __builtin_INODE_IMCE_COMPUTE(0, 12);
    // endgenerate: imce_0_2 compute
    // generate: imce_0_3 compute
    __builtin_INODE_IMCE_COMPUTE(0, 13);
    // endgenerate: imce_0_3 compute
    // generate: wait all imce compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(10, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: wait all imce compute enable
    // generate: send - TensorEdge(((67, -52), fused_scale), ((67, 61), fused_scale)), inode_0_0 -> imce_0_1
    var4 = 8256;
    for (int i1 = 0; i1 < 2; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 4, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge(((67, -52), fused_scale), ((67, 61), fused_scale)), inode_0_0 -> imce_0_1
    // generate: send - TensorEdge(((67, -53), fused_bias), ((67, 61), fused_bias)), inode_0_0 -> imce_0_1
    var4 = 8320;
    for (int i1 = 0; i1 < 2; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 5, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge(((67, -53), fused_bias), ((67, 61), fused_bias)), inode_0_0 -> imce_0_1
    // generate: send - TensorEdge(((67, -51), scale), ((67, 62), rhs)), inode_0_0 -> imce_0_1
    var4 = 8192;
    for (int i1 = 0; i1 < 2; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 3, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge(((67, -51), scale), ((67, 62), rhs)), inode_0_0 -> imce_0_1
    // generate: send - TensorEdge((-58, scale), ((67, 63), rhs)), inode_0_0 -> imce_0_1
    var4 = 8480;
    for (int i1 = 0; i1 < 2; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 6, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge((-58, scale), ((67, 63), rhs)), inode_0_0 -> imce_0_1
    // generate: send - TensorEdge((-57, config), (66, config)), inode_0_0 -> imce_0_2
    var4 = 8448;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 8, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-57, config), (66, config)), inode_0_0 -> imce_0_2
    // generate: send - TensorEdge((-54, min), (65, min)), inode_0_0 -> imce_0_3
    var4 = 8384;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 9, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-54, min), (65, min)), inode_0_0 -> imce_0_3
    // generate: send - TensorEdge((-55, max), (65, max)), inode_0_0 -> imce_0_3
    var4 = 8416;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 10, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-55, max), (65, max)), inode_0_0 -> imce_0_3
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
    // generate: send - TensorEdge((-30, odata), (53, lhs)), inode_0_0 -> imce_1_3
    var5 = (int*)(32768);
    var4 = 0;
    var6 = var5[0];
    __asm__ volatile("nop");
    for (int i1 = 0; i1 < var6; i1++) { // generate
      __builtin_INODE_STANDBY(8, 1);
      __builtin_INODE_SET_FLAG(1);
      __builtin_INODE_STANDBY(8, 0);
      __builtin_INODE_SET_FLAG(0);
      __builtin_INODE_SEND(var4 + i1*32, 0, 2, 2);

    } // endgenerate
    // endgenerate: send - TensorEdge((-30, odata), (53, lhs)), inode_0_0 -> imce_1_3
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
    var1 = 8288;
    for (int i1 = 0; i1 < 11; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 0);
    } // endgenerate
    var1 = 8640;
    for (int i1 = 0; i1 < 12; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 9024;
    for (int i1 = 0; i1 < 9; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 2);
    } // endgenerate
    var1 = 9312;
    for (int i1 = 0; i1 < 6; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 3);
    } // endgenerate
    var1 = 9504;
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
    var2 = 9568;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 37; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 7);
    } // endgenerate
    // endgenerate: imem write: imce_1_1
    // generate: imem write: imce_1_2
    var2 = 10752;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 20; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 8);
    } // endgenerate
    // endgenerate: imem write: imce_1_2
    // generate: imem write: imce_1_3
    var2 = 11392;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 21; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 9);
    } // endgenerate
    // endgenerate: imem write: imce_1_3
    // generate: imem write: imce_1_4
    var2 = 12064;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 10);
    // endgenerate
    // endgenerate: imem write: imce_1_4
    // generate: imcu write
    var3 = 0;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 256; i1++) { // generate
      __builtin_INODE_WR_IMCU(var3 + i1*32, 0, 2);
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
    __builtin_INODE_IMCE_COMPUTE(0, 7);
    // endgenerate: imce_1_1 compute
    // generate: imce_1_2 compute
    __builtin_INODE_IMCE_COMPUTE(0, 8);
    // endgenerate: imce_1_2 compute
    // generate: imce_1_3 compute
    __builtin_INODE_IMCE_COMPUTE(0, 9);
    // endgenerate: imce_1_3 compute
    // generate: wait all imce compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(10, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: wait all imce compute enable
    // generate: send - TensorEdge((-44, config), (55, config)), inode_1_0 -> imce_1_1
    var4 = 8256;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 3, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-44, config), (55, config)), inode_1_0 -> imce_1_1
    // generate: send - TensorEdge((-41, min), (54, min)), inode_1_0 -> imce_1_2
    var4 = 8192;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 4, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-41, min), (54, min)), inode_1_0 -> imce_1_2
    // generate: send - TensorEdge((-42, max), (54, max)), inode_1_0 -> imce_1_2
    var4 = 8224;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 5, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-42, max), (54, max)), inode_1_0 -> imce_1_2
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
    // generate: send - TensorEdge((-31, odata), (53, rhs)), inode_1_0 -> imce_1_3
    var7 = (int*)(32768);
    var4 = 0;
    var8 = var7[0];
    __asm__ volatile("nop");
    for (int i1 = 0; i1 < var8; i1++) { // generate
      __builtin_INODE_STANDBY(8, 1);
      __builtin_INODE_SET_FLAG(1);
      __builtin_INODE_STANDBY(8, 0);
      __builtin_INODE_SET_FLAG(0);
      __builtin_INODE_SEND(var4 + i1*32, 0, 6, 3);

    } // endgenerate
    // endgenerate: send - TensorEdge((-31, odata), (53, rhs)), inode_1_0 -> imce_1_3
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
    var1 = 8416;
    for (int i1 = 0; i1 < 12; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 0);
    } // endgenerate
    var1 = 8800;
    for (int i1 = 0; i1 < 13; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 9216;
    for (int i1 = 0; i1 < 8; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 2);
    } // endgenerate
    var1 = 9472;
    __builtin_INODE_PU(var1, 0, 0, 3);
    __builtin_INODE_PU(var1, 32, 1, 3);
    __builtin_INODE_PU(var1, 64, 2, 3);
    var1 = 9568;
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
    var2 = 9632;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 50; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 8);
    } // endgenerate
    // endgenerate: imem write: imce_2_1
    // generate: imem write: imce_2_2
    var2 = 11232;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 95; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 9);
    } // endgenerate
    // endgenerate: imem write: imce_2_2
    // generate: imem write: imce_2_3
    var2 = 14272;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 10);
    // endgenerate
    // endgenerate: imem write: imce_2_3
    // generate: imem write: imce_2_4
    var2 = 14304;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 11);
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
    __builtin_INODE_IMCE_COMPUTE(0, 8);
    // endgenerate: imce_2_1 compute
    // generate: imce_2_2 compute
    __builtin_INODE_IMCE_COMPUTE(0, 9);
    // endgenerate: imce_2_2 compute
    // generate: wait all imce compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: wait all imce compute enable
    // generate: send - TensorEdge(((56, -37), fused_scale), ((56, 50), fused_scale)), inode_2_0 -> imce_2_1
    var4 = 8192;
    for (int i1 = 0; i1 < 2; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 4, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge(((56, -37), fused_scale), ((56, 50), fused_scale)), inode_2_0 -> imce_2_1
    // generate: send - TensorEdge(((56, -38), fused_bias), ((56, 50), fused_bias)), inode_2_0 -> imce_2_1
    var4 = 8256;
    for (int i1 = 0; i1 < 2; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 5, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge(((56, -38), fused_bias), ((56, 50), fused_bias)), inode_2_0 -> imce_2_1
    // generate: send - TensorEdge(((56, -39), min), ((56, 51), min)), inode_2_0 -> imce_2_1
    var4 = 8320;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 2, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((56, -39), min), ((56, 51), min)), inode_2_0 -> imce_2_1
    // generate: send - TensorEdge(((56, -40), max), ((56, 51), max)), inode_2_0 -> imce_2_1
    var4 = 8352;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 3, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((56, -40), max), ((56, 51), max)), inode_2_0 -> imce_2_1
    // generate: send - TensorEdge((-46, config), (58, config)), inode_2_0 -> imce_2_2
    var4 = 8384;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 7, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-46, config), (58, config)), inode_2_0 -> imce_2_2
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
    // generate: recv: TensorID(68, func_out1)
    var9 = (int*)(32768);
    var11 = 0;
    var10 = var9[0];
    __asm__ volatile("nop");
    for (int i1 = 0; i1 < var10; i1++) { // generate
      __builtin_INODE_RECV(var11 + i1*32, 0, 0, 2);

    } // endgenerate
    // endgenerate: recv: TensorID(68, func_out1)
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
    var1 = 8352;
    for (int i1 = 0; i1 < 10; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 0);
    } // endgenerate
    var1 = 8672;
    for (int i1 = 0; i1 < 12; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 9056;
    for (int i1 = 0; i1 < 9; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 2);
    } // endgenerate
    var1 = 9344;
    __builtin_INODE_PU(var1, 0, 0, 3);
    __builtin_INODE_PU(var1, 32, 1, 3);
    __builtin_INODE_PU(var1, 64, 2, 3);
    var1 = 9440;
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
    var2 = 9504;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 38; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 6);
    } // endgenerate
    // endgenerate: imem write: imce_3_1
    // generate: imem write: imce_3_2
    var2 = 10720;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 145; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 7);
    } // endgenerate
    // endgenerate: imem write: imce_3_2
    // generate: imem write: imce_3_3
    var2 = 15360;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 8);
    // endgenerate
    // endgenerate: imem write: imce_3_3
    // generate: imem write: imce_3_4
    var2 = 15392;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 9);
    // endgenerate
    // endgenerate: imem write: imce_3_4
    // generate: imcu write
    var3 = 0;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 256; i1++) { // generate
      __builtin_INODE_WR_IMCU(var3 + i1*32, 0, 4);
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
    __builtin_INODE_IMCE_COMPUTE(0, 6);
    // endgenerate: imce_3_1 compute
    // generate: imce_3_2 compute
    __builtin_INODE_IMCE_COMPUTE(0, 7);
    // endgenerate: imce_3_2 compute
    // generate: wait all imce compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(10, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: wait all imce compute enable
    // generate: send - TensorEdge((-47, fused_scale), (60, fused_scale)), inode_3_0 -> imce_3_1
    var4 = 8224;
    for (int i1 = 0; i1 < 2; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 2, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge((-47, fused_scale), (60, fused_scale)), inode_3_0 -> imce_3_1
    // generate: send - TensorEdge((-48, fused_bias), (60, fused_bias)), inode_3_0 -> imce_3_1
    var4 = 8288;
    for (int i1 = 0; i1 < 2; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 3, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge((-48, fused_bias), (60, fused_bias)), inode_3_0 -> imce_3_1
    // generate: send - TensorEdge(((59, -35), config), ((59, 47), config)), inode_3_0 -> imce_3_2
    var4 = 8192;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 5, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((59, -35), config), ((59, 47), config)), inode_3_0 -> imce_3_2
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
    // generate: recv: TensorID(68, func_out0)
    var12 = (int*)(32768);
    var11 = 0;
    var13 = var12[0];
    __asm__ volatile("nop");
    for (int i1 = 0; i1 < var13; i1++) { // generate
      __builtin_INODE_RECV(var11 + i1*32, 0, 0, 2);

    } // endgenerate
    // endgenerate: recv: TensorID(68, func_out0)
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
