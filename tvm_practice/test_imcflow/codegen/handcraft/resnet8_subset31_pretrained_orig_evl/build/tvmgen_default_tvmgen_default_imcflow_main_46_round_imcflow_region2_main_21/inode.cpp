#include "../common_decl.h"
void tvmgen_default_tvmgen_default_imcflow_main_46_round_imcflow_region2_main_21() {
  int hid = __builtin_INODE_GET_CORE_HID();
  int wid = 0;
  int var1; // policy_table_start_address
  int var2; // imem_start_address
  int var3; // imcu_start_address
  int var4; // send_data_base_address
  int* var5; // sm30_d55_53lhs_send_data_base_address
  int var6; // sm30_d55_53lhs_tile_loop_count
  int* var7; // sm31_d55_53rhs_send_data_base_address
  int var8; // sm31_d55_53rhs_tile_loop_count
  int* var9; // s69_63_d70func_out1_split1_recv_data_base_address
  int var10; // s69_63_d70func_out1_split1_tile_loop_count
  int var11; // recv_data_base_address
  int* var12; // s62_d70func_out0_split0_recv_data_base_address
  int var13; // s62_d70func_out0_split0_tile_loop_count
  if (hid == 0 && wid == 0) { // inode_0_0
    // generate: clear flag before policy update
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: clear flag before policy update
    // generate: policy update
    var1 = 8352;
    for (int i1 = 0; i1 < 12; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 0);
    } // endgenerate
    var1 = 8736;
    for (int i1 = 0; i1 < 14; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 9184;
    for (int i1 = 0; i1 < 11; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 2);
    } // endgenerate
    var1 = 9536;
    for (int i1 = 0; i1 < 7; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 3);
    } // endgenerate
    var1 = 9760;
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
    var2 = 9824;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 41; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 8);
    } // endgenerate
    // endgenerate: imem write: imce_0_1
    // generate: imem write: imce_0_2
    var2 = 11136;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 20; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 9);
    } // endgenerate
    // endgenerate: imem write: imce_0_2
    // generate: imem write: imce_0_3
    var2 = 11776;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 20; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 10);
    } // endgenerate
    // endgenerate: imem write: imce_0_3
    // generate: imem write: imce_0_4
    var2 = 12416;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 11);
    // endgenerate
    // endgenerate: imem write: imce_0_4
    // generate: imcu write
    var3 = 0;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 256; i1++) { // generate
      __builtin_INODE_WR_IMCU(var3 + i1*32, 0, 1);
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
    __builtin_INODE_IMCE_COMPUTE(0, 8);
    // endgenerate: imce_0_1 compute
    // generate: imce_0_2 compute
    __builtin_INODE_IMCE_COMPUTE(0, 9);
    // endgenerate: imce_0_2 compute
    // generate: imce_0_3 compute
    __builtin_INODE_IMCE_COMPUTE(0, 10);
    // endgenerate: imce_0_3 compute
    // generate: wait all imce compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(10, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: wait all imce compute enable
    // generate: send - TensorEdge((-46, config), (57, config)), inode_0_0 -> imce_0_1
    var4 = 8256;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 2, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-46, config), (57, config)), inode_0_0 -> imce_0_1
    // generate: send - TensorEdge((-43, min), (56, min)), inode_0_0 -> imce_0_2
    var4 = 8192;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 3, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-43, min), (56, min)), inode_0_0 -> imce_0_2
    // generate: send - TensorEdge((-44, max), (56, max)), inode_0_0 -> imce_0_2
    var4 = 8224;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 4, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-44, max), (56, max)), inode_0_0 -> imce_0_2
    // generate: send - TensorEdge((-54, min), (65, min)), inode_0_0 -> imce_0_3
    var4 = 8288;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 6, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-54, min), (65, min)), inode_0_0 -> imce_0_3
    // generate: send - TensorEdge((-55, max), (65, max)), inode_0_0 -> imce_0_3
    var4 = 8320;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 7, 1);

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
    // generate: send - TensorEdge((-30, odata), ((55, 53), lhs)), inode_0_0 -> imce_1_2
    var5 = (int*)(32768);
    var4 = 0;
    var6 = var5[0];
    __asm__ volatile("nop");
    for (int i1 = 0; i1 < var6; i1++) { // generate
      __builtin_INODE_STANDBY(7, 1);
      __builtin_INODE_SET_FLAG(1);
      __builtin_INODE_STANDBY(7, 0);
      __builtin_INODE_SET_FLAG(0);
      __builtin_INODE_SEND(var4 + i1*32, 0, 5, 2);

    } // endgenerate
    // endgenerate: send - TensorEdge((-30, odata), ((55, 53), lhs)), inode_0_0 -> imce_1_2
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
    var1 = 8416;
    for (int i1 = 0; i1 < 12; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 0);
    } // endgenerate
    var1 = 8800;
    for (int i1 = 0; i1 < 14; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 9248;
    for (int i1 = 0; i1 < 9; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 2);
    } // endgenerate
    var1 = 9536;
    for (int i1 = 0; i1 < 8; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 3);
    } // endgenerate
    var1 = 9792;
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
    var2 = 9856;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 52; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 8);
    } // endgenerate
    // endgenerate: imem write: imce_1_1
    // generate: imem write: imce_1_2
    var2 = 11520;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 21; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 9);
    } // endgenerate
    // endgenerate: imem write: imce_1_2
    // generate: imem write: imce_1_3
    var2 = 12192;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 72; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 10);
    } // endgenerate
    // endgenerate: imem write: imce_1_3
    // generate: imem write: imce_1_4
    var2 = 14496;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 11);
    // endgenerate
    // endgenerate: imem write: imce_1_4
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
    // generate: send - TensorEdge(((58, -37), fused_scale), ((58, 50), fused_scale)), inode_1_0 -> imce_1_1
    var4 = 8192;
    for (int i1 = 0; i1 < 2; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 3, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge(((58, -37), fused_scale), ((58, 50), fused_scale)), inode_1_0 -> imce_1_1
    // generate: send - TensorEdge(((58, -38), fused_bias), ((58, 50), fused_bias)), inode_1_0 -> imce_1_1
    var4 = 8256;
    for (int i1 = 0; i1 < 2; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 4, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge(((58, -38), fused_bias), ((58, 50), fused_bias)), inode_1_0 -> imce_1_1
    // generate: send - TensorEdge(((58, -39), min), ((58, 51), min)), inode_1_0 -> imce_1_1
    var4 = 8320;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 1, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((58, -39), min), ((58, 51), min)), inode_1_0 -> imce_1_1
    // generate: send - TensorEdge(((58, -40), max), ((58, 51), max)), inode_1_0 -> imce_1_1
    var4 = 8352;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 2, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((58, -40), max), ((58, 51), max)), inode_1_0 -> imce_1_1
    // generate: send - TensorEdge((-57, config), (66, config)), inode_1_0 -> imce_1_3
    var4 = 8384;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 7, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-57, config), (66, config)), inode_1_0 -> imce_1_3
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
    // generate: send - TensorEdge((-31, odata), ((55, 53), rhs)), inode_1_0 -> imce_1_2
    var7 = (int*)(32768);
    var4 = 0;
    var8 = var7[0];
    __asm__ volatile("nop");
    for (int i1 = 0; i1 < var8; i1++) { // generate
      __builtin_INODE_STANDBY(7, 1);
      __builtin_INODE_SET_FLAG(1);
      __builtin_INODE_STANDBY(7, 0);
      __builtin_INODE_SET_FLAG(0);
      __builtin_INODE_SEND(var4 + i1*32, 0, 5, 3);

    } // endgenerate
    // endgenerate: send - TensorEdge((-31, odata), ((55, 53), rhs)), inode_1_0 -> imce_1_2
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
    var1 = 16576;
    for (int i1 = 0; i1 < 12; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 0);
    } // endgenerate
    var1 = 16960;
    for (int i1 = 0; i1 < 15; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 17440;
    for (int i1 = 0; i1 < 10; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 2);
    } // endgenerate
    var1 = 17760;
    for (int i1 = 0; i1 < 7; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 3);
    } // endgenerate
    var1 = 17984;
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
    var2 = 18048;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 149; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 8);
    } // endgenerate
    // endgenerate: imem write: imce_2_1
    // generate: imem write: imce_2_2
    var2 = 22816;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 70; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 9);
    } // endgenerate
    // endgenerate: imem write: imce_2_2
    // generate: imem write: imce_2_3
    var2 = 25056;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 45; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 10);
    } // endgenerate
    // endgenerate: imem write: imce_2_3
    // generate: imem write: imce_2_4
    var2 = 26496;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 11);
    // endgenerate
    // endgenerate: imem write: imce_2_4
    // generate: imcu write
    var3 = 0;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 256; i1++) { // generate
      __builtin_INODE_WR_IMCU(var3 + i1*32, 0, 2);
    } // endgenerate
    var3 = 8192;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 256; i1++) { // generate
      __builtin_INODE_WR_IMCU(var3 + i1*32, 0, 4);
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
    // generate: imce_2_3 compute
    __builtin_INODE_IMCE_COMPUTE(0, 10);
    // endgenerate: imce_2_3 compute
    // generate: wait all imce compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: wait all imce compute enable
    // generate: send - TensorEdge(((61, -35), config), ((61, 47), config)), inode_2_0 -> imce_2_1
    var4 = 16384;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 3, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((61, -35), config), ((61, 47), config)), inode_2_0 -> imce_2_1
    // generate: send - TensorEdge((-48, config), (60, config)), inode_2_0 -> imce_2_2
    var4 = 16416;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 5, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-48, config), (60, config)), inode_2_0 -> imce_2_2
    // generate: send - TensorEdge((-58, fused_scale), (67, fused_scale)), inode_2_0 -> imce_2_3
    var4 = 16448;
    for (int i1 = 0; i1 < 2; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 6, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge((-58, fused_scale), (67, fused_scale)), inode_2_0 -> imce_2_3
    // generate: send - TensorEdge((-59, fused_bias), (67, fused_bias)), inode_2_0 -> imce_2_3
    var4 = 16512;
    for (int i1 = 0; i1 < 2; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 7, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge((-59, fused_bias), (67, fused_bias)), inode_2_0 -> imce_2_3
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
    var9 = (int*)(32768);
    var11 = 0;
    var10 = var9[0];
    __asm__ volatile("nop");
    for (int i1 = 0; i1 < var10; i1++) { // generate
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
    var1 = 256;
    for (int i1 = 0; i1 < 10; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 0);
    } // endgenerate
    var1 = 576;
    for (int i1 = 0; i1 < 12; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 960;
    for (int i1 = 0; i1 < 8; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 2);
    } // endgenerate
    var1 = 1216;
    for (int i1 = 0; i1 < 6; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 3);
    } // endgenerate
    var1 = 1408;
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
    var2 = 1472;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 45; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 6);
    } // endgenerate
    // endgenerate: imem write: imce_3_1
    // generate: imem write: imce_3_2
    var2 = 2912;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 10; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 7);
    } // endgenerate
    // endgenerate: imem write: imce_3_2
    // generate: imem write: imce_3_3
    var2 = 3232;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 22; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 8);
    } // endgenerate
    // endgenerate: imem write: imce_3_3
    // generate: imem write: imce_3_4
    var2 = 3936;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 9);
    // endgenerate
    // endgenerate: imem write: imce_3_4

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
    // generate: imce_3_3 compute
    __builtin_INODE_IMCE_COMPUTE(0, 8);
    // endgenerate: imce_3_3 compute
    // generate: wait all imce compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(10, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: wait all imce compute enable
    // generate: send - TensorEdge((-49, fused_scale), (62, fused_scale)), inode_3_0 -> imce_3_1
    var4 = 0;
    for (int i1 = 0; i1 < 2; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 2, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge((-49, fused_scale), (62, fused_scale)), inode_3_0 -> imce_3_1
    // generate: send - TensorEdge((-50, fused_bias), (62, fused_bias)), inode_3_0 -> imce_3_1
    var4 = 64;
    for (int i1 = 0; i1 < 2; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 3, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge((-50, fused_bias), (62, fused_bias)), inode_3_0 -> imce_3_1
    // generate: send - TensorEdge((-60, scale), ((69, 63), rhs)), inode_3_0 -> imce_3_2
    var4 = 192;
    for (int i1 = 0; i1 < 2; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 4, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge((-60, scale), ((69, 63), rhs)), inode_3_0 -> imce_3_2
    // generate: send - TensorEdge((-53, scale), (68, rhs)), inode_3_0 -> imce_3_3
    var4 = 128;
    for (int i1 = 0; i1 < 2; i1++) { // generate
      __builtin_INODE_SEND(var4 + i1*32, 0, 5, 1);

    } // endgenerate
    // endgenerate: send - TensorEdge((-53, scale), (68, rhs)), inode_3_0 -> imce_3_3
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
    var12 = (int*)(32768);
    var11 = 0;
    var13 = var12[0];
    __asm__ volatile("nop");
    for (int i1 = 0; i1 < var13; i1++) { // generate
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
