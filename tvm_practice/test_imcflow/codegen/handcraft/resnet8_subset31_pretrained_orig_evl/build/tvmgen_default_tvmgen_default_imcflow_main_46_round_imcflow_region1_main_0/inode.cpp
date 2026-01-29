#include "../common_decl.h"
void tvmgen_default_tvmgen_default_imcflow_main_46_round_imcflow_region1_main_0() {
  int hid = __builtin_INODE_GET_CORE_HID();
  int wid = 0;
  int var1; // policy_table_start_address
  int var2; // imem_start_address
  int var3; // imcu_start_address
  int var4; // send_data_base_address
  int* var5; // sm13_d38data_send_data_base_address
  int var6; // sm13_d38data_tile_loop_count
  int* var7; // s41_33_d43func_out0_split0_recv_data_base_address
  int var8; // s41_33_d43func_out0_split0_tile_loop_count
  int var9; // recv_data_base_address
  int* var10; // s42_d43func_out1_split1_recv_data_base_address
  int var11; // s42_d43func_out1_split1_tile_loop_count
  if (hid == 0 && wid == 0) { // inode_0_0
    // generate: clear flag before policy update
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: clear flag before policy update
    // generate: policy update
    var1 = 64;
    for (int i1 = 0; i1 < 8; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 0);
    } // endgenerate
    var1 = 320;
    for (int i1 = 0; i1 < 9; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 608;
    __builtin_INODE_PU(var1, 0, 0, 2);
    __builtin_INODE_PU(var1, 32, 1, 2);
    __builtin_INODE_PU(var1, 64, 2, 2);
    __builtin_INODE_PU(var1, 96, 3, 2);
    var1 = 736;
    __builtin_INODE_PU(var1, 0, 0, 3);
    __builtin_INODE_PU(var1, 32, 1, 3);
    __builtin_INODE_PU(var1, 64, 2, 3);
    var1 = 832;
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
    var2 = 896;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 14; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 4);
    } // endgenerate
    // endgenerate: imem write: imce_0_1
    // generate: imem write: imce_0_2
    var2 = 1344;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 5);
    // endgenerate
    // endgenerate: imem write: imce_0_2
    // generate: imem write: imce_0_3
    var2 = 1376;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 6);
    // endgenerate
    // endgenerate: imem write: imce_0_3
    // generate: imem write: imce_0_4
    var2 = 1408;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 7);
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
    __builtin_INODE_IMCE_COMPUTE(0, 4);
    // endgenerate: imce_0_1 compute
    // generate: wait all imce compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(10, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: wait all imce compute enable
    // generate: send - TensorEdge((-24, min), (38, min)), inode_0_0 -> imce_0_1
    var4 = 0;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 2, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-24, min), (38, min)), inode_0_0 -> imce_0_1
    // generate: send - TensorEdge((-25, max), (38, max)), inode_0_0 -> imce_0_1
    var4 = 32;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 3, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-25, max), (38, max)), inode_0_0 -> imce_0_1
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
    for (int i1 = 0; i1 < 3; i1++) { // generate
      // generate: send - TensorEdge((-13, odata), (38, data)), inode_0_0 -> imce_0_1
      var5 = (int*)(13312);
      var4 = 0;
      var6 = var5[0];
      __asm__ volatile("nop");
      for (int i2 = 0; i2 < var6; i2++) { // generate
        __builtin_INODE_SEND(var4 + i2*32, 0, 1, 2);

      } // endgenerate
      // endgenerate: send - TensorEdge((-13, odata), (38, data)), inode_0_0 -> imce_0_1
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
    } // endgenerate
  }
  else if (hid == 1 && wid == 0) { // inode_1_0
    // generate: clear flag before policy update
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: clear flag before policy update
    // generate: policy update
    var1 = 8288;
    for (int i1 = 0; i1 < 10; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 0);
    } // endgenerate
    var1 = 8608;
    for (int i1 = 0; i1 < 12; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 8992;
    __builtin_INODE_PU(var1, 0, 0, 2);
    __builtin_INODE_PU(var1, 32, 1, 2);
    __builtin_INODE_PU(var1, 64, 2, 2);
    __builtin_INODE_PU(var1, 96, 3, 2);
    __builtin_INODE_PU(var1, 128, 4, 2);
    var1 = 9152;
    __builtin_INODE_PU(var1, 0, 0, 3);
    __builtin_INODE_PU(var1, 32, 1, 3);
    __builtin_INODE_PU(var1, 64, 2, 3);
    var1 = 9248;
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
    var2 = 9312;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 130; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 6);
    } // endgenerate
    // endgenerate: imem write: imce_1_1
    // generate: imem write: imce_1_2
    var2 = 13472;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 7);
    // endgenerate
    // endgenerate: imem write: imce_1_2
    // generate: imem write: imce_1_3
    var2 = 13504;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 8);
    // endgenerate
    // endgenerate: imem write: imce_1_3
    // generate: imem write: imce_1_4
    var2 = 13536;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 9);
    // endgenerate
    // endgenerate: imem write: imce_1_4
    // generate: imcu write
    var3 = 0;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 256; i1++) { // generate
      __builtin_INODE_WR_IMCU(var3 + i1*32, 0, 3);
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
    __builtin_INODE_IMCE_COMPUTE(0, 6);
    // endgenerate: imce_1_1 compute
    // generate: wait all imce compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(10, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: wait all imce compute enable
    // generate: send - TensorEdge(((39, -21), config), ((39, 35), config)), inode_1_0 -> imce_1_1
    var4 = 8192;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 4, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((39, -21), config), ((39, 35), config)), inode_1_0 -> imce_1_1
    // generate: send - TensorEdge(((39, -22), fused_scale), ((39, 36), fused_scale)), inode_1_0 -> imce_1_1
    var4 = 8224;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 1, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((39, -22), fused_scale), ((39, 36), fused_scale)), inode_1_0 -> imce_1_1
    // generate: send - TensorEdge(((39, -23), fused_bias), ((39, 36), fused_bias)), inode_1_0 -> imce_1_1
    var4 = 8256;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 2, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((39, -23), fused_bias), ((39, 36), fused_bias)), inode_1_0 -> imce_1_1
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
    for (int i1 = 0; i1 < 3; i1++) { // generate
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
    } // endgenerate
  }
  else if (hid == 2 && wid == 0) { // inode_2_0
    // generate: clear flag before policy update
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: clear flag before policy update
    // generate: policy update
    var1 = 96;
    for (int i1 = 0; i1 < 9; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 0);
    } // endgenerate
    var1 = 384;
    for (int i1 = 0; i1 < 11; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 736;
    for (int i1 = 0; i1 < 7; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 2);
    } // endgenerate
    var1 = 960;
    __builtin_INODE_PU(var1, 0, 0, 3);
    __builtin_INODE_PU(var1, 32, 1, 3);
    __builtin_INODE_PU(var1, 64, 2, 3);
    var1 = 1056;
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
    var2 = 1120;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 17; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 5);
    } // endgenerate
    // endgenerate: imem write: imce_2_1
    // generate: imem write: imce_2_2
    var2 = 1664;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 9; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 6);
    } // endgenerate
    // endgenerate: imem write: imce_2_2
    // generate: imem write: imce_2_3
    var2 = 1952;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 7);
    // endgenerate
    // endgenerate: imem write: imce_2_3
    // generate: imem write: imce_2_4
    var2 = 1984;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 8);
    // endgenerate
    // endgenerate: imem write: imce_2_4

    // generate: sync before compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: sync before compute enable
    // generate: imce_2_1 compute
    __builtin_INODE_IMCE_COMPUTE(0, 5);
    // endgenerate: imce_2_1 compute
    // generate: imce_2_2 compute
    __builtin_INODE_IMCE_COMPUTE(0, 6);
    // endgenerate: imce_2_2 compute
    // generate: wait all imce compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: wait all imce compute enable
    // generate: send - TensorEdge((-26, min), (40, min)), inode_2_0 -> imce_2_1
    var4 = 0;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 2, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-26, min), (40, min)), inode_2_0 -> imce_2_1
    // generate: send - TensorEdge((-27, max), (40, max)), inode_2_0 -> imce_2_1
    var4 = 32;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 3, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-27, max), (40, max)), inode_2_0 -> imce_2_1
    // generate: send - TensorEdge((-28, odata), (42, rhs)), inode_2_0 -> imce_2_2
    var4 = 64;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 4, 3);

    // endgenerate
    // endgenerate: send - TensorEdge((-28, odata), (42, rhs)), inode_2_0 -> imce_2_2
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
    for (int i1 = 0; i1 < 3; i1++) { // generate
      // generate: recv: TensorID(43, func_out1)
      var10 = (int*)(13312);
      var9 = 0;
      var11 = var10[0];
      __asm__ volatile("nop");
      for (int i2 = 0; i2 < var11; i2++) { // generate
        __builtin_INODE_RECV(var9 + i2*32, 0, 0, 2);

      } // endgenerate
      // endgenerate: recv: TensorID(43, func_out1)
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
    } // endgenerate
  }
  else if (hid == 3 && wid == 0) { // inode_3_0
    // generate: clear flag before policy update
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: clear flag before policy update
    // generate: policy update
    var1 = 8288;
    for (int i1 = 0; i1 < 10; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 0);
    } // endgenerate
    var1 = 8608;
    for (int i1 = 0; i1 < 11; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 8960;
    __builtin_INODE_PU(var1, 0, 0, 2);
    __builtin_INODE_PU(var1, 32, 1, 2);
    __builtin_INODE_PU(var1, 64, 2, 2);
    __builtin_INODE_PU(var1, 96, 3, 2);
    var1 = 9088;
    __builtin_INODE_PU(var1, 0, 0, 3);
    __builtin_INODE_PU(var1, 32, 1, 3);
    __builtin_INODE_PU(var1, 64, 2, 3);
    var1 = 9184;
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
    var2 = 9248;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 130; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 6);
    } // endgenerate
    // endgenerate: imem write: imce_3_1
    // generate: imem write: imce_3_2
    var2 = 13408;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 7);
    // endgenerate
    // endgenerate: imem write: imce_3_2
    // generate: imem write: imce_3_3
    var2 = 13440;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 8);
    // endgenerate
    // endgenerate: imem write: imce_3_3
    // generate: imem write: imce_3_4
    var2 = 13472;
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
    // generate: wait all imce compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(10, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: wait all imce compute enable
    // generate: send - TensorEdge(((41, -16), config), ((41, 32), config)), inode_3_0 -> imce_3_1
    var4 = 8192;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 5, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((41, -16), config), ((41, 32), config)), inode_3_0 -> imce_3_1
    // generate: send - TensorEdge(((41, -17), fused_scale), ((41, 33), fused_scale)), inode_3_0 -> imce_3_1
    var4 = 8224;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 2, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((41, -17), fused_scale), ((41, 33), fused_scale)), inode_3_0 -> imce_3_1
    // generate: send - TensorEdge(((41, -18), fused_bias), ((41, 33), fused_bias)), inode_3_0 -> imce_3_1
    var4 = 8256;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 3, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((41, -18), fused_bias), ((41, 33), fused_bias)), inode_3_0 -> imce_3_1
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
    for (int i1 = 0; i1 < 3; i1++) { // generate
      // generate: recv: TensorID(43, func_out0)
      var7 = (int*)(45056);
      var9 = 0;
      var8 = var7[0];
      __asm__ volatile("nop");
      for (int i2 = 0; i2 < var8; i2++) { // generate
        __builtin_INODE_RECV(var9 + i2*32, 0, 0, 2);

      } // endgenerate
      // endgenerate: recv: TensorID(43, func_out0)
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
    } // endgenerate
  }
}
