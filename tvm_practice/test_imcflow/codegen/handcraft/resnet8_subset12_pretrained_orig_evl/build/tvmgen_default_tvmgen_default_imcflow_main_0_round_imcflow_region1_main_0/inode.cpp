#include "../common_decl.h"
void tvmgen_default_tvmgen_default_imcflow_main_0_round_imcflow_region1_main_0() {
  int hid = __builtin_INODE_GET_CORE_HID();
  int wid = 0;
  int var1; // policy_table_start_address
  int var2; // imem_start_address
  int var3; // imcu_start_address
  int var4; // send_data_base_address
  int* var5; // sm10_d21data_send_data_base_address
  int var6; // sm10_d21data_tile_loop_count
  int* var7; // s24_16_d26func_out0_split0_recv_data_base_address
  int var8; // s24_16_d26func_out0_split0_tile_loop_count
  int var9; // recv_data_base_address
  int* var10; // s25_d26func_out1_split1_recv_data_base_address
  int var11; // s25_d26func_out1_split1_tile_loop_count
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
    for (int i1 = 0; i1 < 6; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 384;
    __builtin_INODE_PU(var1, 0, 0, 2);
    __builtin_INODE_PU(var1, 32, 1, 2);
    __builtin_INODE_PU(var1, 64, 2, 2);
    __builtin_INODE_PU(var1, 96, 3, 2);
    var1 = 512;
    __builtin_INODE_PU(var1, 0, 0, 3);
    __builtin_INODE_PU(var1, 32, 1, 3);
    __builtin_INODE_PU(var1, 64, 2, 3);
    var1 = 608;
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
    var2 = 672;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 2);
    // endgenerate
    // endgenerate: imem write: imce_0_1
    // generate: imem write: imce_0_2
    var2 = 704;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 3);
    // endgenerate
    // endgenerate: imem write: imce_0_2
    // generate: imem write: imce_0_3
    var2 = 736;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 4);
    // endgenerate
    // endgenerate: imem write: imce_0_3
    // generate: imem write: imce_0_4
    var2 = 768;
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
    for (int i1 = 0; i1 < 3; i1++) { // generate
      // generate: send - TensorEdge((-10, odata), (21, data)), inode_0_0 -> imce_3_4
      var5 = (int*)(13312);
      var4 = 0;
      var6 = var5[0];
      __asm__ volatile("nop");
      for (int i2 = 0; i2 < var6; i2++) { // generate
        __builtin_INODE_STANDBY(14, 1);
        __builtin_INODE_SEND(var4 + i2*32, 0, 1, 2);

      } // endgenerate
      // endgenerate: send - TensorEdge((-10, odata), (21, data)), inode_0_0 -> imce_3_4
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
    var1 = 0;
    __builtin_INODE_PU(var1, 0, 0, 0);
    __builtin_INODE_PU(var1, 32, 1, 0);
    __builtin_INODE_PU(var1, 64, 2, 0);
    __builtin_INODE_PU(var1, 96, 3, 0);
    __builtin_INODE_PU(var1, 128, 4, 0);
    var1 = 160;
    for (int i1 = 0; i1 < 6; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 352;
    __builtin_INODE_PU(var1, 0, 0, 2);
    __builtin_INODE_PU(var1, 32, 1, 2);
    __builtin_INODE_PU(var1, 64, 2, 2);
    __builtin_INODE_PU(var1, 96, 3, 2);
    __builtin_INODE_PU(var1, 128, 4, 2);
    var1 = 512;
    __builtin_INODE_PU(var1, 0, 0, 3);
    __builtin_INODE_PU(var1, 32, 1, 3);
    __builtin_INODE_PU(var1, 64, 2, 3);
    __builtin_INODE_PU(var1, 96, 3, 3);
    var1 = 640;
    __builtin_INODE_PU(var1, 0, 0, 4);
    __builtin_INODE_PU(var1, 32, 1, 4);
    __builtin_INODE_PU(var1, 64, 2, 4);
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
    var2 = 736;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 1);
    // endgenerate
    // endgenerate: imem write: imce_1_1
    // generate: imem write: imce_1_2
    var2 = 768;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 2);
    // endgenerate
    // endgenerate: imem write: imce_1_2
    // generate: imem write: imce_1_3
    var2 = 800;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 3);
    // endgenerate
    // endgenerate: imem write: imce_1_3
    // generate: imem write: imce_1_4
    var2 = 832;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 4);
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
    var1 = 32;
    for (int i1 = 0; i1 < 13; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 0);
    } // endgenerate
    var1 = 448;
    for (int i1 = 0; i1 < 13; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 864;
    for (int i1 = 0; i1 < 12; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 2);
    } // endgenerate
    var1 = 1248;
    for (int i1 = 0; i1 < 7; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 3);
    } // endgenerate
    var1 = 1472;
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
    var2 = 1696;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 3);
    // endgenerate
    // endgenerate: imem write: imce_2_1
    // generate: imem write: imce_2_2
    var2 = 1728;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 4);
    // endgenerate
    // endgenerate: imem write: imce_2_2
    // generate: imem write: imce_2_3
    var2 = 1760;
    __builtin_INODE_SET_ADDR_CNT(0);
    // generate. loop count == 1
    __builtin_INODE_WR_IMEM(var2 + 0*32, 0, 5);
    // endgenerate
    // endgenerate: imem write: imce_2_3
    // generate: imem write: imce_2_4
    var2 = 1792;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 12; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 6);
    } // endgenerate
    // endgenerate: imem write: imce_2_4

    // generate: sync before compute enable
    __builtin_INODE_SET_FLAG(255);
    __builtin_INODE_STANDBY(0, 255);
    __builtin_INODE_STANDBY(5, 255);
    __builtin_INODE_STANDBY(15, 255);
    __asm__ volatile("nop\n" "nop\n" "nop\n" "nop\n");
    __builtin_INODE_SET_FLAG(0);
    // endgenerate: sync before compute enable
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
    // generate: send - TensorEdge((-25, odata), (25, rhs)), inode_2_0 -> imce_2_4
    var4 = 0;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 2, 3);

    // endgenerate
    // endgenerate: send - TensorEdge((-25, odata), (25, rhs)), inode_2_0 -> imce_2_4
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
      // generate: recv: TensorID(26, func_out1)
      var10 = (int*)(13312);
      var9 = 0;
      var11 = var10[0];
      __asm__ volatile("nop");
      for (int i2 = 0; i2 < var11; i2++) { // generate
        __builtin_INODE_RECV(var9 + i2*32, 0, 0, 2);

      } // endgenerate
      // endgenerate: recv: TensorID(26, func_out1)
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
    var1 = 16704;
    for (int i1 = 0; i1 < 18; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 0);
    } // endgenerate
    var1 = 17280;
    for (int i1 = 0; i1 < 13; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 1);
    } // endgenerate
    var1 = 17696;
    for (int i1 = 0; i1 < 12; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 2);
    } // endgenerate
    var1 = 18080;
    for (int i1 = 0; i1 < 9; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 3);
    } // endgenerate
    var1 = 18368;
    for (int i1 = 0; i1 < 6; i1++) { // generate
      __builtin_INODE_PU(var1 + i1*32, 0, i1, 4);
    } // endgenerate
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
    var2 = 18560;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 143; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 2);
    } // endgenerate
    // endgenerate: imem write: imce_3_1
    // generate: imem write: imce_3_2
    var2 = 23136;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 21; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 3);
    } // endgenerate
    // endgenerate: imem write: imce_3_2
    // generate: imem write: imce_3_3
    var2 = 23808;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 247; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 4);
    } // endgenerate
    // endgenerate: imem write: imce_3_3
    // generate: imem write: imce_3_4
    var2 = 31712;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 16; i1++) { // generate
      __builtin_INODE_WR_IMEM(var2 + i1*32, 0, 5);
    } // endgenerate
    // endgenerate: imem write: imce_3_4
    // generate: imcu write
    var3 = 0;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 256; i1++) { // generate
      __builtin_INODE_WR_IMCU(var3 + i1*32, 0, 11);
    } // endgenerate
    var3 = 8192;
    __builtin_INODE_SET_ADDR_CNT(0);
    for (int i1 = 0; i1 < 256; i1++) { // generate
      __builtin_INODE_WR_IMCU(var3 + i1*32, 0, 15);
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
    // generate: send - TensorEdge(((24, -13), config), ((24, 15), config)), inode_3_0 -> imce_3_1
    var4 = 16384;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 12, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((24, -13), config), ((24, 15), config)), inode_3_0 -> imce_3_1
    // generate: send - TensorEdge(((24, -14), fused_scale), ((24, 16), fused_scale)), inode_3_0 -> imce_3_1
    var4 = 16416;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 6, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((24, -14), fused_scale), ((24, 16), fused_scale)), inode_3_0 -> imce_3_1
    // generate: send - TensorEdge(((24, -15), fused_bias), ((24, 16), fused_bias)), inode_3_0 -> imce_3_1
    var4 = 16448;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 10, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((24, -15), fused_bias), ((24, 16), fused_bias)), inode_3_0 -> imce_3_1
    // generate: send - TensorEdge((-23, min), (23, min)), inode_3_0 -> imce_3_2
    var4 = 16640;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 7, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-23, min), (23, min)), inode_3_0 -> imce_3_2
    // generate: send - TensorEdge((-24, max), (23, max)), inode_3_0 -> imce_3_2
    var4 = 16672;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 13, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-24, max), (23, max)), inode_3_0 -> imce_3_2
    // generate: send - TensorEdge(((22, -18), config), ((22, 18), config)), inode_3_0 -> imce_3_3
    var4 = 16480;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 16, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((22, -18), config), ((22, 18), config)), inode_3_0 -> imce_3_3
    // generate: send - TensorEdge(((22, -19), fused_scale), ((22, 19), fused_scale)), inode_3_0 -> imce_3_3
    var4 = 16512;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 8, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((22, -19), fused_scale), ((22, 19), fused_scale)), inode_3_0 -> imce_3_3
    // generate: send - TensorEdge(((22, -20), fused_bias), ((22, 19), fused_bias)), inode_3_0 -> imce_3_3
    var4 = 16544;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 14, 1);

    // endgenerate
    // endgenerate: send - TensorEdge(((22, -20), fused_bias), ((22, 19), fused_bias)), inode_3_0 -> imce_3_3
    // generate: send - TensorEdge((-21, min), (21, min)), inode_3_0 -> imce_3_4
    var4 = 16576;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 9, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-21, min), (21, min)), inode_3_0 -> imce_3_4
    // generate: send - TensorEdge((-22, max), (21, max)), inode_3_0 -> imce_3_4
    var4 = 16608;
    // generate. loop count == 1
    __builtin_INODE_SEND(var4 + 0*32, 0, 17, 1);

    // endgenerate
    // endgenerate: send - TensorEdge((-22, max), (21, max)), inode_3_0 -> imce_3_4
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
      // generate: recv: TensorID(26, func_out0)
      var7 = (int*)(45056);
      var9 = 0;
      var8 = var7[0];
      __asm__ volatile("nop");
      for (int i2 = 0; i2 < var8; i2++) { // generate
        __builtin_INODE_RECV(var9 + i2*32, 0, 0, 2);

      } // endgenerate
      // endgenerate: recv: TensorID(26, func_out0)
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
