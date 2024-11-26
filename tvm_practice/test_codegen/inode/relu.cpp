void test() {
  int hid = __builtin_INODE_GET_CORE_HID();

  if (hid == 0) {
    /*generate: AddPolicyUpdate*/
    int policy_table_start_address;
    
    policy_table_start_address = 1024;
    __builtin_INODE_PU(policy_table_start_address, 0, 0, 0);
    __builtin_INODE_PU(policy_table_start_address, 32, 1, 0);
    __builtin_INODE_PU(policy_table_start_address, 64, 2, 0);
    __builtin_INODE_PU(policy_table_start_address, 96, 3, 0);
    
    policy_table_start_address = 1152;
    __builtin_INODE_PU(policy_table_start_address, 0, 0, 1);
    __builtin_INODE_PU(policy_table_start_address, 32, 1, 1);
    __builtin_INODE_PU(policy_table_start_address, 64, 2, 1);
    __builtin_INODE_PU(policy_table_start_address, 96, 3, 1);
    
    policy_table_start_address = 1280;
    __builtin_INODE_PU(policy_table_start_address, 0, 0, 2);
    __builtin_INODE_PU(policy_table_start_address, 32, 1, 2);
    __builtin_INODE_PU(policy_table_start_address, 64, 2, 2);
    __builtin_INODE_PU(policy_table_start_address, 96, 3, 2);
    
    policy_table_start_address = 1408;
    __builtin_INODE_PU(policy_table_start_address, 0, 0, 3);
    __builtin_INODE_PU(policy_table_start_address, 32, 1, 3);
    __builtin_INODE_PU(policy_table_start_address, 64, 2, 3);
    __builtin_INODE_PU(policy_table_start_address, 96, 3, 3);
    
    policy_table_start_address = 1536;
    __builtin_INODE_PU(policy_table_start_address, 0, 0, 4);
    __builtin_INODE_PU(policy_table_start_address, 32, 1, 4);
    __builtin_INODE_PU(policy_table_start_address, 64, 2, 4);
    __builtin_INODE_PU(policy_table_start_address, 96, 3, 4);
    /*endgenerate: AddPolicyUpdate*/
    
    /*generate: AddCtrl*/
    __builtin_INODE_SET_FLAG(0);
    __builtin_INODE_HALT();
    /*endgenerate: AddCtrl*/
    
    /*generate: AddWriteIMEM*/
    int imem_size;
    int imem_start_address;
    
    imem_start_address = 3072;
    for (int i = 0; i < imem_size; i += 32) {
      __builtin_INODE_WR_IMEM(i, 0, 0);
    }
    
    imem_start_address = 3232;
    for (int i = 0; i < imem_size; i += 32) {
      __builtin_INODE_WR_IMEM(i, 0, 1);
    }
    
    imem_start_address = 3264;
    for (int i = 0; i < imem_size; i += 32) {
      __builtin_INODE_WR_IMEM(i, 0, 2);
    }
    
    imem_start_address = 3296;
    for (int i = 0; i < imem_size; i += 32) {
      __builtin_INODE_WR_IMEM(i, 0, 3);
    }
    /*endgenerate: AddWriteIMEM*/
    
    /*generate: AddSend*/
    int send_start_address;
    int send_size;
    
    send_start_address = 0;
    for (int i = 0; i < send_size; i += 32) {
      __builtin_INODE_SEND(i, 0, 4,1);
    }
    /*endgenerate: AddSend*/
    
    /*generate: AddCtrl*/
    __builtin_INODE_SET_FLAG(1);
    __builtin_INODE_HALT();
    /*endgenerate: AddCtrl*/
    
  }
  if (hid == 1) {
    /*generate: AddPolicyUpdate*/
    int policy_table_start_address;
    
    policy_table_start_address = 1024;
    __builtin_INODE_PU(policy_table_start_address, 0, 0, 0);
    __builtin_INODE_PU(policy_table_start_address, 32, 1, 0);
    __builtin_INODE_PU(policy_table_start_address, 64, 2, 0);
    __builtin_INODE_PU(policy_table_start_address, 96, 3, 0);
    
    policy_table_start_address = 1152;
    __builtin_INODE_PU(policy_table_start_address, 0, 0, 1);
    __builtin_INODE_PU(policy_table_start_address, 32, 1, 1);
    __builtin_INODE_PU(policy_table_start_address, 64, 2, 1);
    __builtin_INODE_PU(policy_table_start_address, 96, 3, 1);
    
    policy_table_start_address = 1280;
    __builtin_INODE_PU(policy_table_start_address, 0, 0, 2);
    __builtin_INODE_PU(policy_table_start_address, 32, 1, 2);
    __builtin_INODE_PU(policy_table_start_address, 64, 2, 2);
    __builtin_INODE_PU(policy_table_start_address, 96, 3, 2);
    
    policy_table_start_address = 1408;
    __builtin_INODE_PU(policy_table_start_address, 0, 0, 3);
    __builtin_INODE_PU(policy_table_start_address, 32, 1, 3);
    __builtin_INODE_PU(policy_table_start_address, 64, 2, 3);
    __builtin_INODE_PU(policy_table_start_address, 96, 3, 3);
    
    policy_table_start_address = 1536;
    __builtin_INODE_PU(policy_table_start_address, 0, 0, 4);
    __builtin_INODE_PU(policy_table_start_address, 32, 1, 4);
    __builtin_INODE_PU(policy_table_start_address, 64, 2, 4);
    __builtin_INODE_PU(policy_table_start_address, 96, 3, 4);
    /*endgenerate: AddPolicyUpdate*/
    
    /*generate: AddCtrl*/
    __builtin_INODE_STANDBY(0, 0);
    __builtin_INODE_DONE();
    __builtin_INODE_INTRT(0);
    __builtin_INODE_HALT();
    /*endgenerate: AddCtrl*/
    
    /*generate: AddWriteIMEM*/
    int imem_size;
    int imem_start_address;
    
    imem_start_address = 3072;
    for (int i = 0; i < imem_size; i += 32) {
      __builtin_INODE_WR_IMEM(i, 0, 0);
    }
    
    imem_start_address = 3232;
    for (int i = 0; i < imem_size; i += 32) {
      __builtin_INODE_WR_IMEM(i, 0, 1);
    }
    
    imem_start_address = 3264;
    for (int i = 0; i < imem_size; i += 32) {
      __builtin_INODE_WR_IMEM(i, 0, 2);
    }
    
    imem_start_address = 3296;
    for (int i = 0; i < imem_size; i += 32) {
      __builtin_INODE_WR_IMEM(i, 0, 3);
    }
    /*endgenerate: AddWriteIMEM*/
    
    /*generate: AddRecv*/
    int recv_start_address;
    int recv_size;
    
    recv_start_address = 0;
    for (int i = 0; i < recv_size; i += 32) {
      __builtin_INODE_RECV(i, 0, 0,1);
    }
    /*endgenerate: AddRecv*/
    
    /*generate: AddCtrl*/
    __builtin_INODE_STANDBY(0, 1);
    __builtin_INODE_DONE();
    __builtin_INODE_INTRT(0);
    __builtin_INODE_HALT();
    /*endgenerate: AddCtrl*/
    
  }
}
