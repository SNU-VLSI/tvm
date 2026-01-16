#define IMCFLOW_ADDR 2147483648
#define IMCFLOW_LEN 266368
#define INT_ACK_GEN_ADDR 0
#define INT_ACK_GEN_LEN 0
#define IMCFLOW_DEVICE "/dev/uio5"
#define INT_ACK_GEN_DEVICE "/dev/uio4"
#define SET_IDLE_CODE 0
#define SET_RUN_CODE 1
#define SET_PROGRAM_CODE 2
#define STATE_REG_IDX 0
#define PC_REG_IDX 2
#define INTR_DONE_REG_IDX 7
#define INODE_PC_START_P1_ENUM_VAL 0
#define INODE_PC_START_EXTERN_ENUM_VAL 1
#define INODE_PC_START_P0_ENUM_VAL 2
#define INODE_NUM 4
#define INODE_0_0_IMEM_BASE_ADDR 128
#define INODE_0_0_POLICY_BASE_ADDR 1152
#define IMCE_0_1_POLICY_BASE_ADDR 1344
#define IMCE_0_2_POLICY_BASE_ADDR 1536
#define IMCE_0_3_POLICY_BASE_ADDR 1696
#define IMCE_0_4_POLICY_BASE_ADDR 1824
#define IMCE_0_1_IMEM_BASE_ADDR 1920
#define IMCE_0_2_IMEM_BASE_ADDR 1952
#define IMCE_0_3_IMEM_BASE_ADDR 1984
#define IMCE_0_4_IMEM_BASE_ADDR 2016
#define INODE_1_0_IMEM_BASE_ADDR 66688
#define INODE_1_0_POLICY_BASE_ADDR 67712
#define IMCE_1_1_POLICY_BASE_ADDR 67872
#define IMCE_1_2_POLICY_BASE_ADDR 68032
#define IMCE_1_3_POLICY_BASE_ADDR 68160
#define IMCE_1_4_POLICY_BASE_ADDR 68256
#define IMCE_1_1_IMEM_BASE_ADDR 68352
#define IMCE_1_2_IMEM_BASE_ADDR 68384
#define IMCE_1_3_IMEM_BASE_ADDR 68416
#define IMCE_1_4_IMEM_BASE_ADDR 68448
#define INODE_2_0_IMEM_BASE_ADDR 133248
#define INODE_2_0_POLICY_BASE_ADDR 134304
#define IMCE_2_1_POLICY_BASE_ADDR 134528
#define IMCE_2_2_POLICY_BASE_ADDR 134752
#define IMCE_2_3_POLICY_BASE_ADDR 134944
#define IMCE_2_4_POLICY_BASE_ADDR 135104
#define IMCE_2_1_IMEM_BASE_ADDR 135264
#define IMCE_2_2_IMEM_BASE_ADDR 135296
#define IMCE_2_3_IMEM_BASE_ADDR 135328
#define IMCE_2_4_IMEM_BASE_ADDR 135360
#define INODE_3_0_IMEM_BASE_ADDR 199808
#define INODE_3_0_POLICY_BASE_ADDR 217536
#define IMCE_3_1_POLICY_BASE_ADDR 218112
#define IMCE_3_2_POLICY_BASE_ADDR 218720
#define IMCE_3_3_POLICY_BASE_ADDR 219168
#define IMCE_3_4_POLICY_BASE_ADDR 219520
#define IMCE_3_1_IMEM_BASE_ADDR 219712
#define IMCE_3_2_IMEM_BASE_ADDR 223936
#define IMCE_3_3_IMEM_BASE_ADDR 224480
#define IMCE_3_4_IMEM_BASE_ADDR 228704
#define RHS_m25_BASE_ADDR 134272
#define WEIGHT_m12_BASE_ADDR 200832
#define WEIGHT_m17_BASE_ADDR 209024
#define CONFIG_m13_BASE_ADDR 217216
#define FUSED_SCALE_m14_BASE_ADDR 217248
#define FUSED_BIAS_m15_BASE_ADDR 217280
#define CONFIG_m18_BASE_ADDR 217312
#define SM11_D27DATA_CNT_BASE_ADDR_BASE_ADDR 3200
#define SM10_D31LHS_CNT_BASE_ADDR_BASE_ADDR 3232
#define S31_D32FUNC_OUT1_SPLIT1_CNT_BASE_ADDR_BASE_ADDR 136320
#define S30_22_D32FUNC_OUT0_SPLIT0_CNT_BASE_ADDR_BASE_ADDR 209024
#define DATA_m10_BASE_ADDR 1152
#define FUNC_OUT1_31_BASE_ADDR 134272
#define FUNC_OUT0_22_BASE_ADDR 200832
#include <stdlib.h>
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/c_backend_api.h>
#include <dlpack/dlpack.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

extern "C" { 
  // Inode, IMCE inst binaries
  extern const int32_t _binary_for_scan_inode_inst_start[];
  extern const int32_t _binary_for_scan_inode_inst_end[];
  extern const int32_t _binary_for_scan_imce_inst_start[];
  extern const int32_t _binary_for_scan_imce_inst_end[];

  // extern const int32_t _binary_for_scan_value_start[];
  // extern const int32_t _binary_for_scan_value_end[];
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t program_scan_reg(char* file_name) {
  fprintf(stderr,"START PRGGRAMMING SCAN REGS\n");

  // pointer setting
  uint32_t* npu_pointer = (uint32_t*)IMCFLOW_ADDR;
  uint32_t* int_ack_gen_pointer = (uint32_t*)INT_ACK_GEN_ADDR;

  // transfer scan values to inode DMEM

  // transfer imce inst binaries to inode DMEM

  // transfer policy table bianaries to inode DMEM

  // transfer inode inst binaries to inode **IMEM**

  // set inode PC for policy update
  for(int i=0; i<INODE_NUM; i++) {
    npu_pointer[(PC_REG_IDX + i)] = (INODE_PC_START_EXTERN_ENUM_VAL << 30 + 0);
  }

  // goto SET_PROGRAM state
  npu_pointer[STATE_REG_IDX] = SET_PROGRAM_CODE;
  wait_for_idle(npu_pointer);
  npu_pointer[INTR_DONE_REG_IDX] = 1;

  // set inode PC for RUN
  for(int i=0; i<INODE_NUM; i++) {
    npu_pointer[(PC_REG_IDX + i)] = (INODE_PC_START_P1_ENUM_VAL << 30 + 0);
  }

  // goto RUN state
  npu_pointer[STATE_REG_IDX] = SET_RUN_CODE;
  wait_for_idle(npu_pointer);
  npu_pointer[INTR_DONE_REG_IDX] = 1;

  return 0;
}