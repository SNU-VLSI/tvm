#!/bin/bash

(
  cd ${GEM5_RTL_TEST_DIR}
  rm -rf ${GEM5_RTL_TEST_DIR}/logs
  ./run_standalone.sh \
    ${TVM_CODEGEN_TEST_DIR}/scan_gen/scan_executable_make/build/program_scan_reg \
    no \
    "./logs" \
    "266368" \
    "1234" \
    "${TVM_CODEGEN_TEST_DIR}/scan_gen/scan_reg_files"
  
  rm -rf ${TVM_CODEGEN_TEST_DIR}/scan_gen/scan_gen_test_logs
  cp -r ${GEM5_RTL_TEST_DIR}/logs ${TVM_CODEGEN_TEST_DIR}/scan_gen/scan_gen_test_logs
)
