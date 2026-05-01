ln -snf B2_opt_scan_reg_files/set.0325/opt_scan scan_gen/scan_reg_files
./run_dataset_eval.sh -s 1 -o eval_results/set.0325/scan_opt 100

ln -snf const_scan_reg_files/0x00 scan_gen/scan_reg_files
./run_dataset_eval.sh -s 1 -o eval_results/const/0x00 100

# for d in scan_gen/const_scan_reg_files/*/; do
#   name=$(basename "$d")
#   ln -snf "const_scan_reg_files/${name}" scan_gen/scan_reg_files
#   ./run_dataset_eval.sh -s 1,2,4 -o "eval_results/const/${name}" 100
# done
