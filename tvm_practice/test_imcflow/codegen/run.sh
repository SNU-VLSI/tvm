#!/bin/bash

# --------------------------------------------------------------------------------------------------------------------
# for i in {2..10}; do
# for i in 13 15 17 18 31; do
#     python main.py --model resnet8_subset31_pretrained_orig --dataset cifar10 --sample $i --single-qconv --start-at simulate
#     mv eval_dir/resnet8_subset31_pretrained_orig_evl.baremetal/test_outputs/rtl_runner eval_dir/resnet8_subset31_pretrained_orig_evl.baremetal/test_outputs/rtl_runner_$i
# done

# no noise version both correct indices
# ./run_dataset_eval.sh -i 2141,2167,2168,2170,2182,2185,2190,2194,2196,2199,2211,2225,2236,2250,2254,2297,2318,2319,2322,2325,2326,2328,2330,2341,2354,2398,2399,2405,2430,2433,2437,2440,2448,2467,2471,2480,2494,2502,2503,2527,2553,2586,2588,2600,2604,2610,2615,2617,2626,2627,2638,2640,2651,2652,2657,2674,2676,2711,2713,2729,2731,2734,2739,2751,2753,2756,2761,2768,2770,2776,2780,2789,2791,2816,2827,2852,2860,2866,2867,2876,2880,2882,2883,2888,2889,2891,2893,2895,2898,2904,2913,2915,2921,2930,2933,2935,2939,2941,2942,2958,2965,3002,3005,3021,3022,3026,3031,3040,3041,3042,3052,3053,3060,3076,3082,3093,3101,3104,3109,3115,3116,3123,3134,3135,3140,3148,3172,3178,3186,3187,3188,3189,3195,3196,3203,3205,3210,3217,3223,3227,3232,3233,3244,3245,3256,3260,3261,3265,3269,3272,3273,3277,3283,3292,3304,3308,3311,3327,3330,3332,3333,3334,3337,3345,3350,3375,3377,3378,3380,3383,3384,3385,3393,3395,3403,3437,3451,3453,3458,3461,3470,3472,3476,3480,3482,3513,3531,3537,3578,3585,3590,3592,3596,3599,3609,3626,3630,3644,3653,3665,3681,3695,3698,3705,3722,3739,3751,3760,3769,3781,3802,3804,3807,3813,3827,3836,3858,3860,3872,3876,3892,3894,3902,3913,3914,3915,3917,3919,3931,3938,3949,3964,3972,3975,3982,3985,3986,3989,3994,3999,4000,4006,4018,4024,4034,4037,4039,4041,4047,4058,4063,4064,4068,4085,4086,4088,4090,4092,4103,4105,4116,4142,4168,4178,4180,4183,4186,4214,4222,4233,4234,4251,4261,4267,4272,4289,4296,4297,4311,4317,4319,4326,4333,4340,4351,4354,4355,4362,4364,4370,4382,4395,4397,4403,4412,4429,4456,4481,4486,4489,4492,4496,4502,4503,4507,4512,4534,4538,4541,4544,4566,4587,4592,4604,4619,4622,4629,4634,4655,4657,4659,4676,4681,4690,4694,4707,4711,4725,4728,4735,4745,4768,4781,4787,4802,4812,4825,4832,4837,4844,4851,4853,4854,4869,4871,4878,4881,4905,4923,4924,4926,4932,4934,4944,4956,4958,4967,4972,4977,4984,4991,4994,5006,5020,5022,5054,5070,5073,5078,5080,5087,5100,5103,5105,5114,5140,5145,5153,5178,5187,5188,5194,5211,5219,5221,5223,5244,5246,5252,5256,5257,5273,5303,5304,5305,5325,5326,5330,5331,5337,5338,5351,5356,5363,5365,5382,5383,5406,5423,5427,5457,5464,5481,5521,5527,5551,5558,5568,5574,5589,5606,5619,5621,5623,5648,5650,5662,5663,5669,5672,5688,5696,5698,5699,5706,5712,5729,5746,5770,5778,5780,5782,5788,5790,5791,5793,5810,5811,5818,5824,5842,5844,5847,5856,5866,5869,5885,5898,5904,5913,5929,5939,5946,5948,5953,5955,5965,5975,5989,6004,6008,6011,6012,6030,6032,6039,6044,6046,6057,6064,6109,6120,6132,6157,6179,6180,6188,6209,6217,6223,6234,6259,6276,6278,6280,6295,6303,6313,6318,6319,6324,6325,6343,6346,6348,6358,6365,6368,6374,6385,6388,6390,6395,6408,6409,6412,6416,6420,6429,6456,6478,6489,6494,6506,6509,6511,6517,6519,6521,6526,6534,6545,6564,6569,6573,6576,6599,6600,6602,6612,6613,6627,6631,6632,6636,6657,6663,6666,6676,6682,6691,6699,6720,6723,6725,6726,6734,6740,6748,6754,6758,6777,6786,6790,6807,6810,6811,6818,6829,6831,6844,6861,6862,6882,6883,6923,6926,6927,6945,6955,6958,6960,6961,6983,6989,6996,7011,7019,7030,7032,7037,7040,7042,7047,7059,7072,7076,7086,7087,7092,7094,7097,7120,7121,7122,7142,7153,7167,7187,7190,7191,7195,7216,7217,7219,7222,7226,7235,7250,7293,7303,7307,7342,7343,7350,7355,7364,7381,7396,7418,7425,7426,7432,7459,7478,7492,7505,7506,7507,7511,7513,7518,7526,7564,7568,7576,7599,7612,7622,7633,7635,7637,7653,7675,7696,7703,7718,7724,7745,7762,7782,7808,7825,7826,7832,7836,7837,7838,7847,7859,7863,7886,7888,7899,7914,7927,7951,7952,7965,7966,7968,7987,7995,8004,8011,8014,8016,8018,8041,8042,8063,8067,8072,8076,8078,8106,8109,8112,8116,8117,8124,8128,8132,8133,8135,8137,8149,8152,8159,8177,8188,8208,8219,8221,8223,8226,8237,8240,8269,8272,8278,8284,8295,8319,8322,8328,8335,8343,8357,8364,8373,8380,8407,8413,8415,8419,8431,8437,8439,8441,8444,8447,8460,8461,8510,8514,8516,8519,8530,8531,8535,8540,8560,8568,8587,8614,8615,8627,8628,8629,8653,8672,8675,8685,8695,8696,8704,8722,8732,8737,8745,8757,8759,8777,8779,8782,8793,8838,8843,8845,8852,8864,8872,8875,8883,8888,8890,8896,8899,8909,8913,8929,8931,8934,8935,8937,8943,8949,8953,8968,8970,8974,8976,8983,8994,9000,9004,9010,9011,9013,9014,9015,9028,9033,9038,9056,9057,9063,9091,9103,9108,9114,9130,9140,9144,9158,9167,9176,9202,9203,9206,9209,9210,9236,9256,9259,9268,9278,9280,9282,9284,9296,9301,9303,9309,9312,9313,9318,9329,9331,9332,9334,9360,9374,9379,9385,9390,9391,9394,9395,9399,9411,9412,9428,9434,9440,9458,9463,9466,9472,9476,9478,9508,9543,9552,9554,9570,9575,9581,9585,9587,9599,9617,9619,9632,9650,9667,9671,9676,9695,9698,9702,9710,9717,9726,9736,9757,9769,9784,9799,9806,9814,9832,9833,9864,9867,9871,9879,9890,9899,9900,9914,9930,9932,9943,9944,9946,9953,9973,9975,9981,9986,9990,9992,9995,9999
# CONSOLE_LOG_LEVEL=INFO ./run_dataset_eval.sh 200
# --------------------------------------------------------------------------------------------------------------------

# # --------------------------------------------------------------------------------------------------------------------
# BATCH_SIZE=50
# START=0
# END=9999
# SKIP_SETUP=${SKIP_SETUP:-false}  # Set SKIP_SETUP=true to skip build/transfer/scan for all batches

# for ((i=START; i<=END; i+=BATCH_SIZE)); do
#   # Build comma-separated index list for this batch
#   LAST=$((i + BATCH_SIZE - 1))
#   if [ $LAST -gt $END ]; then
#     LAST=$END
#   fi

#   INDICES=""
#   for ((j=i; j<=LAST; j++)); do
#     if [ -n "$INDICES" ]; then
#       INDICES="${INDICES},$j"
#     else
#       INDICES="$j"
#     fi
#   done

#   echo ""
#   echo "========================================"
#   echo "  Batch: samples $i - $LAST"
#   echo "========================================"

#   # First batch: run build + transfer (steps 1-5) unless SKIP_SETUP=true
#   if [ "$SKIP_SETUP" = false ] && [ "$i" -eq "$START" ]; then
#     ./run_dataset_eval.sh -i "$INDICES"
#   else
#     ./run_dataset_eval.sh -i "$INDICES" -s 1,2,3,4,5
#   fi

#   if [ $? -ne 0 ]; then
#     echo "⚠️  Batch $i-$LAST failed, continuing to next batch..."
#   fi
# done
# # --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# source imcflow-baremetal.sh
# source imcflow-linux.sh
# --noise-csv /root/project/CIM/noise/noise_df/B2_out/hid0_wid1/N32/B2_noise_matrix_per_ch_ordered.csv
# python main.py --model resnet8_subset31_pretrained_orig --dataset cifar10 --sample 1 --single-qconv
# python -m debugpy --listen $TVM_DEBUG_PORT --wait-for-client main.py --model resnet8_subset31_pretrained_orig --driver-v2 \
#       --column-disable-config /root/project/CIM/trained_models/image_classification/NAT/prange_half_psum_duplication_1/B1/col_disabled/disabled.json --random-seed 42
# BOARD="B2" \
# IMCFLOW_SKIP_RELAY_VIZ=1 \
# python main.py --model resnet8_subset31_pretrained_orig --driver-v2 \
#       --column-disable-config /root/project/CIM/noise/noise_df/B2_out/N32/disabled.json --random-seed 42
# python main.py --model resnet8_subset31_pretrained_orig --driver-v2 --ref-models float transformed \
#       --column-disable-config /root/project/CIM/noise/noise_df/B2_out/N32/disabled.json --random-seed 42
# python_dbg main.py --model resnet8_subset31_pretrained_orig --driver-v2 --ref-models float \
#       --noise-csv /root/project/CIM/noise/noise_df/B2_out/hid0_wid1/N32/B2_noise_matrix_per_ch_ordered.csv \
#       --column-disable-config /root/project/CIM/noise/noise_df/B2_out/hid0_wid1/N32/disabled.json --random-seed 42 \
#       --start-at simulate --stop-at simulate

python main.py --model resnet8_subset31_pretrained_orig --driver-v2 --ref-models transformed \
      --num-disable-columns 32 --column-disable-config /root/project/CIM/noise/noise_df/B2_out/N32/disabled.json --random-seed 42 \
      --noise-csv /root/project/CIM/noise/noise_df/B2_out/N32/B2_noise_matrix_per_ch_concat.csv \
      --noise-layout-json /root/project/CIM/noise/noise_df/B2_out/N32/concat_per_core.json

python main.py --model resnet8_subset31_pretrained_orig --driver-v2 --ref-models transformed \
      --num-disable-columns 32 --column-disable-config /root/project/CIM/noise/noise_df/B2_out/N32/disabled.json --random-seed 42 \
      --noise-csv /root/project/CIM/noise/noise_df/B2_out/N32/B2_noise_matrix_per_ch_concat.csv \
      --noise-layout-json /root/project/CIM/noise/noise_df/B2_out/N32/concat_per_core.json \
      --stop-at compile

python main.py --model resnet8_subset31_pretrained_orig --driver-v2 --ref-models transformed \
      --num-disable-columns 32 --column-disable-config /root/project/CIM/noise/noise_df/B2_out/N32/disabled.json --random-seed 42 \
      --noise-csv /root/project/CIM/noise/noise_df/B2_out/N32/B2_noise_matrix_per_ch_concat.csv \
      --noise-layout-json /root/project/CIM/noise/noise_df/B2_out/N32/concat_per_core.json \
      --dataset cifar10 --sample 1 \
      --noise-mode greedy \
      --start-at simulate --stop-at simulate

for s in 0 1 2 ; do
  python main.py --model resnet8_subset31_pretrained_orig --driver-v2 --ref-models transformed \
    --num-disable-columns 32 --column-disable-config /root/project/CIM/noise/noise_df/B2_out/N32/disabled.json --random-seed 42 \
    --noise-csv /root/project/CIM/noise/noise_df/B2_out/N32/B2_noise_matrix_per_ch_concat.csv \
    --noise-layout-json /root/project/CIM/noise/noise_df/B2_out/N32/concat_per_core.json \
    --dataset cifar10 --sample $s --noise-mode greedy \
    --start-at simulate --stop-at simulate
done

for s in 0 1 2 ; do
  python main.py --model resnet8_subset31_pretrained_orig --driver-v2 --ref-models transformed \
    --num-disable-columns 32 --column-disable-config /root/project/CIM/noise/noise_df/B2_out/N32/disabled.json --random-seed 42 \
    --noise-csv /root/project/CIM/noise/noise_df/B2_out/N32/B2_noise_matrix_per_ch_concat.csv \
    --noise-layout-json /root/project/CIM/noise/noise_df/B2_out/N32/concat_per_core.json \
    --dataset cifar10 --sample $s \
    --start-at simulate --stop-at simulate
done
# for s in 1 2 3 4 5; do                                                                                                                                                                                  
#   python main.py --model resnet8_subset31_pretrained_orig --driver-v2 --ref-models transformed \                                                                                                        
#     --num-disable-columns 32 --column-disable-config /root/project/CIM/noise/noise_df/B2_out/N32/disabled.json --random-seed 42 \                                                                       
#     --noise-csv /root/project/CIM/noise/noise_df/B2_out/N32/B2_noise_matrix_per_ch_concat.csv \                                                                                                         
#     --noise-layout-json /root/project/CIM/noise/noise_df/B2_out/N32/concat_per_core.json \                                                                                                              
#     --dataset cifar10 --sample $s --noise-mode greedy \                                                                                                                                                 
#     --start-at simulate --stop-at simulate                                                                                                                                                              
# done

# --------------------------------------------------------------------------------------------------------------------