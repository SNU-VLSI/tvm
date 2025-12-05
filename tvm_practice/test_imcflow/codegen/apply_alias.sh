alias run="python test.py     -k \"test_resnet_cifar10_small_pretrained\" -s 2>&1 | tee ./logs/test_resnet8_and_pretrain_small.log"
alias dbg="python_dbg test.py -k \"test_resnet_cifar10_small_pretrained\" -s 2>&1 | tee ./logs/test_resnet8_and_pretrain_small.log"

alias run="python test.py     -k \"test_resnet and pretrained and (not small)\" -s 2>&1 | tee ./logs/test_resnet8_and_pretrain.log"
alias dbg="python_dbg test.py -k \"test_resnet and pretrained and (not small)\" -s 2>&1 | tee ./logs/test_resnet8_and_pretrain.log"

alias run="python test.py     -k \"test_one_conv_evl\" -s 2>&1 | tee ./logs/test_one_conv_evl.log"
alias dbg="python_dbg test.py -k \"test_one_conv_evl\" -s 2>&1 | tee ./logs/test_one_conv_evl.log"