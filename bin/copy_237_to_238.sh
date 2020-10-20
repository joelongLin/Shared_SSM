# test "Hello"="HelloWorld"
file_237='/home/lzl/pycharm/gluon/gluonts/lzl_shared_ssm/evaluate/results/btc_eth_slice\(overlap\)_past\(30\)_pred\(1\)/shared_ssm_without*'
scp -r lzl@10.21.25.237:$file_237 /data1/lzl/python/gluon/gluonts/lzl_shared_ssm/evaluate/results/btc_eth_length\(503\)_slice\(overlap\)_past\(30\)_pred\(1\)/
# for marker in 2 3 4 5
#     do
#     scp -r lzl@10.21.25.237:"$file_237"_"$marker".pkl /data1/lzl/python/gluon/gluonts/lzl_shared_ssm/evaluate/results/btc_eth_length\(503\)_slice\(overlap\)_past\(30\)_pred\(1\)/
#     done
