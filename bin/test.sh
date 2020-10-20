#!/bin/bash
#sum=0

#for i in `seq 1 100`
#do
#	echo $i
#done

var="language is $LANG";
echo $var

cd 'gluonts/lzl_shared_ssm/evaluate/analysis/btc_eth_length(503)_slice(overlap)_past(90)_pred(5)/corr_analysis_pic'

#! echo 会把空格直接换行处理
for i in `ls`
do
   # rm -v 
   echo $i | xargs -n3 #| grep "z_mean_scaled\ vs"
done