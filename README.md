# README
## Data Preparing
* Download [gold](https://finance.yahoo.com/quote/GOLD/history?p=GOLD) and rename it as `GOLD.csv` ,then put it under `gluonts/lzl_shared_ssm/data_process/raw_data` or Use the dataset in the `/attachment`
* I put the btc and eth dataset(which are clean) under `/attachment`, also move them to the `gluonts/lzl_shared_ssm/data_process/raw_data`
* You can Download the Second Experiment Dataset from [SIGIR FINIR 2020](https://www.biendata.xyz/competition/finir/data/) And then put them under the same folder as above dataset. Uncompress them and put them like photo below
  ![SIGIR dataset](https://gitee.com/joelonglin/pig-go_image/raw/master/img2106/20210608231954.png)


## Needed python package
* Our Model are implemented by python3.6.5 with tensorflow1.14
* There are serveral baseline are use mxnet. This work get a lot of help from Amazon gluonts, you can use `pip install gluonts` to help you install major of packages

## Quick Start

1. **use shell script to train models and make prediction** : 
   * You Can use `bin/run_shared_ssm.sh` to quickly run our Shared State Space Model. Change dataset or Hyperparameter
   *  Other baseline script are also put in the same folder and named as `bin/run_{BASELINE}.sh`
2. **evaluate our result** :
   * you can use `python evaluate/acc_result.py` to checkout accuracy or you can use `python evaluate/rmse_result.py` to checkout the rmse
3. **debug with right working space** : you should checkout that every python script run under `gluonts/lzl_shared_ssm` if you want to `debug` our code.


## Models Source Code
* All Code are put under `gluonts/lzl_shared_ssm`
* `data_process` are for raw_data and data preprocessing code
* `evaluate` are for evaluation. We can use `acc_result.py` or `rmse_result.py` to test output of models. Our Model store output under `evaluate/analysis` and `evaluate/results`
* `models` contain all main result about our model and baseline models. Our model details are put in `shared_SSM.py`