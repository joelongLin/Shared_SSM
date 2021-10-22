# README
This is source code of "Shared State Space Model for Background Information Extraction and Time Series Prediction" (accepted by Neurocomputing)

Author: Ruichu Cai, Zhaolong Lin, Wei Chen, Zhifeng Hao

If this work helps you, please cite our work with bibtex below
```
@article{cai2021shared,
  title={Shared State Space Model for Background Information Extraction and Time Series Prediction},
  author={Cai, Ruichu and Lin, Zhaolong and Chen, Wei and Hao, Zhifeng},
  journal={Neurocomputing},
  year={2021},
  publisher={Elsevier}
}
```

## Data Preparing
* Download [gold](https://finance.yahoo.com/quote/GOLD/history?p=GOLD) and rename it as `GOLD.csv` ,then put it under `gluonts/lzl_shared_ssm/data_process/raw_data` or Use the dataset in the `/attachment`
* I put the cryptocurrency dataset and air quality dataset(which are clean) under `/attachment`, move them to the `gluonts/lzl_shared_ssm/data_process/raw_data`.
* You can Download the Second Experiment Dataset from [SIGIR FINIR 2020](https://www.biendata.xyz/competition/finir/data/) And then put them under the same folder as above dataset. Uncompress them and put them like photo below
  ![SIGIR dataset](https://gitee.com/joelonglin/pig-go_image/raw/master/img2106/20210608231954.png)


## Needed python package
* Our Model is implemented by python3.6.5 with tensorflow1.14
* There are serveral baseline are implemented by mxnet. This work get a lot of help from Amazon gluonts, you can use `pip install gluonts` to help you install major of packages

## Quick Start

1. **use shell script to train models and make prediction** : 
   * You Can use `bin/run_shared_ssm.sh` to quickly run our Shared State Space Model.  The Hyperparameter are set same range as the paper.
   *  Other baseline script are also put in the same folder and named as `bin/run_{BASELINE}.sh`
2. **evaluate prediction of models** :
   * you can use `python evaluate/acc_result.py` to checkout accuracy or you can use `python evaluate/rmse_result.py` to checkout the rmse
3. <u>debug with right working space</u>
   * you should checkout that every python script run under `gluonts/lzl_shared_ssm` if you want to `debug` our code.


## Models Source Code
* All Code are put under `gluonts/lzl_shared_ssm`
* `data_process` are folder for raw_data, data preprocessing code and preprocessed data.
* `evaluate` are for evaluation. We can use `acc_result.py` or `rmse_result.py` to test output of models. Our Model store output under `evaluate/analysis` and `evaluate/results`
* `models` contain all main result about our model and baseline models. Our model details are put in `shared_SSM.py`