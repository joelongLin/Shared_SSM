import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from gluonts.dataset.repository.datasets import get_dataset
import os
from tqdm import tqdm
import pickle
from gluonts.evaluation import Evaluator

if ('/lzl_test' not in os.getcwd()):
     os.chdir('gluonts/lzl_test')
     print('change os dir : ',os.getcwd())
dataset_name = "electricity"
elec_ds = get_dataset(dataset_name, regenerate=False)

forecasts_path = 'data/predict_result/forecasts_electricity_{}_{}.pkl'.format(
                336
                ,168
            )
tss_path = 'data/predict_result/tss_electricity_{}_{}.pkl'.format(
                336
                ,168
            )

def plot_prob_forecasts(ts_entry, forecast_entry , no):
    plot_length = 180
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.savefig('../lzl_deepstate/pic/mx_result/result_output_{}.png'.format(no))
    plt.close(fig)

if __name__ == '__main__':
    with open(forecasts_path, 'rb') as fp:
        forecast_list = pickle.load(fp)
    with open(tss_path, 'rb') as fp:
        tss_list = pickle.load(fp)

    for i in tqdm(range(321)):
        plot_prob_forecasts(tss_list[i], forecast_list[i] ,i)

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(tss_list), iter(forecast_list), num_series=len(elec_ds.test))

    print(json.dumps(agg_metrics, indent=4))