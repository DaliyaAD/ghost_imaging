#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For the chosen regression model (linear or random forest), this code will determine
the predicted SSIM value for each matrix statistic. 
The data is sorted into algorithm type as it is assumed this will have disparate
effects on the SSIM so can not be pooled into the same datasets.

The code plots the predicted SSIM against the true SSIM with subplots for each algorithm.

For the same reason, this code only works for a specified sampling ratio.
Code for generating the relevant file 'dataset.csv' is in main.py. 
As of 5/8, it generates over three seeds and sampling ratios 30%, 60%, 90%. 

"""
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

SAVING_INFO = {
    'save_data': True,
    'data_file_name': 'dataset',
    'save_pattern_image': False,
    'image_title': None,
    'image_file_name': None,
    'save_plot': True,
    'plot_file_name': None,
    'plot_title': None,
    'x_label': None,
    'y_label': None,
}


def create_plot(x_data_list, y_data_list, labels, plot_file_name, plot_title):

    x_label = 'True SSIM'
    y_label = 'Predicted SSIM'

    figure = plt.figure(figsize=(8, 9))
    plt.rcParams['font.family'] = 'Times New Roman'
    axes_main_plot = figure.add_subplot(211)
    axes_main_plot.grid()

    for x_data, y_data, label in zip(x_data_list, y_data_list, labels):
        axes_main_plot.scatter(x_data, y_data, marker='x', label=label)

    axes_main_plot.set_title(plot_title, fontsize=14)
    axes_main_plot.set_xlabel(x_label, fontsize=18)
    axes_main_plot.set_ylabel(y_label, fontsize=18)
    axes_main_plot.tick_params(axis='both', labelsize=12)

    axes_main_plot.legend(fontsize=12)

    plt.savefig(plot_file_name, dpi=400)


def fit_model(x_data, y_data, model_type="linear", **model_kwargs):

    if model_type == "Linear":
        model = LinearRegression(**model_kwargs)
    elif model_type == "Random Forest":
        model = RandomForestRegressor(random_state=42, **model_kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(x_data, y_data)
    y_pred = model.predict(x_data)

    r2 = model.score(x_data, y_data)
    mae = mean_absolute_error(y_data, y_pred)
    rmse = np.sqrt(mean_squared_error(y_data, y_pred))

    result = {"r2": r2, "mae": mae, "rmse": rmse}

    if model_type == "Linear":
        result["slope"] = model.coef_[0]
        result["intercept"] = model.intercept_
    elif model_type == "Random Forest":
        result["feature_importance"] = model.feature_importances_[0]

    return result, y_pred


df = pd.read_csv('dataset.csv')
stats = []
label = ['CGI', 'DGI', 'Ridge']
model_type = "Random Forest"
selected_samp_rat = 60
df_filtered = df[df["samp_rat"] == selected_samp_rat]
metric_columns = df_filtered.loc[:, "mean":"mutual coherence"].columns

for column in metric_columns:
    x_data_list = []
    y_data_list = []
    y_pred_list = []
    labels = []

    for recon, subset in df_filtered.groupby("recon_type"):
        metrics = subset.loc[:, "mean":"mutual coherence"]
        ssim = subset["ssim"]

        x_data = metrics[[column]]
        y_data = ssim

        result, y_pred = fit_model(x_data, y_data, model_type=model_type)

        x_data_list.append(x_data)
        y_data_list.append(y_data)
        y_pred_list.append(y_pred)
        labels.append(recon)

        data = {
            "regression_model": model_type,
            "recon_type": recon,
            "samp_rat": selected_samp_rat,
            "metric": column,
            **result,
        }
        stats.append(data)

    plot_file_name = f'{model_type}_{column}_regression.png'
    plot_title = f'Predicted versus true SSIM for {column} using {model_type} regression'
    create_plot(y_data_list, y_pred_list, label, plot_file_name, plot_title)


results_df = pd.DataFrame(stats)
results_df.to_csv('regression_models.csv', index=False)
