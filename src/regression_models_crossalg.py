#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The purpose of this code is to ascertain whether regression models trained on 
one algorithm can be used to predict the outcome of another algorithm.

It has similarites to regression_models.py as the code have not been decomposed 
into reuseable funcitons as of 5/8.

Iterating over all 9 combinations of training and testing algorithms, this code
will plot true and predicted data for each matrix statistics.

Crucially, it generates a heatmap which visualises how successful the algorithms
are at predicting one another's statistics.' 
"""
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns

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


def create_plot(x_data_list, y_data_list, plot_file_name, plot_title, x_label, y_label):

    figure = plt.figure(figsize=(8, 9))
    plt.rcParams['font.family'] = 'Times New Roman'
    axes_main_plot = figure.add_subplot(211)
    axes_main_plot.grid()
    labels = ["True data", "Predicted data"]
    for x_data, y_data, label in zip(x_data_list, y_data_list, labels):
        axes_main_plot.scatter(x_data, y_data, marker='x', label=label)

    axes_main_plot.set_title(plot_title, fontsize=14)
    axes_main_plot.set_xlabel(x_label, fontsize=18)
    axes_main_plot.set_ylabel(y_label, fontsize=18)
    axes_main_plot.tick_params(axis='both', labelsize=12)

    axes_main_plot.legend(fontsize=12)

    plt.savefig(plot_file_name, dpi=400)
    plt.close(figure)


def fit_model(x_train, x_test, y_train, y_test, model_type, **model_kwargs):

    if model_type == "Linear":
        model = LinearRegression(**model_kwargs)
    elif model_type == "Random Forest":
        model = RandomForestRegressor(random_state=42, **model_kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    r2 = model.score(x_test, y_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

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
selected_samp_rat = 60
df_filtered = df[df["samp_rat"] == selected_samp_rat]
metric_columns = df_filtered.loc[:, "mean":"mutual coherence"].columns


model_type = "Linear"

subsets = {
    "CGI": df_filtered[df_filtered["recon_type"] == "CGI"],
    "DGI": df_filtered[df_filtered["recon_type"] == "DGI"],
    "Ridge": df_filtered[df_filtered["recon_type"] == "Ridge"],
}

algorithms = ["CGI", "DGI", "Ridge"]

for column in metric_columns:
    for train_algo in algorithms:
        for test_algo in algorithms:
            train_subset = subsets[train_algo]
            test_subset = subsets[test_algo]

            x_train = train_subset[[column]]
            y_train = train_subset["ssim"]
            x_test = test_subset[[column]]
            y_test = test_subset["ssim"]

            result, y_pred = fit_model(
                x_train, x_test, y_train, y_test, model_type)

            if SAVING_INFO['save_plot']:
                plot_file_name = f'{train_algo}_train_{test_algo}_test_{column}_{model_type}.png'
                create_plot(
                    [x_test, x_test],
                    [y_test, y_pred],
                    plot_file_name,
                    f"Testing {test_algo} {column} Using {train_algo} {model_type} Regression Model", f'{column}', 'SSIM'
                )

            data = {
                "regression_model": model_type,
                "training_alg": train_algo,
                "testing_alg": test_algo,
                "samp_rat": selected_samp_rat,
                "metric": column,
                **result,
            }
            stats.append(data)

    results_df = pd.DataFrame(stats)
    column_results = results_df[results_df["metric"] == column]

    pivot_table = column_results.pivot_table(
        index='training_alg',
        columns='testing_alg',
        values='r2',
        aggfunc='mean',
    )

    plt.rcParams['font.family'] = 'Times New Roman'
    figure, axes = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt='.3f',
        cmap='viridis',
        cbar_kws={'label': 'R²'},
        ax=axes
    )
    axes.set_title(
        f'Training Algorithm × Testing Algorithm for {model_type} (Matrix Stat = {column})', fontsize=14)
    axes.set_xlabel('Testing Algorithm', fontsize=14)
    axes.set_ylabel('Training Algorithm', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{column}_regression_model_heatmap_{model_type}.png', dpi=400)
    plt.close(figure)

results_df = pd.DataFrame(stats)
results_df.to_csv(f'cross_alg_{model_type}_results.csv')
