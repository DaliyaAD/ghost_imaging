import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

SAVING_INFO = {
    'save_data': True,
    'data_file_name': 'Hadamard',
    'save_pattern_image': False,
    'image_title': None,
    'image_file_name':  None,
    'save_plot': False,
    'plot_file_name': 'SR_vs_SSIM_Ridge',
    'plot_title': None,
    'x_label': 'Sampling Ratio (%)',
    'y_label': 'SSIM',
}


def create_plot(x_data_list, y_data_list, labels, plot_file_name):

    x_label = 'Matrix Statistics'
    y_label = 'SSIM'

    figure = plt.figure(figsize=(8, 9))
    plt.rcParams['font.family'] = 'Times New Roman'
    axes_main_plot = figure.add_subplot(211)
    axes_main_plot.grid()
    axes_main_plot.xaxis.set_major_locator(plt.MultipleLocator(10))

    for x_data, y_data, label in zip(x_data_list, y_data_list, labels):
        axes_main_plot.scatter(x_data, y_data, label=label)

    axes_main_plot.set_title(SAVING_INFO['plot_title'], fontsize=14)
    axes_main_plot.set_xlabel(x_label, fontsize=18)
    axes_main_plot.set_ylabel(y_label, fontsize=18)
    axes_main_plot.tick_params(axis='both', labelsize=12)
    axes_main_plot.legend(fontsize=12)

    plt.savefig(SAVING_INFO['plot_file_name'], dpi=400)


df = pd.read_csv('dataset.csv')


def pearson_corr(x, y):
    correlation, p_value = pearsonr(x, y)
    return correlation, p_value


def spearman_corr(x, y):
    correlation, p_value = spearmanr(x, y)
    return correlation, p_value


results = []

for (recon, samp_rat), subset in df.groupby(["recon_type", "samp_rat"]):
    metrics = subset.loc[:, "mean":"mutual coherence"]
    ssim = subset["ssim"]

    for column in metrics.columns:
        pearson_r, pearson_p = pearson_corr(metrics[column], ssim)
        spearman_r, spearman_p = spearman_corr(metrics[column], ssim)

        results.append({
            "recon_type": recon,
            "samp_rat": samp_rat,
            "statistic": column,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p
        })


results_df = pd.DataFrame(results)
results_df.to_csv('correlation_values.csv', index=False)

for samp_rat, subset in results_df.groupby('samp_rat'):
    subset.to_csv(f'samp_rat_{samp_rat}.csv', index=False)

    pivot_table = subset.pivot_table(
        index='statistic',
        columns='recon_type',
        values='pearson_r',
    )

    plt.rcParams['font.family'] = 'Times New Roman'
    figure, axes = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        pivot_table,
        annot=True,
        fmt='.3f',
        cmap='viridis',
        cbar_kws={'label': 'SSIM'},
        ax=axes
    )

    axes.set_title(
        f'Matrix Statistic × Algorithm Reconstruction Quality (Sampling Ratio = {samp_rat}%)', fontsize=14)
    axes.set_xlabel('Algorithm', fontsize=14)
    axes.set_ylabel('Matrix Statistics', fontsize=14)

    plt.tight_layout()
    plt.savefig(f'stats_algorithm_heatmap_SR{samp_rat}.png', dpi=400)
