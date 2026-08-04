import pandas as pd
from scipy.stats import pearsonr, spearmanr

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
                    "metric": column,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p
                })

results_df = pd.DataFrame(results)
results_df.to_csv('results.csv', index=False)