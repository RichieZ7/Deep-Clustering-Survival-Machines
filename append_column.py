import pandas as pd

versions = ['AV45', 'FDG', 'VBM']

for version in versions:
    for seed in [42, 73, 666, 777, 1009]:
        df1 = pd.read_csv("fixed_5_runs/new_all_{}_{}.csv".format(version, seed))
        df2 = pd.read_csv('fixed_5_runs/final_NSC_{}_{}.csv'.format(version, seed))

        final_df = df1.copy()
        final_df['NSC'] = df2['NSC']

        final_df.to_csv('fixed_5_runs/5_run_downsampled_clustering_results_{}_{}.csv'.format(version, seed))