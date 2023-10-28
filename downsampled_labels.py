import pandas as pd


for data_name in ['AV45', 'FDG', 'VBM']:
    for seed in [42, 73, 666, 777, 1009]:
        df1 = pd.read_csv('fixed_5_runs/new_switched_labels_{}_{}.csv'.format(data_name, seed))
        df2 = pd.read_csv('rerun/clustering_results_{}.csv'.format(data_name))

        final_df = df2[df2['ID'].isin(df1['ID'])]
        final_df.to_csv('fixed_5_runs/new_all_{}_{}.csv'.format(data_name, seed), index=False)

# for data_name in ['AV45', 'FDG', 'VBM']:
#     df1 = pd.read_csv('rerun/switched_labels_{}.csv'.format(data_name))
#     df2 = pd.read_csv('rerun/clustering_results_{}.csv'.format(data_name))
#
#     final_df = df2[df2['ID'].isin(df1['x'])]
#     final_df.to_csv('rerun/all_downsampled_clustering_results_{}.csv'.format(data_name), index=False)