import pandas as pd
#42, 666
# 73, 777
# 42, 666, 777, 1009
for data_name in ['VBM']:
    for seed in [42, 73, 666, 777, 1009]:
        # df = pd.read_csv('new_data/seed_results_{}_73.csv'.format(data_name))
        df = pd.read_csv('fixed_5_runs/downsampled_clustering_results_{}_{}.csv'.format(data_name, seed))
        if seed in [42, 666, 777, 1009]:
            df['NSC'] = df['NSC'].replace({0: 1, 1: 0})
        # df['Cluster'] = df['Cluster'].replace({0: 1, 1: 0})
        df.to_csv('fixed_5_runs/new_switched_labels_{}_{}.csv'.format(data_name, seed), index=False)

