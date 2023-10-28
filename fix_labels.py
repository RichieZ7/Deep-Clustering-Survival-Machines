import pandas as pd

versions = ['AV45', 'FDG', 'VBM']

for version in versions:
    for seed in [42, 73, 666, 777, 1009]:
        df1 = pd.read_csv("fixed_5_runs/new_all_{}_{}.csv".format(version, seed))
        df2 = pd.read_csv('fixed_5_runs/new_switched_labels_{}_{}.csv'.format(version, seed))

        df2 = df2.set_index('ID')
        df2 = df2.reindex(index = df1['ID'])
        df2.reset_index()

        print(df2)

        df2.to_csv('fixed_5_runs/final_NSC_{}_{}.csv'.format(version, seed))

# for version in versions:
#     df1 = pd.read_csv("new_data/downsampled_clustering_results_{}.csv".format(version))
#     df2 = pd.read_csv('new_data/final_seed_results_{}.csv'.format(version))
#
#     df2 = df2.set_index('x')
#     df2 = df2.reindex(index = df1['ID'])
#     df2.reset_index()
#
#     print(df2)
#
#     df2.to_csv('new_data/ordered_NSC_results_{}.csv'.format(version))
