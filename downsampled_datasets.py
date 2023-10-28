import pandas as pd

for data_name in ['AV45', 'FDG', 'VBM']:
    df1 = pd.read_csv('datasets/final_seed_results_{}.csv'.format(data_name))
    df2 = pd.read_csv('MCI2AD/MCI2AD_{}.csv'.format(data_name))

    final_df = df2[df2['ID'].isin(df1['x'])]
    final_df.to_csv('MCI2AD/downsampled_{}.csv'.format(data_name), index=False)