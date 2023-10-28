"""
This file is for cluster results from features

method : K-means; GNN
"""

from utils.general_utils import combine_t_e
import pandas as pd
from scipy.stats import weibull_min
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_spd_matrix, make_low_rank_matrix
import numpy as np
from numpy.random import multivariate_normal, uniform, choice
import io
import pkgutil
import datetime
from utils.plotting import plot_KM, plot_Weibull_cdf
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
from lifelines.statistics import multivariate_logrank_test
from lifelines import KaplanMeierFitter


def combine_t_e(t, e):  # t is for time and e is for event (indicator)
    # this function combines t and e into y, like a list of tuple
    y = np.zeros(len(t), dtype=[('cens', '?'), ('time', 'f')])
    for idx, item in enumerate(t):
        if e[idx] == 0:
            y[idx] = (False, item)
        else:
            y[idx] = (True, item)
    return y


version = 'AV45'
random_state = [42, 73, 666, 777, 1009]

# versions = ['AV45']
# seeds = [42, 73, 666, 777, 1009]

# for version in versions:
#     for random_state in seeds:

df_all = pd.read_csv("MCI2AD/MCI2AD_{}.csv".format(version))

survival_labels = combine_t_e(df_all['time'], df_all['label'])
final_MCI_ID_list = []
for subj in df_all['ID']:
    final_MCI_ID_list.append(subj)


print('For', version, ', number of subjects are', len(df_all))

#survival_labels = np.array(survival_labels, dtype=[('cens', '?'), ('time', 'f')])

method = ['NSC', 'DCSM']

for i in range(1):
    #x_train, x_test, y_train, y_test, index_train, index_test \
    #    = train_test_split(X, survival_labels, index, test_size=.5, random_state=random_state[i])
    test_MCI_ID_list = []

    y_train = survival_labels
    y_test = survival_labels
    index_test = list(range(0, len(y_test)))
    column_names = ([], final_MCI_ID_list, index_test)

    column_name, MCI_ID_list, index_test = column_names
    # MCI_ID_list_list.append(MCI_ID_list)
    # index_test_list.append(index_test)
    #test_MCI_ID_list.append([MCI_ID_list[i] for i in index_test])

    test_MCI_ID_list = final_MCI_ID_list



    df_cluster  = pd.read_csv(
        'fixed_run/downsampled_clustering_results_{}.csv'.format(version))

    check_subj = []
    index_new = []


    for m in method:

        label = df_cluster[m]

        for aaa in range(len(label)):

            #y_test_new = y_test[id == test_MCI_ID_list]
            #check_subj.append(final_MCI_ID_list[final_MCI_ID_list.index(id)])
            index_new.append(final_MCI_ID_list.index(df_cluster['ID'][aaa]))
            check_subj.append(final_MCI_ID_list[final_MCI_ID_list.index(df_cluster['ID'][aaa])])

        y_test_new = y_test[index_new]

        #if m == 'Kmeans':
        #    y_test_new = y_test

        y_list = []
        index_0 = np.where(label == 0)
        y_list.append(y_test_new[index_0])
        index_1 = np.where(label == 1)
        y_list.append(y_test_new[index_1])

        cluster_tags = label

        cluster_method = m
        data_name = version
        seed = random_state[i]
        # seed = random_state
        is_train = False
        is_lifelines = True
        num_inst = 200
        num_feat = 10

        if is_train:
            stage = 'train'
        else:
            stage = 'test'

        group_indicator = []
        for idx, cluster in enumerate(y_list):
            group_indicator.append([idx] * len(cluster))
        group_indicator = np.concatenate(group_indicator)

        if is_lifelines:
            results = multivariate_logrank_test([item[1] for item in np.concatenate(y_list)],
                                                # item 1 is the survival time
                                                group_indicator,
                                                [int(item[0]) for item in
                                                 np.concatenate(y_list)])  # item 0 is the event
            chisq, pval = results.test_statistic, results.p_value
        else:
            chisq, pval = compare_survival(np.concatenate(y_list), group_indicator)

        print('Test statistic of {}: {:.4e}'.format(stage, chisq))
        print('P value of {}: {:.4e}'.format(stage, pval))
        figure(figsize=(8, 6), dpi=80) #(8.6)

        for idx, cluster in enumerate(y_list):  # each element in the y_list is a cluster
            # use lifelines' KM tool to estimate and plot KM
            # this will provide confidence interval
            if len(cluster) == 0:
                continue
            if is_lifelines:
                kmf = KaplanMeierFitter()
                kmf.fit([item[1] for item in cluster], event_observed=[item[0] for item in cluster],
                        label='Cluster {}, #{}'.format(idx, len(cluster)))
                kmf.plot_survival_function(ci_show=False, show_censors=True)
            else:
                # use scikit-survival's KM tool to estimate and plot KM
                # this does not provide confidence interval
                x, y = kaplan_meier_estimator([item[0] for item in cluster], [item[1] for item in cluster])
                plt.step(x, y, where="post", label='Cluster {}, #{}'.format(idx, len(cluster)))

        plt.title("LogRank: {:.2f}".format(chisq), fontsize=25)
        plt.xlabel("Time", fontsize=25)
        plt.ylabel("Survival Probability", fontsize=25)
        plt.xticks(fontsize=20)#rotation=90
        plt.yticks(fontsize=20)  # rotation=90
        plt.legend(fontsize=25)
        plt.subplots_adjust(left=0.13, bottom=0.15)

        plt.savefig('fixed_5_run_figs/{}_{}_KM_plot_clusters{}_{}.png'.
                    format(cluster_method, stage, len(y_list), version))

        plt.show()
        plt.close()

        print('-----------------------------{} cluster--------------------------------'.format(m))
        print('Logrank is {}'.format(chisq))




#a = np.array([[1, 2], [1, 4], [1, 0],
#              [10, 2], [10, 4], [10, 0]])
#kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(a)

