import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity

filepath = "./page-block.csv"
evaluation_metric = 'F1'
epsilon = 0.01
delta = 0.95

data_set = pd.read_csv(filepath, header=None)

labels = data_set.iloc[:, -1].values
samples = data_set.iloc[:, :-1].values

X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2, stratify=labels)
samples_0 = X_train[y_train == 0]
samples_1 = X_train[y_train == 1]

a1 = len(samples_1) / len(X_train)
a0 = 1 - a1

kde_p0 = KernelDensity(kernel='gaussian', bandwidth='silverman')
kde_p0.fit(samples_0)
kde_p1 = KernelDensity(kernel='gaussian', bandwidth='silverman')
kde_p1.fit(samples_1)

n0_num_samples = int(np.var(kde_p0.score_samples(samples_0)) / (np.square(epsilon) * delta))
n1_num_samples = int(np.var(kde_p1.score_samples(samples_1)) / (np.square(epsilon) * delta))

np.random.seed(42)
sample_indices_0 = np.random.choice(np.arange(len(samples_0)), size=n0_num_samples, replace=True)
sample_indices_1 = np.random.choice(np.arange(len(samples_1)), size=n1_num_samples, replace=True)
samples_P_0 = samples_0[sample_indices_0]
samples_P_1 = samples_1[sample_indices_1]

R_ = lambda x: np.exp(kde_p0.score_samples(x)) / np.exp(kde_p1.score_samples(x))

# RRMP algorithm
samples_P = np.concatenate((samples_P_0, samples_P_1))
samples_labels = np.concatenate((np.zeros(n0_num_samples), np.ones(n1_num_samples)))
R_results_P = R_(samples_P)
sorted_indices = np.argsort(R_results_P)
sorted_R_result = R_results_P[sorted_indices]
sorted_labels = samples_labels[sorted_indices]
sum_tp = 0
sum_fp = 0
pi_1_opt = 0
value_opt = 0

for i in range(len(samples_P)):
    if sorted_labels[i] == 1:
        sum_tp += 1 / n1_num_samples
    else:
        sum_fp += 1 / n0_num_samples
    pi_1 = sorted_R_result[i] / (1 + sorted_R_result[i])
    precision = 1 / (1 + (a0 / a1) * (sum_fp / sum_tp))
    recall = sum_tp
    if evaluation_metric == 'precision':
        value = precision
    elif evaluation_metric == 'recall':
        value = recall
    elif evaluation_metric == 'F1':
        value = 2 * precision * recall / (precision + recall)
    else:
        raise ValueError("Invalid evaluation metric! Please choose from 'precision', 'recall', or 'F1'.")
    if value >= value_opt:
        value_opt = value
        pi_1_opt = pi_1

print(evaluation_metric + " Optimal rebalancing ratio:", pi_1_opt)
