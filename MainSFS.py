import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import seaborn as sn
import math
import tqdm
import gc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, normalize
import FourierClock
from scipy.stats import ks_2samp
from functools import reduce
import random
import os
from numpy.linalg import norm

val_errors1 = []
test_errors1 = []

N_GENES = 50
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

df = pd.read_csv('200407_romanowski_At_data_transcript_tpm_av_exp.csv').T
df_valid = pd.read_csv('200612_yang_At_data_transcript_tpm_all_reps_0counts_rm_LLCTonly.csv').T
df_test = pd.concat((pd.read_csv('200708_forJosh_TPM_lineA.txt').T, pd.read_csv('200708_forJosh_TPM_lineB.txt').T)).iloc[[0, 1, 2, 4, 5], :]
rach_clusters = pd.read_csv('200611_Romanowski_At_WGCNA_6033genes_signed_power16_8clusters_merge_0.15.csv')
Y_data = df.iloc[1:, -1].astype('float64')
Y_copy = Y_data
Y_valid_data = df_valid.iloc[1:, -1].astype('float64')
Y_valid_copy = Y_valid_data

common_IDs = reduce(np.intersect1d, (df.iloc[0, :-1].values, df_valid.iloc[0, :-1].values, df_test.iloc[0, :].values))

idx = np.where(df.iloc[0, :].isin(common_IDs))[0]
df = df.iloc[:, idx]
idx_valid = np.where(df_valid.iloc[0, :].isin(common_IDs))[0]
df_valid = df_valid.iloc[:, idx_valid]
idx_test = np.where(df_test.iloc[0, :].isin(common_IDs))[0]
df_test = df_test.iloc[:, idx_test]

X_data = df.iloc[1:, :].astype('float64')
X_ID = df.iloc[0, :]
X_valid_data = df_valid.iloc[1:, :].astype('float64')
X_valid_ID = df_valid.iloc[0, :]
X_test_data = df_test.iloc[1:, :].astype('float64')
X_test_ID = df_test.iloc[0, :]

X_ID1 = np.argsort(X_ID)
X_ID = X_ID.iloc[X_ID1]
X_data = X_data.iloc[:, X_ID1]
X_data.columns = X_ID
X_ID1 = np.argsort(X_valid_ID)
X_valid_ID = X_valid_ID.iloc[X_ID1]
X_valid_data = X_valid_data.iloc[:, X_ID1]
X_valid_data.columns = X_valid_ID
X_ID1 = np.argsort(X_test_ID)
X_test_ID = X_test_ID.iloc[X_ID1]
X_test_data = X_test_data.iloc[:, X_ID1]
X_test_data.columns = X_test_ID

# Variance threshold
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold()
selector.fit(X_data)
var_idx = selector.variances_ > 8
X_data = X_data.iloc[:, var_idx]
X_ID = X_ID.iloc[var_idx]
X_valid_data = X_valid_data.iloc[:, var_idx]
X_valid_ID = X_valid_ID.iloc[var_idx]
X_test_data = X_test_data.iloc[:, var_idx]
X_test_ID = X_test_ID.iloc[var_idx]

X_data.reset_index(inplace=True, drop=True)
X_valid_data.reset_index(inplace=True, drop=True)
X_test_data.reset_index(inplace=True, drop=True)

X_ID.reset_index(inplace=True, drop=True)
X_valid_ID.reset_index(inplace=True, drop=True)
X_test_ID.reset_index(inplace=True, drop=True)

del df
gc.collect()

n_folds = Y_data.shape[0]
folds = KFold(n_splits=n_folds, random_state=SEED)

y_cos = -np.cos((2 * np.pi * Y_data.astype('float64') / 24)+(np.pi/2))
y_sin = np.sin((2 * np.pi * Y_data.astype('float64') / 24)+(np.pi/2))

Y_valid_cos = -np.cos((2 * np.pi * Y_valid_data.astype('float64') / 24)+(np.pi/2))
Y_valid_sin = np.sin((2 * np.pi * Y_valid_data.astype('float64') / 24)+(np.pi/2))

def cyclical_loss(y_true, y_pred):
    error = 0
    for i in range(y_pred.shape[0]):
        error += np.arccos((y_true[i, :] @ y_pred[i, :]) / (norm(y_true[i, :]) * norm(y_pred[i, :])))
    return error

def custom_loss(y_true, y_pred):
    return tf.reduce_mean((tf.math.acos(tf.matmul(y_true, tf.transpose(y_pred)) / ((tf.norm(y_true) * tf.norm(y_pred)) + tf.keras.backend.epsilon()))**2))

adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)


def larger_model():
    # create model
    model = Sequential()
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, kernel_initializer='normal'))
    # Compile model
    model.compile(loss=custom_loss, optimizer=adam)
    return model

Y_data = np.concatenate((y_cos.values.reshape(-1, 1), y_sin.values.reshape(-1, 1)), axis=1)
Y_valid_data = np.concatenate((Y_valid_cos.values.reshape(-1, 1), Y_valid_sin.values.reshape(-1, 1)), axis=1)

error = 0  # Initialise error
all_preds = np.zeros((Y_data.shape[0], 2))  # Create empty array
all_valid_preds = np.zeros((Y_valid_data.shape[0], 2))  # Create empty array
# all_preds = np.zeros((y.shape[0]))  # Create empty array
# all_preds_circ = np.zeros((y.shape[0], 2))  # Create empty array
early_stop = EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss', mode='min')

auto_indices, auto_clock_genes, auto_scores = FourierClock.get_autocorrelated_genes(X_data, X_ID)
auto_scores = np.abs(np.array(auto_scores))
# auto_scores = np.argsort(auto_scores)
# auto_indices = auto_scores[-N_GENES*6:]
# auto_clock_genes = np.array(auto_clock_genes)
# auto_clock_genes = auto_clock_genes[auto_indices]

cross_indices, cross_clock_genes, cross_scores = FourierClock.cross_corr(X_data, Y_copy, X_ID)
cross_scores = np.abs(np.array(cross_scores))
# cross_scores = np.argsort(cross_scores)
# cross_indices = cross_scores[-N_GENES*6:]
# cross_clock_genes = np.array(cross_clock_genes)
# cross_clock_genes = cross_clock_genes[cross_indices]

scores = np.concatenate((auto_scores.reshape(-1, 1), cross_scores.reshape(-1, 1)), axis=1)

auto_scores = np.argsort(np.mean(scores, axis=1))
scores1 = np.mean(scores, axis=1)
scores2 = np.concatenate((X_data.columns.values.reshape(-1, 1), scores1.reshape(-1, 1)), axis=1)
auto_indices = auto_scores[-N_GENES*50:]
auto_clock_genes = np.array(auto_clock_genes)
auto_clock_genes = auto_clock_genes[auto_indices]
auto_scores1 = scores1[auto_indices]

idx = np.where(X_ID.isin(auto_clock_genes))[0]
X_data = X_data.iloc[:, idx]
scores2 = scores2[idx]
idx_valid = np.where(X_valid_ID.isin(auto_clock_genes))[0]
X_valid_data = X_valid_data.iloc[:, idx_valid]
idx_test = np.where(X_test_ID.isin(auto_clock_genes))[0]
X_test_data = X_test_data.iloc[:, idx_test]

X_ID = X_ID.iloc[idx]
X_valid_ID = X_valid_ID.iloc[idx_valid]
X_test_ID = X_test_ID.iloc[idx_test]

scores = []
pvalues = []

for i in range(X_data.shape[1]):
    l = ks_2samp(X_data.iloc[:, i], X_valid_data.iloc[:, i])
    scores.append(i)
    pvalues.append(l.pvalue)

pvalues_idx = np.argsort(pvalues)
scores = pvalues_idx[(pvalues_idx.shape[0]-30*N_GENES):]

similar_genes = auto_clock_genes[scores]
X_data = X_data.iloc[:, scores]
scores2 = scores2[scores]
X_ID = X_ID.iloc[scores]
X_valid_data = X_valid_data.iloc[:, scores]
X_test_data = X_test_data.iloc[:, scores]
auto_scores1 = auto_scores1[scores]

Y_copy_res = np.array([0, 4, 8, 12, 16, 20, 0, 4, 8, 12, 16, 20])
# indices, clock_genes = FourierClock.cross_corr(X_data, Y_copy_res, X_ID)

# X_data = X_data.iloc[:, indices[-1000:]]

X_ID2 = X_data.columns.values

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(X_data)
X_data = scaler.transform(X_data)
# scaler.fit(X_valid_data)
X_valid_data = scaler.transform(X_valid_data)
X_test_data = scaler.transform(X_test_data)

X_data = pd.DataFrame(data=X_data, columns=X_ID2)
X_valid_data = pd.DataFrame(data=X_valid_data, columns=X_ID2)
X_test_data = pd.DataFrame(data=X_test_data, columns=X_ID2)

column_max = np.max(X_valid_data.values, axis=0)
column_min = np.min(X_valid_data.values, axis=0)
column_idx = column_max < 1.4
column_idx1 = column_min > -0.2
column_idx = np.logical_and(column_idx, column_idx1)
X_data = X_data.iloc[:, column_idx]
scores2 = scores2[column_idx]
X_ID = X_ID.iloc[column_idx]
X_valid_data = X_valid_data.iloc[:, column_idx]
X_test_data = X_test_data.iloc[:, column_idx]
auto_scores1 = auto_scores1[column_idx]

P = pd.DataFrame(data=scores2, columns=['transcript', 'score'])
L = pd.merge(P, rach_clusters, how='left', left_on='transcript', right_on='transcript')

L.fillna(value='None', inplace=True)
print(L['moduleColor'].value_counts())

colours = L['moduleColor'].unique()

results = {'idx': [], 'train_error': [], 'val_error': [], 'test_error': []}
results1 = []
idx_perm = []
graph_res = []

n_genes = 0
i = 0
counter = 0

keep = []

for i in range(colours.shape[0]):
    colour = colours[i]
    genes = L.loc[L['moduleColor'] == colour]
    genes.sort_values(by='score', inplace=True, ascending=False)
    keep.append(genes.index.values[:8])

keep = np.concatenate(keep)
L = L.iloc[keep, :]
X_data = X_data.iloc[:, keep]
X_valid_data = X_valid_data.iloc[:, keep]
X_test_data = X_test_data.iloc[:, keep]

L.reset_index(drop=True, inplace=True)


def reset_seeds(reset_graph_with_backend=None):
    if reset_graph_with_backend is not None:
        K = reset_graph_with_backend
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        print("KERAS AND TENSORFLOW GRAPHS RESET")  # optional

    np.random.seed(1)
    random.seed(2)
    tf.compat.v1.set_random_seed(3)
    # print("RANDOM SEEDS RESET")  # optional

i = 0

while n_genes < N_GENES:
# while n_genes < 2:
    n_genes += 1
    i %= colours.shape[0]
    colour = colours[i]
    genes = L.loc[L['moduleColor'] == colour]
    idx = genes.index.values
    result_iter = {'idx': [], 'train_error': [], 'val_error': [], 'test_error': []}

    for j in tqdm.tqdm(range(idx.shape[0])):
    # for j in tqdm.tqdm(range(2)):
        idx1 = idx[j]
        if n_genes > 2:
            if idx1 in idx_perm:
                result_iter['idx'].append(idx1)
                result_iter['train_error'].append(999.99)
                result_iter['val_error'].append(999.99)
                result_iter['test_error'].append(999.99)
                continue
        if counter == 1:
            idx1 = np.concatenate((np.array([idx1]), np.array(idx_perm).reshape(-1)))
        result_iter['idx'].append(idx1)
        X_d = X_data.iloc[:, idx1].values
        X_v = X_valid_data.iloc[:, idx1].values
        X_t = X_test_data.iloc[:, idx1].values

        valid_preds = []
        test_preds = []
        error = 0  # Initialise error
        all_preds = np.zeros((Y_data.shape[0], 2))  # Create empty array
        all_valid_preds = np.zeros((Y_valid_data.shape[0], 2))  # C

        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_data, Y_data)):
            X_train, Y_train = X_d[train_idx], Y_data[train_idx]  # Define training data for this iteration
            X_valid, Y_valid = X_d[valid_idx], Y_data[valid_idx]
            if n_genes == 1:
                X_train = X_train.reshape(X_train.shape[0], 1)
                X_valid = X_valid.reshape(X_valid.shape[0], 1)
                X_t = X_t.reshape(X_test_data.shape[0], 1)

            # reg = lgb.LGBMRegressor(n_estimators=10000, min_data=1, min_data_in_bin=1)
            # reg = MultiOutputRegressor(lgb.LGBMRegressor(n_estimators=10, min_data=1, min_data_in_bin=1), n_jobs=-1)
            # reg.fit(X_train, Y_train)
            # preds = reg.predict(X_valid)
            # print(cyclical_loss(Y_valid, preds))
            reset_seeds()
            model = larger_model()
            model.fit(X_train.astype('float64'), Y_train.astype('float64'), validation_data=(X_valid.astype('float64'), Y_valid.astype('float64')),
                      batch_size=3, epochs=100, callbacks=[early_stop], verbose=0)  # Fit the model on the training data
            # model.fit(train_x, train_y)
            preds = normalize(model.predict(X_valid))  # Predict on the validation data
            all_preds[valid_idx] = normalize(model.predict(X_valid))
            all_valid_preds += (normalize(model.predict(X_v)) / n_folds)
            valid_preds.append(normalize(model.predict(X_v)))
            test_preds.append(normalize(model.predict(X_t)))
            # y_cos = np.cos((2 * np.pi * y / 24)-(np.pi/2))
            # y_sin = np.sin((2 * np.pi * y / 24)-(np.pi/2))
            # y_circ = np.stack((y_cos, y_sin)).T
            # preds_cos = np.cos((2 * np.pi * preds / 24)-(np.pi/2))
            # preds_sin = np.sin((2 * np.pi * preds / 24)-(np.pi/2))
            # preds_circ = np.stack((preds_cos, preds_sin)).T
            # all_preds_circ[valid_idx] = preds_circ
            error += cyclical_loss(Y_valid.astype('float64'), preds.astype('float64'))  # Evaluate the predictions
            # print(cyclical_loss(Y_valid.astype('float64'), preds.astype('float64')) / Y_valid.shape[0])

        angles = []
        for k in range(all_preds.shape[0]):
            angles.append(math.atan2(all_preds[k, 0], all_preds[k, 1]) / math.pi * 12)

        for l in range(len(angles)):
            if angles[l] < 0:
                angles[l] = angles[l] + 24

        # ax = sn.scatterplot(Y_data[:, 0], Y_data[:, 1])
        # ax = sn.scatterplot(all_preds[:, 0], all_preds[:, 1])
        # plt.show()
        angles_arr = np.vstack(angles)
        hour_pred = angles_arr

        # plt.figure(dpi=500)
        # ax = sn.lineplot(np.arange(Y_copy.shape[0]), Y_copy)
        # ax = sn.lineplot(np.arange(Y_copy.shape[0]), angles_arr.ravel())
        # plt.show()


        angles = []
        for k in range(all_preds.shape[0]):
            angles.append(math.atan2(all_preds[k, 0], all_preds[k, 1]) / math.pi * 12)

        for l in range(len(angles)):
            if angles[l] < 0:
                angles[l] = angles[l] + 24


        valid_angles = []
        valid_preds = np.mean(valid_preds, axis=0)
        for k in range(valid_preds.shape[0]):
            valid_angles.append(math.atan2(valid_preds[k, 0], valid_preds[k, 1]) / math.pi * 12)

        for m in range(len(valid_angles)):
            if valid_angles[m] < 0:
                valid_angles[m] = valid_angles[m] + 24
        valid_preds = normalize(valid_preds)
        # ax = sn.scatterplot(Y_valid_data[:, 0], Y_valid_data[:, 1])
        # ax = sn.scatterplot(valid_preds[:, 0], valid_preds[:, 1])
        # plt.show()
        angles_arr_valid = np.vstack(valid_angles)
        hour_pred_valid = angles_arr_valid


        # plt.figure(dpi=500)
        # ax = sn.lineplot(np.arange(Y_valid_copy.shape[0]), Y_valid_copy)
        # ax = sn.lineplot(np.arange(Y_valid_copy.shape[0]), angles_arr_valid.ravel())
        # plt.show()

        # print("Average error = {}".format(cyclical_loss(Y_data.astype('float64'), all_preds.astype('float64')) / Y_data.shape[0]))
        # print("Average training error = {} minutes".format(60 * 12 * cyclical_loss(Y_data.astype('float64'), all_preds.astype('float64')) / (Y_data.shape[0] * np.pi)))
        result_iter['train_error'].append(60 * 12 * cyclical_loss(Y_data.astype('float64'), all_preds.astype('float64')) / (Y_data.shape[0] * np.pi))

        # print("Average error = {}".format(cyclical_loss(Y_valid_data.astype('float64'), all_valid_preds.astype('float64')) / Y_valid_data.shape[0]))
        # print("Average validation error = {} minutes".format(60 * 12 * cyclical_loss(Y_valid_data.astype('float64'), valid_preds.astype('float64')) / (Y_valid_data.shape[0] * np.pi)))
        result_iter['val_error'].append(60 * 12 * cyclical_loss(Y_valid_data.astype('float64'), valid_preds.astype('float64')) / (Y_valid_data.shape[0] * np.pi))


        Y_copy1 = np.array([2, 5, 8, 11, 14, 17, 20, 23, 2, 5, 8, 11, 14, 17, 20, 23])
        from sklearn.metrics import mean_absolute_error
        # print(mean_absolute_error(Y_copy1, hour_pred_valid.ravel()) * 60, "minutes")

        test_angles = []
        test_preds_copy = test_preds
        test_preds = np.mean(test_preds, axis=0)
        for l in range(len(test_preds_copy)):
            for k in range(test_preds.shape[0]):
                test_preds_copy[l][k, 0] = math.atan2(test_preds_copy[l][k, 0], test_preds_copy[l][k, 1]) / math.pi * 12
                if test_preds_copy[l][k, 0] < 0:
                    test_preds_copy[l][k, 0] += 24
            test_preds_copy[l] = np.delete(test_preds_copy[l], 1, 1)

        for k in range(test_preds.shape[0]):
            test_angles.append(math.atan2(test_preds[k, 0], test_preds[k, 1]) / math.pi * 12)
        for m in range(len(test_angles)):
            if test_angles[m] < 0:
                test_angles[m] = test_angles[m] + 24
        test_preds = normalize(test_preds)
        angles_arr_test = np.vstack(test_angles)
        hour_pred_test = angles_arr_test
        Y_test = np.array([12, 0, 12, 0])

        # print(mean_absolute_error(Y_test, hour_pred_test.ravel()) * 60, "minutes")
        Y_test_cos = -np.cos((2 * np.pi * Y_test.astype('float64') / 24) + (np.pi / 2))
        Y_test_sin = np.sin((2 * np.pi * Y_test.astype('float64') / 24) + (np.pi / 2))
        Y_test_ang = np.concatenate((Y_test_cos.reshape(-1, 1), Y_test_sin.reshape(-1, 1)), axis=1)
        # print("Average test error = {} minutes".format(60 * 12 * cyclical_loss(Y_test_ang.astype('float64'), test_preds.astype('float64')) / (Y_test_ang.shape[0] * np.pi)))

        result_iter['test_error'].append(60 * 12 * cyclical_loss(Y_test_ang.astype('float64'), test_preds.astype('float64')) / (Y_test_ang.shape[0] * np.pi))
    i += 1
    counter = 1
    # idx_perm.append(idx[result_iter['val_error'].index(min(result_iter['val_error']))])
    idx_perm = result_iter['idx'][result_iter['val_error'].index(min(result_iter['val_error']))]
    print(n_genes, idx[result_iter['val_error'].index(min(result_iter['val_error']))], result_iter['train_error'][result_iter['val_error'].index(min(result_iter['val_error']))], result_iter['val_error'][result_iter['val_error'].index(min(result_iter['val_error']))], result_iter['test_error'][result_iter['val_error'].index(min(result_iter['val_error']))])
    graph_res.append((n_genes, idx[result_iter['val_error'].index(min(result_iter['val_error']))], result_iter['train_error'][result_iter['val_error'].index(min(result_iter['val_error']))], result_iter['val_error'][result_iter['val_error'].index(min(result_iter['val_error']))], result_iter['test_error'][result_iter['val_error'].index(min(result_iter['val_error']))]))
    results1.append(result_iter)

# print(L['moduleColor'].value_counts())





# plt.figure(dpi=500)
# ax = sn.lineplot(np.arange(Y_valid_copy.shape[0]), Y_valid_copy)
# ax = sn.lineplot(np.arange(Y_valid_copy.shape[0]), angles_arr_valid.ravel())
# plt.title('Validation circadian time predictions', size=17)
# plt.xlabel('Sample ID', size=17)
# plt.xticks(size=13)
# plt.yticks(size=13)
# plt.ylabel('Circadian time (hr)', size=17)
# plt.legend(['Actual', 'Predicted'], prop={'size': 15})
# plt.tight_layout()
# plt.savefig('SuppFigValError')
# plt.show()

# df = pd.read_csv('200407_romanowski_At_data_transcript_tpm_av_exp.csv').T
# df = df.T
# df_1 = df.loc[df['transcript'].isin(genes.values)]
# df_valid = df_valid.T
# df_valid1 = df_valid.loc[df_valid['transcript'].isin(genes.values)]
# df_test = df_test.T
# df_test1 = df_test.loc[df_test['transcript'].isin(genes.values)]
# i = 13
# ax = sn.lineplot(np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]), df_1.iloc[i, 1:].values.astype('float'))
# ax = sn.lineplot(np.array([2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47]), df_valid1.iloc[i, 1:].values.astype('float'))
# ax = sn.scatterplot(np.array([12, 24]), df_test1.iloc[i, [1, 2]].values.astype('float'))
# ax = sn.scatterplot(np.array([12, 24]), df_test1.iloc[i, [3, 4]].values.astype('float'))
# plt.title('{}'.format(genes.iloc[i]), size=17)
# plt.xlabel('Circadian time (hr)', size=17)
# plt.ylabel('Gene expression (tpm)', size=17)
# plt.xticks(size=13)
# plt.yticks(size=13)
# plt.legend(['Train', 'Validation', 'Test_A', 'Test_B'], prop={'size': 12})
# plt.savefig('{}.png'.format(genes.iloc[i]), dpi=500)
# plt.show()

import pickle
with open('SFSResults3.p', 'wb') as handle:
    pickle.dump(graph_res, handle)


with open('SFSResults4.p', 'wb') as handle:
    pickle.dump(results1, handle)

