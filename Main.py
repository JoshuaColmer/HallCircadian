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
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, normalize, StandardScaler
import FourierClock
from scipy.stats import ks_2samp
from functools import reduce
import random
import os

N_GENES = 250
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED']=str(SEED)

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
selector = VarianceThreshold(threshold=1)
selector.fit(X_data)
var_idx = selector.variances_ > 1
X_data = X_data.iloc[:, var_idx]
X_ID = X_ID.iloc[var_idx]
X_valid_data = X_valid_data.iloc[:, var_idx]
X_valid_ID = X_valid_ID.iloc[var_idx]
X_test_data = X_test_data.iloc[:, var_idx]
X_test_ID = X_test_ID.iloc[var_idx]
# X_data = pd.DataFrame(selector.transform(X_data))
# X_valid_data = pd.DataFrame(selector.transform(X_valid_data))
# X_test_data = pd.DataFrame(selector.transform(X_test_data))

X_data.reset_index(inplace=True, drop=True)
X_valid_data.reset_index(inplace=True, drop=True)
X_test_data.reset_index(inplace=True, drop=True)

X_ID.reset_index(inplace=True, drop=True)
X_valid_ID.reset_index(inplace=True, drop=True)
X_test_ID.reset_index(inplace=True, drop=True)

del df
gc.collect()

n_folds = Y_data.shape[0]
# n_folds = 6
folds = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

y_cos = np.cos((2 * np.pi * Y_data.astype('float64') / 24)-(np.pi/2))
y_sin = np.sin((2 * np.pi * Y_data.astype('float64') / 24)-(np.pi/2))

Y_valid_cos = np.cos((2 * np.pi * Y_valid_data.astype('float64') / 24)-(np.pi/2))
Y_valid_sin = np.sin((2 * np.pi * Y_valid_data.astype('float64') / 24)-(np.pi/2))

def cyclical_loss(y_true, y_pred):
    error = 0
    for i in range(y_pred.shape[0]):
        error += np.abs(math.atan2(y_pred[i, 1], y_pred[i, 0]) - math.atan2(y_true[i, 1], y_true[i, 0]))
    return error

def custom_loss(y_true, y_pred):
    error = tf.abs(tf.atan2(y_pred[:, 1], y_pred[:, 0]) - tf.atan2(y_true[:, 1], y_true[:, 0]))
    return error

adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)


def larger_model():
    # create model
    model = Sequential()
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(512, kernel_initializer='normal', activation='relu'))
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
early_stop = EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss', mode='min')

indices, clock_genes, scores = FourierClock.get_autocorrelated_genes(X_data, X_ID)
scores = np.abs(np.array(scores))
scores = np.argsort(scores)
indices = np.array(indices)
indices = indices[scores]
clock_genes = np.array(clock_genes)
clock_genes = clock_genes[:N_GENES*6]

idx = np.where(X_ID.isin(clock_genes))[0]
X_data = X_data.iloc[:, idx]
idx_valid = np.where(X_valid_ID.isin(clock_genes))[0]
X_valid_data = X_valid_data.iloc[:, idx_valid]
idx_test = np.where(X_test_ID.isin(clock_genes))[0]
X_test_data = X_test_data.iloc[:, idx_test]

scores = []
pvalues = []

for i in range(X_data.shape[1]):
    l = ks_2samp(X_data.iloc[:, i], X_valid_data.iloc[:, i])
    scores.append(i)
    pvalues.append(l.pvalue)

pvalues_idx = np.argsort(pvalues)
scores = pvalues_idx[(pvalues_idx.shape[0]-N_GENES*6):]

similar_genes = clock_genes[scores]
X_data = X_data.iloc[:, scores]
X_ID = X_ID.iloc[scores]
X_valid_data = X_valid_data.iloc[:, scores]
X_test_data = X_test_data.iloc[:, scores]

Y_copy_res = np.array([0, 4, 8, 12, 16, 20, 0, 4, 8, 12, 16, 20])
indices, clock_genes = FourierClock.cross_corr(X_data, Y_copy_res, X_ID)

# X_data = X_data.iloc[:, indices[-1000:]]

P = pd.DataFrame(X_ID).astype('str')
L = pd.merge(P, rach_clusters, how='left', left_on='transcript', right_on='transcript')
print(L['moduleColor'].value_counts())

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(X_data)
X_data = scaler.transform(X_data)
# scaler.fit(X_valid_data)
X_valid_data = scaler.transform(X_valid_data)
X_test_data = scaler.transform(X_test_data)

column_max = np.max(X_valid_data, axis=0)
column_min = np.min(X_valid_data, axis=0)
column_idx = column_max < 1.2
column_idx1 = column_min > -0.2
column_idx = np.logical_and(column_idx, column_idx1)
X_data = X_data[:, column_idx]
X_valid_data = X_valid_data[:, column_idx]
X_test_data = X_test_data[:, column_idx]

# column_max = np.max(X_test_data, axis=0)
# column_min = np.min(X_test_data, axis=0)
# column_idx = column_max < 1.3
# column_idx1 = column_min > -0.3
# column_idx = np.logical_and(column_idx, column_idx1)
# X_data = X_data[:, column_idx]
# X_valid_data = X_valid_data[:, column_idx]
# X_test_data = X_test_data[:, column_idx]

# X_data = X_data[:, [0, 3]]
# X_valid_data = X_valid_data[:, [0, 3]]
# X_test_data = X_test_data[:, [0, 3]]

valid_preds = []
test_preds = []

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_data, Y_data)):
    X_train, Y_train = X_data[train_idx], Y_data[train_idx]  # Define training data for this iteration
    X_valid, Y_valid = X_data[valid_idx], Y_data[valid_idx]

    # reg = lgb.LGBMRegressor(n_estimators=10000, min_data=1, min_data_in_bin=1)
    # reg = MultiOutputRegressor(lgb.LGBMRegressor(n_estimators=10, min_data=1, min_data_in_bin=1), n_jobs=-1)
    # reg.fit(X_train, Y_train)
    # preds = reg.predict(X_valid)
    # print(cyclical_loss(Y_valid, preds))

    model = larger_model()
    model.fit(X_train.astype('float64'), Y_train.astype('float64'), validation_data=(X_valid.astype('float64'), Y_valid.astype('float64')),
              batch_size=2, epochs=5000, callbacks=[early_stop])  # Fit the model on the training data
    # model.fit(train_x, train_y)
    preds = normalize(model.predict(X_valid))  # Predict on the validation data
    all_preds[valid_idx] = normalize(model.predict(X_valid))
    all_valid_preds += (normalize(model.predict(X_valid_data)) / n_folds)
    valid_preds.append(normalize(model.predict(X_valid_data)))
    test_preds.append(normalize(model.predict(X_test_data)))
    # y_cos = np.cos((2 * np.pi * y / 24)-(np.pi/2))
    # y_sin = np.sin((2 * np.pi * y / 24)-(np.pi/2))
    # y_circ = np.stack((y_cos, y_sin)).T
    # preds_cos = np.cos((2 * np.pi * preds / 24)-(np.pi/2))
    # preds_sin = np.sin((2 * np.pi * preds / 24)-(np.pi/2))
    # preds_circ = np.stack((preds_cos, preds_sin)).T
    # all_preds_circ[valid_idx] = preds_circ
    error += cyclical_loss(Y_valid.astype('float64'), preds.astype('float64'))  # Evaluate the predictions
    # error += cyclical_loss(y_circ.astype('float64'), preds_circ.astype('float64'))  # Evaluate the predictions
    print(cyclical_loss(Y_valid.astype('float64'), preds.astype('float64')) / Y_valid.shape[0])

angles = []
for i in range(all_preds.shape[0]):
    angles.append(math.atan2(all_preds[i, 0], all_preds[i, 1]) / math.pi * 6)
ax = sn.scatterplot(Y_data[:, 0], Y_data[:, 1])
ax = sn.scatterplot(all_preds[:, 0], all_preds[:, 1])
plt.show()
angles_arr = np.vstack(angles)
hour_pred = 12 - 2 * angles_arr

plt.figure(dpi=500)
ax = sn.lineplot(np.arange(Y_copy.shape[0]), Y_copy)
ax = sn.lineplot(np.arange(Y_copy.shape[0]), 12 - 2 * angles_arr.ravel())
plt.show()

valid_angles = []
valid_preds = np.mean(valid_preds, axis=0)
for i in range(valid_preds.shape[0]):
    valid_angles.append(math.atan2(valid_preds[i, 0], valid_preds[i, 1]) / math.pi * 6)
valid_preds = normalize(valid_preds)
ax = sn.scatterplot(Y_valid_data[:, 0], Y_valid_data[:, 1])
ax = sn.scatterplot(valid_preds[:, 0], valid_preds[:, 1])
plt.show()
angles_arr_valid = np.vstack(valid_angles)
hour_pred_valid = 12 - 2 * angles_arr_valid


plt.figure(dpi=500)
ax = sn.lineplot(np.arange(Y_valid_copy.shape[0]), Y_valid_copy)
ax = sn.lineplot(np.arange(Y_valid_copy.shape[0]), 12 - 2 * angles_arr_valid.ravel())
plt.show()

print("Average error = {}".format(cyclical_loss(Y_data.astype('float64'), all_preds.astype('float64')) / Y_data.shape[0]))
print("Average error = {} minutes".format(60 * 12 * cyclical_loss(Y_data.astype('float64'), all_preds.astype('float64')) / (Y_data.shape[0] * np.pi)))

print("Average error = {}".format(cyclical_loss(Y_valid_data.astype('float64'), all_valid_preds.astype('float64')) / Y_valid_data.shape[0]))
print("Average error = {} minutes".format(60 * 12 * cyclical_loss(Y_valid_data.astype('float64'), all_valid_preds.astype('float64')) / (Y_valid_data.shape[0] * np.pi)))

Y_copy1 = np.array([2, 5, 8, 11, 14, 17, 20, 23, 2, 5, 8, 11, 14, 17, 20, 23])
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(Y_copy1, hour_pred_valid.ravel()) * 60, "minutes")

test_angles = []
test_preds = np.mean(test_preds, axis=0)
for i in range(test_preds.shape[0]):
    test_angles.append(math.atan2(test_preds[i, 0], test_preds[i, 1]) / math.pi * 6)
test_preds = normalize(test_preds)
angles_arr_test = np.vstack(test_angles)
hour_pred_test = 12 - 2 * angles_arr_test
Y_test = np.array([0, 12, 0, 12])
print(mean_absolute_error(Y_test, hour_pred_test.ravel()) * 60, "minutes")
