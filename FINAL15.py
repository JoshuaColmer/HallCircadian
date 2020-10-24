import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import seaborn as sn
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Nadam, SGD, RMSprop
from sklearn.preprocessing import MinMaxScaler, normalize
from functools import reduce
import random
from numpy.linalg import norm
import os

val_errors1 = []
test_errors1 = []

SEED = 1000
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

genes = ['AT1G13650.1',
'AT3G55450.1',
'AT1G02930.2',
'AT1G79500.3',
'AT5G24850.1',
'AT5G06870.1',
'AT5G41460.1',
'AT5G01820.1',
'AT4G08870.1',
'AT1G75100.1',
'AT2G29650.2',
'AT5G06690.1',
'AT3G17609.2',
'AT4G15690.1',
'AT1G06040.1'
]

X_data = X_data.loc[:, genes]
X_valid_data = X_valid_data.loc[:, genes]
X_test_data = X_test_data.loc[:, genes]

n_folds = Y_data.shape[0]
folds = KFold(n_splits=n_folds, random_state=SEED, shuffle=True)

y_cos = -np.cos((2 * np.pi * Y_data.astype('float64') / 24)+(np.pi/2))
y_sin = np.sin((2 * np.pi * Y_data.astype('float64') / 24)+(np.pi/2))

Y_valid_cos = -np.cos((2 * np.pi * Y_valid_data.astype('float64') / 24)+(np.pi/2))
Y_valid_sin = np.sin((2 * np.pi * Y_valid_data.astype('float64') / 24)+(np.pi/2))

scaler = MinMaxScaler()
scaler.fit(X_data)
X_data = scaler.transform(X_data)
X_valid_data = scaler.transform(X_valid_data)
X_test_data = scaler.transform(X_test_data)

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
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
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
early_stop = EarlyStopping(patience=50, restore_best_weights=True, monitor='val_loss', mode='min')


def reset_seeds(reset_graph_with_backend=None, seed=0):
    if reset_graph_with_backend is not None:
        K = reset_graph_with_backend
        K.clear_session()
        tf.compat.v1.reset_default_graph()

    np.random.seed(seed)
    random.seed(seed)
    tf.compat.v1.set_random_seed(seed)








#
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
              batch_size=1, epochs=5000, callbacks=[early_stop])  # Fit the model on the training data
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
    print(cyclical_loss(Y_valid.astype('float64'), preds.astype('float64')) / Y_valid.shape[0])

angles = []
for i in range(all_preds.shape[0]):
    angles.append(math.atan2(all_preds[i, 0], all_preds[i, 1]) / math.pi * 12)

for j in range(len(angles)):
    if angles[j] < 0:
        angles[j] = angles[j] + 24

ax = sn.scatterplot(Y_data[:, 0], Y_data[:, 1])
ax = sn.scatterplot(all_preds[:, 0], all_preds[:, 1])
plt.show()
angles_arr = np.vstack(angles)
hour_pred = angles_arr

plt.figure(dpi=500)
ax = sn.lineplot(np.arange(Y_copy.shape[0]), Y_copy)
ax = sn.lineplot(np.arange(Y_copy.shape[0]), angles_arr.ravel())
plt.show()


angles = []
for i in range(all_preds.shape[0]):
    angles.append(math.atan2(all_preds[i, 0], all_preds[i, 1]) / math.pi * 12)

for j in range(len(angles)):
    if angles[j] < 0:
        angles[j] = angles[j] + 24


valid_angles = []
valid_preds = np.mean(valid_preds, axis=0)
for i in range(valid_preds.shape[0]):
    valid_angles.append(math.atan2(valid_preds[i, 0], valid_preds[i, 1]) / math.pi * 12)

for j in range(len(valid_angles)):
    if valid_angles[j] < 0:
        valid_angles[j] = valid_angles[j] + 24
valid_preds = normalize(valid_preds)
ax = sn.scatterplot(Y_valid_data[:, 0], Y_valid_data[:, 1])
ax = sn.scatterplot(valid_preds[:, 0], valid_preds[:, 1])
plt.show()
angles_arr_valid = np.vstack(valid_angles)
hour_pred_valid = angles_arr_valid


plt.figure(dpi=500)
ax = sn.lineplot(np.arange(Y_valid_copy.shape[0]), Y_valid_copy)
ax = sn.lineplot(np.arange(Y_valid_copy.shape[0]), angles_arr_valid.ravel())
plt.show()

# print("Average error = {}".format(cyclical_loss(Y_data.astype('float64'), all_preds.astype('float64')) / Y_data.shape[0]))
print("Average training error = {} minutes".format(60 * 12 * cyclical_loss(Y_data.astype('float64'), all_preds.astype('float64')) / (Y_data.shape[0] * np.pi)))

# print("Average error = {}".format(cyclical_loss(Y_valid_data.astype('float64'), all_valid_preds.astype('float64')) / Y_valid_data.shape[0]))
print("Average validation error = {} minutes".format(60 * 12 * cyclical_loss(Y_valid_data.astype('float64'), valid_preds.astype('float64')) / (Y_valid_data.shape[0] * np.pi)))

Y_copy1 = np.array([2, 5, 8, 11, 14, 17, 20, 23, 2, 5, 8, 11, 14, 17, 20, 23])
from sklearn.metrics import mean_absolute_error
# print(mean_absolute_error(Y_copy1, hour_pred_valid.ravel()) * 60, "minutes")

test_angles = []
test_preds_copy = test_preds
test_preds = np.mean(test_preds, axis=0)
for j in range(len(test_preds_copy)):
    for i in range(test_preds.shape[0]):
        test_preds_copy[j][i, 0] = math.atan2(test_preds_copy[j][i, 0], test_preds_copy[j][i, 1]) / math.pi * 12
        if test_preds_copy[j][i, 0] < 0:
            test_preds_copy[j][i, 0] += 24
    test_preds_copy[j] = np.delete(test_preds_copy[j], 1, 1)

for i in range(test_preds.shape[0]):
    test_angles.append(math.atan2(test_preds[i, 0], test_preds[i, 1]) / math.pi * 12)
for j in range(len(test_angles)):
    if test_angles[j] < 0:
        test_angles[j] = test_angles[j] + 24
test_preds = normalize(test_preds)
angles_arr_test = np.vstack(test_angles)
hour_pred_test = angles_arr_test
Y_test = np.array([12, 0, 12, 0])

# print(mean_absolute_error(Y_test, hour_pred_test.ravel()) * 60, "minutes")
Y_test_cos = -np.cos((2 * np.pi * Y_test.astype('float64') / 24) + (np.pi / 2))
Y_test_sin = np.sin((2 * np.pi * Y_test.astype('float64') / 24) + (np.pi / 2))
Y_test_ang = np.concatenate((Y_test_cos.reshape(-1, 1), Y_test_sin.reshape(-1, 1)), axis=1)
print("Average test error = {} minutes".format(60 * 12 * cyclical_loss(Y_test_ang.astype('float64'), test_preds.astype('float64')) / (Y_test_ang.shape[0] * np.pi)))
val_errors1.append(60 * 12 * cyclical_loss(Y_valid_data.astype('float64'), all_valid_preds.astype('float64')) / (Y_valid_data.shape[0] * np.pi))
test_errors1.append(60 * 12 * cyclical_loss(Y_test_ang.astype('float64'), test_preds.astype('float64')) / (Y_test_ang.shape[0] * np.pi))
