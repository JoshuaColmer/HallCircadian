from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
import seaborn as sn
import tqdm

df = pd.read_csv('200407_romanowski_At_data_transcript_norm_av_exp.csv').T
Y_data = df.iloc[1:, -1].astype('float64')
Y_copy = Y_data
J = df.iloc[1:, :-1].astype('float64')
X_ID = df.iloc[0, :-1]
scaler = MinMaxScaler()
periods = []
clock_genes = []
indices = []

def autocorr(x):
    n = x.size
    norm = (x - np.mean(x))
    result = np.correlate(norm, norm, mode='same')
    acorr = result[n//2 + 1:] / (x.var() * np.arange(n-1, n//2, -1))
    lag = np.abs(acorr).argmax() + 1
    r = acorr[lag-1]
    # if np.abs(r) > 0.5:
    #   print('Appears to be autocorrelated with r = {}, lag = {}'. format(r, lag))
    # else:
    #   print('Appears to be not autocorrelated')
    return r, lag

def get_autocorrelated_genes(J, X_ID):
    clock_genes = []
    indices = []
    scores = []
    for i in tqdm.tqdm(range(J.shape[1])):
        fft = np.fft.rfft(J.iloc[:, i], norm="ortho")
        def abs2(x):
            return x.real**2 + x.imag**2

        r, lag = autocorr(J.iloc[:, i])

        if np.abs(r) > 0.5:
            # ax = sn.lineplot(np.arange(J.shape[0])*4, J.iloc[:, i])
            # plt.show()
            clock_genes.append(X_ID.iloc[i])
            scores.append(r)
            indices.append(i)

    return indices, clock_genes, scores


def cross_corr(y1, y2, X_ID):
  """Calculates the cross correlation and lags without normalization.

  The definition of the discrete cross correlation is in:
  https://www.mathworks.com/help/matlab/ref/xcorr.html

  Args:
    y1, y2: Should have the same length.

  Returns:
    max_corr: Maximum correlation without normalization.
    lag: The lag in terms of the index.
  """

  clock_genes = []
  indices = []

  for i in tqdm.tqdm(range(y1.shape[1])):
      y1_auto_corr = np.dot(y1.iloc[:, i], y1.iloc[:, i]) / len(y1.iloc[:, i])
      y2_auto_corr = np.dot(y2, y2) / len(y1.iloc[:, i])
      corr = np.correlate(y1.iloc[:, i], y2, mode='same')

      # The unbiased sample size is N - lag.
      unbiased_sample_size = np.correlate(
          np.ones(len(y1.iloc[:, i])), np.ones(len(y1.iloc[:, i])), mode='same')
      corr = corr / unbiased_sample_size / np.sqrt(y1_auto_corr * y2_auto_corr)
      shift = len(y1.iloc[:, i]) // 2

      max_corr = np.max(corr)
      argmax_corr = np.argmax(corr)
      if max_corr > 1:
          clock_genes.append(X_ID.iloc[i])
          indices.append(i)
  return indices, clock_genes

# cross_corrs = []
# lags = []
#
# Y_copy = np.array([0, 4, 8, 12, 16, 20, 0, 4, 8, 12, 16, 20])
# for i in tqdm.tqdm(range(J.shape[1])):
#     r, l = cross_corr(J.iloc[:, i], Y_copy)
#     cross_corrs.append(r)
#     lags.append(l)
#
# cross_corrs = np.vstack(cross_corrs)
# lags = np.vstack(lags)
#
# cross_corrs1 = np.argsort(cross_corrs.ravel())
# lags1 = lags[cross_corrs1]
#
# ax = sn.lineplot(np.arange(J.shape[0]), J.iloc[:, cross_corrs1[-1]])
# ax = sn.lineplot(np.arange(J.shape[0]), Y_copy)
# plt.show()
