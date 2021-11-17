import numpy as np
import math
from ct_support_code import *
import time
from scipy.stats import norm

data = np.load('ct_data.npz')
X_train = data['X_train']; X_val = data['X_val']; X_test = data['X_test']
y_train = data['y_train']; y_val = data['y_val']; y_test = data['y_test']

start = time.time()
mean_y_train = y_train.mean()
print('y_train mean: ', mean_y_train)

mean_y_val = y_val.mean()
sem_y_val = np.std(y_val)/np.sqrt(len(y_val))

mean_y_train_reduced = y_train[:5785].mean()
sem_y_train_reduced = np.std(y_train[:5785])/np.sqrt(5785)

print('y_val mean with standard error: ', mean_y_val, u"\u00B1", sem_y_val)
print('y_train[5785:] mean with standard error: ',
      mean_y_train_reduced, u"\u00B1", sem_y_train_reduced)

# Finding indices of constant columns
b = X_train == X_train[0, :]
c = b.all(axis=0)
constant_columns_indices = set([idx for idx, value in enumerate(c) if value])
print(constant_columns_indices)

# Finding indices of duplicate columns and deleting them
n = X_train.shape[1]
compare = np.arange(n)
X_train_dup, indices = np.unique(X_train, axis=1, return_index=True)
duplicate_column_indices = set(compare).difference(indices)
print(duplicate_column_indices)

# Getting all indices to remove
to_remove = duplicate_column_indices.union(constant_columns_indices)
X_train = np.delete(X_train, np.array(list(to_remove)), axis=1)
X_test = np.delete(X_test, np.array(list(to_remove)), axis=1)
X_val = np.delete(X_val, np.array(list(to_remove)), axis=1)


def fit_linreg(X, yy, alpha):
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    
    # Regularizing only weights not biases
    b = np.sqrt(alpha)*np.identity(X.shape[1]-1)
    new_alpha = np.zeros((X.shape[1], X.shape[1]))
    new_alpha[:X.shape[1]-1,:X.shape[1]-1] = b
    
    X = np.vstack((X, new_alpha))
    yy = np.hstack((yy, np.zeros(X.shape[1])))
    w = np.linalg.lstsq(X, yy, rcond=None)[0]
    return w[:-1], w[-1]
    
def rmse(ww, bb, XX, yy):
    return np.sqrt(np.mean(((np.matmul(XX, ww)+bb) - yy)**2))


alpha = 30
w1, b1 = fit_linreg(X_train, y_train, alpha)
w2, b2 = fit_linreg_gradopt(X_train, y_train, alpha)
print('Training RMSE for fit_linreg: ', rmse(w1, b1, X_train, y_train))
print('Validation RMSE for fit_linreg: ', rmse(w1, b1, X_val, y_val))
print('Training RMSE for fit_linreg_gradopt: ', rmse(w2, b2, X_train, y_train))
print('Validation RMSE for fit_linreg_gradopt: ', rmse(w2, b2, X_val, y_val))

K = 20 # number of thresholded classification problems to fit
mx = np.max(y_train); mn = np.min(y_train); hh = (mx-mn)/(K+1)
thresholds = np.linspace(mn+hh, mx-hh, num=K, endpoint=True)


def fit_logreg_gradopt(X, yy, alpha):
    D = X.shape[1]
    args = (X, yy, alpha)
    init = (np.zeros(D), np.array(0))
    ww, bb = minimize_list(logreg_cost, init, args)
    return ww, bb

q3_ww = np.zeros((X_train.shape[1], K))
q3_bb = np.zeros(K)

for kk in range(K):
    labels = y_train > thresholds[kk]
    q3_ww[:, kk], q3_bb[kk] = fit_logreg_gradopt(X_train, labels, alpha)


def activation_log(X, ww, bb):
    s = np.matmul(X, ww) + bb
    return 1 / (1 + np.exp(-s))

X_train_transform = activation_log(X_train, q3_ww, q3_bb)
X_val_transform = activation_log(X_val, q3_ww, q3_bb)

q3_weights, q3_bias = fit_linreg_gradopt(X_train_transform, y_train, alpha)
print('Training RMSE: ',
      rmse(q3_weights, q3_bias, X_train_transform, y_train))
print('Validation RMSE: ',
      rmse(q3_weights, q3_bias, X_val_transform, y_val))

def random_params(K,D):
    ww = np.random.randn(K)
    bb = np.array(0)
    V = np.random.randn(K,D)
    bk = np.random.randn(K)
    return (ww, bb, V, bk)

def fit_nn_gradopt(X, yy, alpha, K, init = "Default"):
    D = X.shape[1]
    args = (X, yy, alpha)
    if init == "Default":
        # Sizes based on the nn_cost arguments
        init = random_params(K,D)
    ww, bb, V, bk = minimize_list(nn_cost, init, args)
    return ww, bb, V, bk

def nn_error(init, X, yy):
    return np.sqrt(np.mean((nn_cost(init, X) - yy)**2))

alpha = 30
K = 20
# With random parameters
default_random = fit_nn_gradopt(X_train, y_train, alpha, K)
# With q3 parameters
question_3 = fit_nn_gradopt(X_train, y_train, alpha, K, (q3_weights, q3_bias, q3_ww.T, q3_bb))
print("RMSE for training set with random initialization parameters: ", nn_error(default_random, X_train, y_train))
print("RMSE for validation set with random initialization parameters: ", nn_error(default_random, X_val, y_val))
print("RMSE for training set with q3 parameters: ", nn_error(question_3, X_train, y_train))
print("RMSE for validation set with q3 parameters: ", nn_error(question_3, X_val, y_val))

default_random = fit_nn_gradopt(X_train, y_train, 30, 20)
baseline_log = np.log(nn_error(default_random, X_val, y_val))

def train_nn_reg(X, y, alpha):
    return nn_error(fit_nn_gradopt(X_train, y_train, alpha, K), X, y)

def get_ys(alphas):
    yy = []
    for a in alphas:
        yy.append(baseline_log -np.log(train_nn_reg(X_val, y_val, a)))
    return np.array(yy)

def acquisition(mu, cov, yy, alphas):
    pi = np.asarray([])
    for i in range(len(alphas)):
        pi = np.append(pi, norm.cdf((mu[i]-np.max(yy))/np.sqrt(cov[i,i])))
    return pi

alphas = np.arange(0, 50, 0.02)
train_alphas = np.random.choice(alphas,3, replace=False)
test_alphas = np.delete(alphas, [round(i/0.02) for i in train_alphas])
yy_gp = get_ys(train_alphas)

for i in range(5):
    mu, cov = gp_post_par(test_alphas, train_alphas, yy_gp)
    pis = acquisition(mu, cov, yy_gp, test_alphas)
    max_pi = np.amax(pis)
    max_alpha = test_alphas[np.argmax(max_pi)]
    train_alphas = np.append(train_alphas, max_alpha)
    test_alphas = np.delete(test_alphas, np.where(test_alphas==max_alpha))
    yy_gp = np.append(yy_gp, baseline_log -np.log(train_nn_reg(X_val, y_val, max_alpha)))

max_alpha = train_alphas[np.argmax(yy_gp)]

val_rmse = train_nn_reg(X_val, y_val, max_alpha)
test_rmse = train_nn_reg(X_test, y_test, max_alpha)

print('Max alpha: ', max_alpha)
print('RMSE for the Validation Set : ', val_rmse)
print('RMSE for the Test Set : ', test_rmse)