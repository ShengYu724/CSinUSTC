x_train_path = '/home/shengyu/workspcace/study_before_master/classification/data/X_train'
y_train_path = '/home/shengyu/workspcace/study_before_master/classification/data/Y_train'
x_test_path  = '/home/shengyu/workspcace/study_before_master/classification/data/X_test'
output_fpath = './output_{}.csv'

import numpy as np


with open(x_train_path) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(y_train_path) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
with open(x_test_path) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)

def _normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized. If 'None', all columns
    #         will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data

    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
        X_std  = np.std(X[:, specified_column], 0).reshape(1, -1)

    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
     
    return X, X_mean, X_std

# Normalize training and testing data
X_train, X_mean, X_std = _normalize(X_train, train = True)
X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)

#常用函数
def _shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def _sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

def _f(X, w, b):
    # This is the logistic regression function, parameterized by w and b
    #
    # Arguements:
    #     X: input data, shape = [batch_size, data_dimension]
    #     w: weight vector, shape = [data_dimension, ]
    #     b: bias, scalar
    # Output:
    #     predicted probability of each row of X being positively labeled, shape = [batch_size, ]
    return _sigmoid(np.matmul(X, w) + b)

def _predict(X, w, b):
    # This function returns a truth value prediction for each row of X 
    # by rounding the result of logistic regression function.
    return np.round(_f(X, w, b)).astype(np.int)
    
def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

#Generative model
# 将两类数据分开，分别计算其对应的均值和协方差
#最终的方差采用两者的加权平均数
x_train_0 = np.array([x for x,y in zip(X_train,Y_train) if y == 0])
x_train_1 = np.array([x for x,y in zip(X_train,Y_train) if y == 1])
x_mean_0 = np.mean(x_train_0,0)
x_mean_1 = np.mean(x_train_1,0)
x_cov_0 = np.dot(np.transpose([x_train_0 - x_mean_0]).reshape(510,-1),x_train_0 - x_mean_0)/x_train_0.shape[0]
x_cov_1 = np.dot(np.transpose([x_train_1 - x_mean_1]).reshape(510,-1),x_train_1 - x_mean_1)/x_train_1.shape[0]
cov_final = (x_cov_0*x_train_0.shape[0] + x_cov_1*x_train_1.shape[0])/X_train.shape[0]

u, s, v = np.linalg.svd(cov_final, full_matrices=False)
inv = np.matmul(v.T * 1 / s, u.T)

# Directly compute weights and bias
w = np.dot(inv, x_mean_0 - x_mean_1)
b =  (-0.5) * np.dot(x_mean_0, np.dot(inv, x_mean_0)) + 0.5 * np.dot(x_mean_1, np.dot(inv, x_mean_1))\
    + np.log(float(x_train_0.shape[0]) / x_train_1.shape[0]) 

# Compute accuracy on training set
Y_train_pred = 1 - _predict(X_train, w, b)
print('Training accuracy: {}'.format(_accuracy(Y_train_pred, Y_train)))


# Predict testing labels
predictions = 1 - _predict(X_test, w, b)
with open(output_fpath.format('generative'), 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))

# Print out the most significant weights
ind = np.argsort(np.abs(w))[::-1]
with open(x_test_path) as f:
    content = f.readline().strip('\n').split(',')
features = np.array(content)
for i in ind[0:10]:
    print(features[i], w[i])
