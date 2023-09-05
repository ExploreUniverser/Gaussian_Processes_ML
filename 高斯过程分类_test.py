import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
# 定义高斯过程分类函数
def gaussian_process_classification(X_train, y_train, X_test, kernel_func, noise_var):
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # 计算训练集的协方差矩阵
    K = kernel_func(X_train, X_train)
    K += noise_var * np.eye(n_train)  # 添加噪声方差
    
    # 计算测试集与训练集之间的协方差矩阵
    K_star = kernel_func(X_test, X_train)
    
    # 计算预测均值和方差
    pred_mean = np.dot(K_star, np.dot(np.linalg.inv(K), y_train))
    pred_cov = kernel_func(X_test, X_test) - np.dot(K_star,np.dot(np.linalg.inv(K), K_star.T))
    print(pred_mean)
    print(pred_cov)
    # 对预测方差进行修正，确保非负
    pred_cov[pred_cov < 0] = 0
    
    # 预测概率为类别为1的概率
    pred_proba = 1 / (1 + np.exp(-pred_mean / np.sqrt(1 + np.diag(pred_cov))))
    #pred_proba = norm.cdf(pred_mean / np.sqrt(1 + np.diag(pred_cov)))
    
    
    return pred_proba

# 定义高斯核函数
def gaussian_kernel(X1, X2, length_scale=1.0):
    dist_sq = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-0.5 / length_scale**2 * dist_sq)
def squared_exponential_kernel(x1, x2,length_scale=1.0):
    if x1.ndim == 1 and x2.ndim == 1:
        x1 = x1.reshape(-1, 1)
        x2 = x2.reshape(-1, 1)
    dx = cdist(x1, x2)
    return np.exp(-0.5*(dx ** 2) / length_scale**2)
# 生成训练集和测试集数据
X_train = np.array([[1], [2], [3], [4], [5]])  # 训练集特征
y_train = np.array([0, 0, 1, 1, 0])  # 训练集标签
X_test = np.array([[2], [3]])  # 测试集特征

# 设置噪声方差和高斯核的长度尺度
noise_var = 0.01
length_scale = 1.0


# 进行高斯过程分类预测
pred_proba = gaussian_process_classification(X_train, y_train, X_test, lambda x1, x2: squared_exponential_kernel(x1, x2, length_scale), noise_var)
print("手动计算预测概率:", pred_proba)

kernel = RBF(length_scale=1.0)  # 高斯核函数
gpc = GaussianProcessClassifier(kernel=kernel)
gpc.fit(X_train, y_train)

y_pred = gpc.predict(X_test)
y_pred_proba = gpc.predict_proba(X_test)
print("库函数计算预测概率:", y_pred_proba)
