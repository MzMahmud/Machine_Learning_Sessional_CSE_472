# %%
# imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# %%
def PCA_get_transformed_data(X,n_components=2):
    # Data standardization
    sigma = X.std(axis=0)
    mu = X.mean(axis=0)
    X_std = (X - mu)/sigma
    
    # Consturct Covarient Matrix
    N = X_std.shape[0] # number of instances
    mu = X_std.mean(axis=0)
    X_minus_mu = X_std - mu
    cov_mat = X_minus_mu.T.dot(X_minus_mu) / N

    # eigen vectors and eigen values of the co-variance matrix
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    # choose two best eigen vector
    eigen_val_vec_pairs = []
    for i in range(len(eig_vals)):
        eigen_val = np.abs(eig_vals[i])
        eigen_vec = eig_vecs[:,i]
        eigen_val_vec_pairs.append(
            (eigen_val,eigen_vec)
        )
    # sort by absalute value of the eigen value
    eigen_val_vec_pairs.sort(key=lambda eigen_val_vec_pair : eigen_val_vec_pair[0],reverse=True)

    # get pricipal components
    principal_components = np.array(
            [eigen_val_vec_pairs[i][1] for i in range(n_components)]
        ).T

    # project data on pricipal components
    X_projected = np.dot(X,principal_components)

    # return X_projected
    return X_projected


# %%
# probability of generation of x from Gaussian distribution with mean_vect, cov_mat
# P(x | mean_vec,cov_mat)
def gaussian_prob(x,mean_vec,cov_mat):
    D = len(x)
    det_cov = np.linalg.det(cov_mat)
    x_minus_mu = x - mean_vec
    inv_cov = np.linalg.inv(cov_mat)
    z_score = (-1/2) * np.dot(np.dot(x_minus_mu,inv_cov), # 1xD . DxD = 1xD
                              x_minus_mu.T)               # 1xD . Dx1 = 1x1
    return np.exp(z_score) * ((2 * np.pi)**D * det_cov )**(-1/2)


# %%
def log_likelihood(X,means,covs,w):
    """Calculate
    ln p(X|μ,Σ,w) = \sum_{i=1}^{N} ln p(x_i|μ,Σ,w)
                  = \sum_{i=1}^{N} ln \sum_{k=1}^{K} w_k N(x_i|μ_k,Σ_k) 
    """
    sum_ln = 0
    for x in X:
        sum_wN = 0
        for k in range(len(means)):
            sum_wN += ( w[k] * gaussian_prob(x,means[k],covs[k]) )
        sum_ln += np.log(sum_wN)
    return sum_ln 


# %%
# EM Algorithm
def em_algorithm(X,n_cluster=3,max_iter=1000,eps=1e-4):
    n_data = len(X)
    # initial mean = choose k random data point as mean
    indices = np.random.choice(range(n_data),n_cluster,replace=False)
    means = X[indices]
    
    # initial covs = cov of the data
    cov_X = np.cov(X.T)
    covs = np.array([cov_X for _ in range(n_cluster)])

    # initial weight = 1/k
    w = np.array([1/n_cluster for _ in range(n_cluster)])

    # initial log likelihood
    likelihood = log_likelihood(X,means,covs,w)
    for _ in range(max_iter):
        # -------------E Step-------------
        p = np.zeros((n_data,n_cluster))
        for i in range(n_data):
            for k in range(n_cluster):
                p[i,k] = w[k] * gaussian_prob(X[i],means[k],covs[k])
            # normalize
            p[i] /= np.sum(p[i])

        # -------------M Step-------------
        for k in range(n_cluster):
            # adjust mean
            new_mean = np.zeros_like(means[k])
            sum_p_ik = 0
            for i in range(n_data):
                new_mean += p[i,k] * X[i]
                sum_p_ik += p[i,k]
            means[k] = new_mean/sum_p_ik

            # adjust cov
            new_cov = np.zeros_like(covs[k])
            for i in range(n_data):
                x_minus_mu = X[i] - means[k]
                new_cov += p[i,k] * np.outer(x_minus_mu,x_minus_mu)
                # np.dot(x.T,x) -> dot product, not (dx1)*(1xd) = (dxd) matrix
                # np.outer(x,x) -> gives      ,     (dx1)*(1xd) = (dxd) matrix
            covs[k] = new_cov/sum_p_ik

            # adjust w
            new_w = np.zeros_like(w[k])
            for i in range(n_data):
                new_w += p[i,k]
            w[k] = new_w/n_data

        # -------------Evaluation-------------
        likelihood_new = log_likelihood(X,means,covs,w)
        relative_diff  = np.abs((likelihood_new - likelihood)/likelihood) 
        if relative_diff < eps: break
        likelihood = likelihood_new
    return (means, covs, w)



# %%
# read data
# DATA_FILE = "data_test.txt" # test data
DATA_FILE = "data.txt" # final data
data = []
with open(DATA_FILE) as data_file:
    for line in data_file:
        nums = [float(num) for num in line.split()]
        data.append(nums)
X = np.array(data)
print(X.shape)


# %%
# get projected  
X_projected = PCA_get_transformed_data(X)


# %%
# plot projected data
plt.title('PCA Plot') 
plt.xlabel('PC1') 
plt.ylabel('PC2') 
plt.scatter(X_projected[:,0],X_projected[:,1],facecolors='none',edgecolors='b')
plt.show()

# %%
# get means, covs, w from EM Algorithm
means, covs, w = em_algorithm(X_projected,3)


# %%
# print means, covs, w for report
print(repr(means))
print(repr(covs))
print(repr(w))