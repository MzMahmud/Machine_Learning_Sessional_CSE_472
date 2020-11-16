# CSE 472: Machine Learning Sessional
This is a course of my **L4T2**(Final Term). As the name suggests its a course on machine learning.
We are to implement different machine learning algorithms from _scrach_.

## Assignment 1: _Decision Tree_ and _AdaBoost_ for Classification

* **Decision Tree** classifier
* **Ensemble Learning** algorithm **AdaBoost** using **Decision Stump**

**Language Used**: `Python`

## Assignment 2: _k-Nearest Neighbor_ and _Naive Bayes_ for Document Classification
- **k-NN algorithm for text classification**
    - **Hamming distance:** each document is represented as a boolean vector, where each bit represents whether the corresponding word appears in the document.
    - **Euclidean distance:** each document is represented as a numeric vector, where each number represents how many times the corresponding word appears in the document.
    - **Cosine similarity with TF-IDF weights:** each document is represented by a numeric vector as in the case of euclidean distance. However, now each number is the [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)(Term Frequency–Inverse Document Frequency) weight for the corresponding word.The similarity between two documents is the dot product of their corresponding vectors, divided by the product of their norms.

    Experimented with $k=1,3,5$ and different distance metric.
- **Naive Bayes for text classification**
    - Considered all the words of document independently,then calculated the probability of the document of being a topic, and then picked up the topic which provides the highest probability score.
    - Tried $10$ different smoothing factors and calculate the accuracy for each value of smoothing factor to get the best performing smoothing factor.
- **T-test for comparison**
    - Ran $50$ iterations with test domcuments.
    - Compared *kNN* and *NB* using **Paired T-test** with **Significance level $\alpha = 0.005,0.01,0.05$**

**Language Used**: `Python`


## Assignment 3: Dimensionality Reduction using Principal Component Analysis and Clustering using Expectation-maximization Algorithm

- **Principal Component Analysis(PCA) implementation** : _X_ be a _NxD_ data matrix where _D_ is the number of dimensions and _N_ is the number of instances.
    - Standardize the data,

        ![](https://latex.codecogs.com/gif.latex?\frac{X-\mu}{\sigma})
    - Construct the co-variance matrix,
    
        ![](https://latex.codecogs.com/gif.latex?S&space;=&space;\frac{1}{N}&space;\sum_{n=1}^{N}&space;(x_i-\mu_i)(x_i-\mu_i)^T)

    - Compute the eigen vectors and eigen values of the co-variance matrix.

    - Now project your data along the two eigen vectors corresponding to the two highest eigen values.

- **Expectation-maximization(EM) Algorithm implementation** : Now we will cluster the two-dimensional data assuming a Gaussian mixture model using the EM algorithm. Let a vector x with dimension D can be generated from any one of the _K_ Gaussian distribution where the probability of selection of Gaussian distribution _k_ is w<sub>k</sub> where,

![](https://latex.codecogs.com/gif.latex?\sum_{k=1}^{K}w_k=1)

and the probability of generation of x from Gaussian distribution is given as,

![](https://latex.codecogs.com/gif.latex?N_k(x_i|\mu_k,\Sigma_k)=\frac{1}{\sqrt{(2\pi)^D|\Sigma_k|}}e^{-\frac{1}{2}(x_i-\mu_k)^T\Sigma_k^{-1}(x_i-\mu_k))

To learn a Gaussian mixture model using EM algorithm, we need to maximize the likelihood function with respect to the parameters. The steps are given below,
1. Initialize the means,covariances and mixing coefficients and evaluate the initial value of the log likelihood.
2. **E step**: Evaluate the conditional distribution of latent factors using the current parameter values,

![](https://latex.codecogs.com/gif.latex?p_{ik}=p(z_i=k|x_i,\mu,\Sigma,w)=\frac{p(x_i|z_i=k,\mu,\Sigma,w)P(z_i=k|\mu,\Sigma,w)}{p(x_i|\mu,\Sigma,w)}=\frac{w_kN_k(x_i|\mu_k,\Sigma_k)}{\sum_{k=1}^{K}w_kN_k(x_i|\mu_k,\Sigma_k)})

3. M step: Re-estimate the parameters using the conditional distribution of latent factors,


![](https://latex.codecogs.com/gif.latex?\mu_k=\frac{\sum_{i=1}^{N}p_{ik}x_i}{\sum_{i=1}^{N}p_{ik}}\\\\\\Sigma_k=\frac{\sum_{i=1}^{N}p_{ik}(x_i-\mu_k)(x_i-\mu_k)^T}{\sum_{i=1}^{N}p_{ik}}\\\\w_k=\frac{\sum_{i=1}^{N}p_{ik}}{N})

4. Evaluate the log likelihood and check for convergence of the log likelihood. If the convergence criterion is not satisfied return to step 2.

![](https://latex.codecogs.com/gif.latex?\ln&space;p(X|\mu,\Sigma,w)=\sum_{i=1}^{N}&space;\ln&space;p(x_i|\mu,\Sigma,w)=\sum_{i=1}^{N}&space;\ln&space;\sum_{k=1}^{K}w_kN_k(x_i|\mu_k,\Sigma_k))