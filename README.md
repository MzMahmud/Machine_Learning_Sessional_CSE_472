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