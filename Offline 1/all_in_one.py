#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MinMaxScaler
from winsound import PlaySound

# ------------------------Churn Dataset Preprocessing----------------------------


def churn_preprocess():
    data = pd.read_csv("data/churn.csv", na_values={"TotalCharges": " "})
    # data
    data.info()
    data.isnull().any(axis=0)
    data["TotalCharges"].isnull().sum()

    # so 11 null values in that column
    # replace with 0 as total charges is not available means not charged
    data["TotalCharges"] = data["TotalCharges"].fillna(0)
    data["TotalCharges"].isnull().sum()
    # no nan
    data.isnull().any(axis=0)
    # all columns have non null values

    data["MonthlyCharges"].describe()
    data["MonthlyCharges"] = pd.qcut(data["MonthlyCharges"], q=5)
    data["MonthlyCharges"].unique()

    data["TotalCharges"].describe()
    data["TotalCharges"] = pd.qcut(data["TotalCharges"], q=5)
    data["TotalCharges"].unique()

    # data
    # dont need customer ID in classification
    data.drop(columns="customerID", inplace=True)
    data.replace({"Yes": 1, "No": 0}, inplace=True)
    # data
    data.to_csv("data/churn_proc.csv", index=False)


# -------------------------------------------------------------------------------


# -----------------------------Adult Dataset Preprocesing------------------------


def adult_preprocess():
    data = pd.read_csv("data/adult.csv")  # ,na_values={'TotalCharges': ' '})
    # data
    data.info()

    data.isnull().sum()

    data["age"].value_counts()
    # replace with 0 as total charges is not available means not charged
    data["age"].plot.kde()
    data["age"].describe()
    # lets bin the age with 10 like 0-10,10-20,..
    # print([10*i for i in range(1,11)])
    data["age"] = pd.cut(data["age"], bins=[10 * i for i in range(1, 10)])

    print(len(data["education"].unique()))
    print(len(data["education-num"].unique()))
    # so if i understand correctly 'education-num' is the labeled values of 'education'
    # so i can drop one lets drop the string one
    data.drop(columns="education", inplace=True)

    data["capital-gain"].plot.kde()
    data["capital-gain"].describe()
    # (max-min)/sd = 13 so binning ins done in 20 catagories
    minmax = MinMaxScaler()
    data.loc[:, "capital-gain"] = minmax.fit_transform(data[["capital-gain"]])
    data["capital-gain"] = pd.cut(data["capital-gain"], bins=20, labels=False)
    data["capital-gain"].plot.kde()

    # capital-loss similar to capital gain
    data["capital-loss"].plot.kde()
    data.loc[:, "capital-loss"] = minmax.fit_transform(data[["capital-loss"]])
    data["capital-loss"].plot.kde()
    data["capital-loss"] = pd.cut(data["capital-loss"], bins=20, labels=False)
    data["capital-loss"].unique()

    data["hours-per-week"].plot.kde()
    data["hours-per-week"].describe()
    # max values of hours per week = 7*24 = 168
    # max value here = 99,menas on avg 14 hours per day
    # so i think safe to divide into 25 bins
    # 0,1,..,24 hours per day (25 levels)
    data["hours-per-week"] = pd.cut(
        data["hours-per-week"], bins=[7 * i for i in range(1, 22)], labels=False
    )
    data["hours-per-week"].plot.kde()
    data["hours-per-week"].unique()
    data["hours-per-week"] = data["hours-per-week"].fillna(0)
    data["hours-per-week"].unique()
    # distribution more or less similar

    # People with similar demographic characteristics should have similar weights.
    data["fnlwgt"].plot.kde()
    data["fnlwgt"].describe()

    # do a minmax scale
    data.loc[:, "fnlwgt"] = minmax.fit_transform(data[["fnlwgt"]])
    data["fnlwgt"].describe()
    # (max-min)/sd=14 so divide into 15 bins

    data["fnlwgt"] = pd.cut(data["fnlwgt"], bins=20, labels=False)
    data["fnlwgt"].plot.kde()

    concols = ["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week"]
    for col in concols:
        print(col, data[col].unique())

    # so all continuous cols are discritized
    data.replace({" <=50K": 0, " >50K": 1}, inplace=True)
    # data
    data.to_csv("data/adult_proc.csv", index=False)


# -------------------------------------------------------------------------------


# -----------------------------Credit Card Dataset Preprocesing------------------
# bin the continuous data
# Freedman–Diaconis' choice to calculate bin size
# choose this because less sensitive than the standard deviation to outliers in data
def freedman_diaconis_rule(data, n):
    min_x = data.min()
    max_x = data.max()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    IQR = q3 - q1
    h = 2 * IQR * np.power(n, -1 / 3)
    n_bins = np.ceil((max_x - min_x) / h)
    return n_bins


def creditcard_preprocess():
    data = pd.read_csv("data/creditcard.csv")  # ,na_values={'TotalCharges': ' '})
    # data
    data.columns
    data.isnull().sum()
    # NO missing values
    data.drop(columns="Time", inplace=True)
    # The time column is irrelivant for classification
    # data
    # all negative samples
    data_neg = data[data["Class"] == 0]
    print("Unique Values in Class of neg ", data_neg["Class"].unique())
    print("Neg Rows = ", len(data_neg.index))

    # all positive samples
    data_pos = data[data["Class"] == 1]
    print("Unique Values in Class of pos ", data_pos["Class"].unique())
    print("Pos Rows = ", len(data_pos.index))
    # sample 20000 negative samples
    data_neg = data_neg.sample(n=20000, replace=False, random_state=1505064)
    # data_neg

    final_data = pd.concat([data_neg, data_pos], ignore_index=True)
    # final_data
    # the colums with continuous values
    cont_cols = final_data.columns[:-1]

    # number of samples for each colums
    n = len(final_data.index)
    for col in cont_cols:
        # get the number of bins
        n_bins = freedman_diaconis_rule(final_data[col], n)

        # cur into bins and replace columns
        final_data[col] = pd.cut(final_data[col], bins=n_bins, labels=False)

        # see if the binning creates finite catagories
        # print(len(final_data[col].unique()))

    final_data.to_csv("data/creditcard_proc.csv", index=False)


def ig_on_split(split, attribute, data, target):

    """
        Information_Gain for a split point
        needed for binarization
    """
    total_entropy = entropy(data[target])

    le_split = data[data[attribute] <= split]
    gt_split = data[data[attribute] > split]

    le_entropy = entropy(le_split[target])
    gt_entropy = entropy(gt_split[target])

    return total_entropy - (le_entropy + gt_entropy)


def binarize_on_ig(attribute, data):
    # last value is the target
    target = data.columns[-1]

    # find sorted uniques in the attribues
    unique = data[attribute].unique()
    unique.sort()

    # find splitting point where information gain maximizes
    n = len(unique)
    max_ig = float("-inf")
    mid_max_ig = -1
    for i in range(0, n - 1):
        # print(unique[i],unique[i + 1])
        mid = (unique[i] + unique[i + 1]) / 2
        # print(mid)
        ig_mid = ig_on_split(mid, attribute, data, target)
        if ig_mid > max_ig:
            max_ig = ig_mid
            mid_max_ig = mid

    binarizer = Binarizer(threshold=mid_max_ig)
    data[[attribute]] = binarizer.fit_transform(data[[attribute]])


def creditcard_preprocess_bin():
    data = pd.read_csv("creditcard.csv")
    data.drop(columns="Time", inplace=True)

    data_neg = data[data["Class"] == 0]
    data_pos = data[data["Class"] == 1]

    # sample 20000 negative samples
    data_neg = data_neg.sample(n=20000, replace=False, random_state=1505064)

    # marge pos and nes
    final_data = pd.concat([data_neg, data_pos], ignore_index=True)

    # the colums with continuous values
    cont_cols = final_data.columns[:-1]
    for col in cont_cols:
        binarize_on_ig(col, final_data)

    final_data.to_csv("creditcard_proc_bin.csv", index=False)


# -------------------------------------------------------------------------------


# ---------------------------Decision Tree---------------------------------------


def entropy(data):
    """
        entropy(data)
        = -sum val in class P(data = val) * log2(P(data = val))
    
    """

    total_count = len(data)
    entropy_val = 0

    _, counts = np.unique(data, return_counts=True)
    for count in counts:
        Prob = count / total_count
        entropy_val += Prob * np.log2(Prob)

    return -entropy_val


def information_gain(attribute, data, target="class"):

    """
        Information_Gain(attribute,data)
        = entropy(data) - entropy(feature,data)
        
        entropy(feature,data) 
        = weighted sum of entropy for every value of fuature
    """

    total_entropy = entropy(data[target])

    total_count = len(data)
    weighted_entropy = 0

    values, counts = np.unique(data[attribute], return_counts=True)
    for value, count in zip(values, counts):
        # data for a specific value
        data_for_value = data.loc[data[attribute] == value][target]
        weighted_entropy += count / total_count * entropy(data_for_value)

    return total_entropy - weighted_entropy


# ID3 Algo
def plurality_value(examples, target):
    """
        input: 
                examples - the example
                target   - the name of target
        output:
                the most occuring target in examples
    """
    values, counts = np.unique(examples[target], return_counts=True)
    max_index = np.argmax(counts)
    return values[max_index]


def decision_tree(examples, attributes, parent_examples, depth):
    # target name is at the last column name of example
    target = examples.columns[-1]

    # cases

    # stopping criteria
    # -----------------------------------------------------
    # 1. examples is empty
    if len(examples) == 0:
        return plurality_value(parent_examples, target)

    # 2. all the example are same
    unique_values = np.unique(examples[target])
    if len(unique_values) <= 1:
        return unique_values[0]

    # 3. attributes is empty or depth limit reached
    if len(attributes) == 0 or depth == 0:
        return plurality_value(examples, target)
    # -----------------------------------------------------

    # 4. grow the tree

    # information gain for each attributes
    importance = [
        information_gain(attribute, examples, target) for attribute in attributes
    ]

    # best information gain
    best_attribute = attributes[np.argmax(importance)]

    # tree with best feature as root
    tree = {best_attribute: {}}

    # attributes_minus_best = attributes - best_attribute
    attributes_minus_best = [a for a in attributes if a != best_attribute]

    # for each value in best_attribute
    for value in np.unique(examples[best_attribute]):

        # split the examples on attrivute values
        split_examples = examples.loc[examples[best_attribute] == value]

        # recursively build the subtree
        subtree = decision_tree(
            split_examples, attributes_minus_best, examples, depth - 1
        )

        # ddd the sub tree under root
        tree[best_attribute][value] = subtree

    # if no value match then default = plurality in the node
    tree[best_attribute]["__default__"] = plurality_value(examples, target)

    return tree


# predict a query
def get_tree_node(query, tree):
    """
        return - matching feature node of query in the tree
    """
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                # if the value does not exist in tree branch
                # that means new data
                # return the default branch
                return tree[key]["__default__"]

            # written again for python scope -_-
            result = tree[key][query[key]]
            return result


def is_leaf(node):
    """
        in this design if node is a dictionary its not a leaf
        if its just a value then its the class type
    """
    return not isinstance(node, dict)


def dt_predict(query, tree):
    """
    input:
        query - in for if a dictionary {feature1: value1,feature2: value2,....}
        tree  - decsion tree in the form of dictionary of dictionaries
        
    output:
        a target class
    """

    # get the corresponding tree node
    node = get_tree_node(query, tree)

    # if the node is leaf just return the result
    if is_leaf(node):
        return node

    # recurse on the node
    return dt_predict(query, node)


def dt_get_pred_true(test_examples, tree):
    # removes last column that contains class
    # converts each row as a dictionary of columns
    queries = test_examples.iloc[:, :-1].to_dict(orient="records")

    y_pred = []
    y_true = []
    for i in range(len(test_examples)):
        prediction = dt_predict(queries[i], tree)
        y_pred.append(prediction)
        y_true.append(test_examples.iloc[i, -1])

    return y_pred, y_true


# -------------------------------------------------------------------------------

# ------------------------------AdaBoost-------------------------------------------------

# adaboost
def resample(examples, w):
    """
        n = |examples|
        resamples n value with replacement from examples
        with probability w
    """
    n = len(examples.index)
    # chooses randon indices based on a probability array
    indices = np.random.choice(range(n), n, p=w)
    return examples.iloc[indices]


def adaboost(examples, algo, n_rounds):
    n = len(examples.index)
    w = np.full(n, 1 / n, dtype="float64")

    # initilize a k size array
    h = [0 for _ in range(n_rounds)]
    z = [0 for _ in range(n_rounds)]
    for k in range(n_rounds):
        # resamples data based on weight
        data = resample(examples, w)

        # attributes are values of column apart from the last one
        attributes = list(examples.columns[:-1])
        h[k] = algo(data, attributes, data, 1)  # train stump->decsion tree with depth 1

        error = 0.0
        for j in range(n):
            query = examples.iloc[j, :-1].to_dict()
            prediction = dt_predict(query, h[k])
            actual_class = examples.iloc[j, -1]
            if prediction != actual_class:
                error += w[j]

        if error > 0.5:
            continue

        for j in range(n):
            query = examples.iloc[j, :-1].to_dict()
            prediction = dt_predict(query, h[k])
            actual_class = examples.iloc[j, -1]
            if prediction == actual_class:
                w[j] *= error / (1 - error)

        # normalize w
        sum_w = sum(w)
        w = [val / sum_w for val in w]

        if error == 0.0:
            z[k] = float("inf")
        else:
            z[k] = np.log2((1 - error) / error)

    return (h, z)


def adaboost_predict(query, h, z):
    # stores the vote for each prediction as {p : v} dictionary
    pred_vote = dict()
    for i in range(len(h)):
        pred = dt_predict(query, h[i])
        if pred in pred_vote:
            pred_vote[pred] += z[i]
        else:
            pred_vote[pred] = z[i]

    # return the key with maximum count
    # ie, prediction with maximum votes
    return max(pred_vote.keys(), key=(lambda k: pred_vote[k]))


def adaboost_get_pred_true(test_examples, h, z):
    # removes last column that contains class
    # converts each row as a dictionary of columns
    queries = test_examples.iloc[:, :-1].to_dict(orient="records")

    y_pred = []
    y_true = []
    for i in range(len(test_examples)):
        prediction = adaboost_predict(queries[i], h, z)
        y_pred.append(prediction)
        y_true.append(test_examples.iloc[i, -1])

    return y_pred, y_true


# -------------------------------------------------------------------------------


# performance measure
def confusion_matrix_2(y_pred, y_true, pos, neg):
    """
    input:
        y_pred - prediction value
        y_true - true value
        pos    - positive attribute
        neg    - negative attribute
        
    output:
        a diftionary with TruePositive,FalsePositive,FalseNegative,TrueNegative
    """
    confusion_matrix = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    for pred, true in zip(y_pred, y_true):
        if pred == pos and true == pos:
            confusion_matrix["TP"] += 1

        elif pred == pos and true == neg:
            confusion_matrix["FP"] += 1

        elif pred == neg and true == pos:
            confusion_matrix["FN"] += 1

        elif pred == neg and true == neg:
            confusion_matrix["TN"] += 1

    return confusion_matrix


# get stat
def get_stat(confusion_matrix):
    TP = confusion_matrix["TP"]
    TN = confusion_matrix["TN"]
    FP = confusion_matrix["FP"]
    FN = confusion_matrix["FN"]

    accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
    print(f"Accuracy\t\t{accuracy}%")

    sensitivity = TP / (TP + FN) * 100
    print(f"Sensitivity\t\t{sensitivity}%")

    specificity = TN / (TN + FP) * 100
    print(f"Specificity\t\t{specificity}%")

    precision = TP / (TP + FP) * 100
    print(f"precision\t\t{precision}%")

    false_discovery_rate = FP / (FP + TP) * 100
    print(f"False Discovery Rate\t{false_discovery_rate}%")

    f1_score = (2 * TP) / (2 * TP + FP + FN) * 100
    print(f"F1 score\t\t{f1_score}%")


def show_dt_stat(examples, train_examples, test_examples_list, pos=1, neg=0):
    # Parse parameter from example
    attributes = list(examples.columns)[:-1]
    depth = len(attributes)

    # Train decision tree
    print('Running Decision Tree')
    print("****Start Training****")
    tree = decision_tree(examples, attributes, train_examples, depth)
    print("****Training Done****")
    # pprint(tree)

    # for each specified test examples show stat
    for test_examples in test_examples_list:
        # Get prediction
        print("\n\n*******Testing*******")
        y_pred, y_true = dt_get_pred_true(test_examples, tree)

        # get confusion_matrix
        print("*****Testing Done*****\n")
        print("*********Stat*********")
        get_stat(confusion_matrix_2(y_pred, y_true, pos, neg))
        print("\n")


def show_ab_stat(train_examples, algo, test_examples_list, n_round_list, pos=1, neg=0):
    # for each specified round generate stat
    print('Running Decision Tree')
    for n_round in n_round_list:
        print(f'---k={n_round}---')
        # Train adaboost
        print("******Training*******")
        h, z = adaboost(train_examples, algo, n_round)
        print("****Training Done****")
        # print(h,z)

        # for each specified test examples show stat
        for test_examples in test_examples_list:
            # Get prediction
            print("\n\n*******Testing*******")
            y_pred, y_true = adaboost_get_pred_true(test_examples, h, z)

            # get confusion_matrix
            print("*****Testing Done*****\n")
            print("*********Stat*********")
            print(f"----K = {n_round}----")
            get_stat(confusion_matrix_2(y_pred, y_true, pos, neg))


# -----------------------------MAIN--------------------------------------------------


def main():
    # Load Example
    # Dataset 1
    #print('Running churn')
    #examples = pd.read_csv('data/churn_proc.csv')

    # Dataset 2
    #print('Running adult')
    #examples = pd.read_csv('data/adult_proc.csv')

    # Dataset 3
    print('Running creditcard_bin')
    examples = pd.read_csv("data/creditcard_proc_bin.csv")

    # Train-Test Split
    train_examples, test_examples = train_test_split(
        examples, test_size=0.2, random_state=1
    )

    # show Dicsion Tree Result
    show_dt_stat(examples, train_examples, [train_examples, test_examples])

    # show Dicsion Tree with AdaBoost Result
    show_ab_stat(
        train_examples, decision_tree, [train_examples, test_examples], [5, 10, 15, 20]
    )

    # play a bell sound at the and :3
    PlaySound("bell.wav", False)


from time import time 

if __name__ == "__main__":
    start_time = time()
    main()
    del_t = (time() - start_time)/60.0
    print(f'\n\nRuntime: {del_t} minutes')
