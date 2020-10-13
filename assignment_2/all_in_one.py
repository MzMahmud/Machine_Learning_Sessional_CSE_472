import os
import re
import string
import numpy as np
from scipy import stats

import nltk
from nltk.tokenize import word_tokenize

# nltk.download('punkt')
from nltk.corpus import stopwords

# nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer

# nltk.download('wordnet')
from nltk.stem import PorterStemmer

from bs4 import BeautifulSoup

from collections import Counter


# file: >>>>>>>> pre_processing


def get_pre_processed(text):
    #     print("\n----RAW----\n",text)

    # remove html tags
    text = BeautifulSoup(text, features="lxml").text
    #     print("\n\n----html tags removed----\n",text)

    # Lowercase the text
    text = text.lower()
    #     print("\n===After Lowercase:===\n", text)

    # Removing Numbers
    text = re.sub(r"[-+]?\d+\b", "", text)
    #     print("\n===After Removing Numbers:===\n", text)

    # Remove punctuations
    text = text.translate((str.maketrans("", "", string.punctuation)))
    #     print("\n===After Removing Punctuations:===\n", text)

    # Tokenize
    text = word_tokenize(text)
    #     print("\n===After Tokenizing:===\n", text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    text = [word for word in text if not word in stop_words]
    #     print("\n===After Stopword Removal:===\n", text)

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]
    #     print("\n===After Lemmatization:===\n", text)

    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    #     print("\n===After Stemming:===\n", text)

    return text


def store_train_validation_test(DATA_DIR, topics, n_train, n_validation, n_test, n_itr):
    train = open("train.in", "w", encoding="utf16")
    validation = open("validation.in", "w", encoding="utf16")

    test_iters = [
        open(f"test_itr_{itr}.in", "w", encoding="utf16") for itr in range(n_itr)
    ]
    # number of document of each class per iteration
    docs_per_itr = n_test // n_itr

    for topic in topics:
        xml_file = os.path.join(DATA_DIR, f"{topic}.xml")
        file = open(xml_file, "rb")
        content = file.read()
        soup = BeautifulSoup(content, "xml")

        num = 1
        for items in soup.findAll("row"):
            body = items.attrs["Body"]
            if len(body) == 0:
                continue
                # print("empty",items)

            text = get_pre_processed(body)
            text.append(topic)
            doc = " ".join(text)
            if num <= n_train:
                # train.write(f"------{items.attrs['Id']}------\n")
                print(doc, file=train)
            elif n_train < num and num <= (n_train + n_validation):
                # validation.write(f"------{items.attrs['Id']}------\n")
                print(doc, file=validation)
            elif (n_train + n_validation) < num and num <= (
                n_train + n_validation + n_test
            ):
                # test.write(f"------{items.attrs['Id']}------\n")
                itr = (num - (n_train + n_validation) - 1) // docs_per_itr
                print(doc, file=test_iters[itr])
            else:
                break
            num += 1
        file.close()

    train.close()
    validation.close()
    for test_iter in test_iters:
        test_iter.close()


def pre_processing():
    # read the topics' name
    with open(topics_txt, "r") as f:
        topics = [topic.strip() for topic in f.readlines()]
    print("topics", topics)

    store_train_validation_test(DATA_DIR, topics, n_train, n_validation, n_test, n_itr)


# file: <<<<<<<< pre_processing

# file: >>>>>>>> kNN


def hamming_distance(p1: Counter, p2: Counter) -> int:
    hd = 0
    for xi in p1 | p2:  # xi in (p1 U p2)
        if (p1[xi] == 0) or (p2[xi] == 0):
            # print(xi)
            # as xi in union and not in any one of it its a mismatch
            hd += 1
    return hd


def euclidean_distance(p1: Counter, p2: Counter):
    ed = 0
    for xi in p1 | p2:  # xi in (p1 U p2)
        # print(f"({p1[xi]}-{p2[xi]})**2",end='+')
        ed += (p1[xi] - p2[xi]) ** 2
    # print()
    # return ed before sqrt: for debug
    return np.sqrt(ed)


def get_tf_format(document):
    X = dict()
    Wd = len(document)  # Word count of data
    for w, Nw in Counter(document).items():
        # print(f"TF {w} = {Nw}/{Wd}")
        X[w] = Nw / Wd  # TF = Nw/Wd
    return X


def get_tf_idf_format(doc, docs):
    eps = 0.000001
    n_docs = len(docs)  # number of documents

    doc_tf_idf = get_tf_format(doc)
    to_ignore = []
    for w in doc_tf_idf:
        Cw = 0
        for doc_i in docs:
            if w in doc_i:
                Cw += 1
        if Cw == 0:
            to_ignore.append(w)
            # new word -> ignore
            # print(f"IDF {w} => new word; ignore")
        else:
            # print(f"IDF {w} = log({n_docs}/1+{Cw})")
            idf = (
                eps if Cw == n_docs else np.log10(n_docs / Cw)
            )  # lecture note: np.log10(n_docs/(1 + Cw))
            doc_tf_idf[w] *= idf

    for w in to_ignore:
        doc_tf_idf.pop(w)

    return doc_tf_idf


def norm(a):
    n = 0.0
    for ai in a.values():
        n += ai * ai
    return np.sqrt(n)


def cosine_similarity(p1, p2):
    xi_both = set(p1).intersection(set(p2))
    dot = 0.0
    for xi in xi_both:
        dot += p1[xi] * p2[xi]
    return dot / (norm(p1) * norm(p2))


def kNN_predict(X_train, Y_train, X_test, k=3, distance_function=cosine_similarity):
    neighbors_class_dist = []
    for data, cls in zip(X_train, Y_train):
        dist = distance_function(X_test, data)
        neighbors_class_dist.append((cls, dist))
    # print("unsorted",neighbors_class_dist)

    neighbors_class_dist.sort(
        key=lambda class_dist: class_dist[1],
        reverse=(distance_function == cosine_similarity)
        # if dist function = cos similarity we have to sort in decending order
    )
    # print("sorted",neighbors_class_dist)

    kNN_class_dist = neighbors_class_dist[:k]
    # print("kNN_class_dist",kNN_class_dist)

    votes = dict()
    for cls, dist in kNN_class_dist:
        # unweighted voting
        votes[cls] = votes.get(cls, 0) + 1
    # print(votes)

    max_vote_class = max(votes, key=lambda cls: votes[cls])
    # print("max_vote_class",max_vote_class)

    return max_vote_class


def kNN_performance_evaluation(
    X_train, Y_train, X_test, Y_test, k_vals, distance_function
):
    # get the proper input format for distance function
    if distance_function == cosine_similarity:
        X_train_tf_idf = [get_tf_idf_format(doc, X_train) for doc in X_train]
        X_test = [get_tf_idf_format(doc, X_train) for doc in X_test]
        X_train = X_train_tf_idf
    else:
        X_train = [Counter(data) for data in X_train]
        X_test = [Counter(data) for data in X_test]
    # print(X_train,"\n",X_test)

    stat = []
    for k in k_vals:
        print(f"---k={k}---{distance_function.__name__}---")
        total, correct, cur = len(X_test), 0, 0
        interval = max(total // 5, 1)
        for doc, actual_class in zip(X_test, Y_test):
            prediction = kNN_predict(X_train, Y_train, doc, k, distance_function)
            if prediction == actual_class:
                correct += 1
            cur += 1
            if cur % interval == 0:
                print(f"Completed: {cur*100/total:.1f}%")
        print("--------------------")
        print(f"Correct : {correct}")
        print(f"Total   : {total}")
        print(f"Accuracy: {(correct*100)/(total):.2f}%")
        print("--------------------")
        stat.append(correct * 100 / total)
    return stat


def get_X_Y_from(file):
    with open(file, "r", encoding="utf16") as f:
        docs = [line.split() for line in f.readlines()]
    X = [doc[:-1] for doc in docs]
    Y = [doc[-1] for doc in docs]
    return X, Y


def kNN_validation():
    # generate stat in markdown for kNN
    # create a markdown file
    with open("kNN_stat.md", "w") as out:
        print("# k Nearest Neighbor (kNN)", file=out)
        print("\n## Topics", file=out)
        for topic in set(Y_train):
            print(f"- {topic}", file=out)

        print(f"\n## Training Data Size\n- **{len(Y_train)}** documents", file=out)
        print(
            f"\n## Validation Data Size\n- **{len(Y_validation)}** documents", file=out
        )
        print("\n## Methodologies and k Matrix", file=out)

        print("|   ", end=" |", file=out)
        for k in k_vals:
            print(f" {k} ", end="|", file=out)
        print("\n|", " --- |" * (len(k_vals) + 1), file=out)

        for distance_function in functions:
            # performance evaluation return accuracy for each k values
            accuracy_vals = kNN_performance_evaluation(
                X_train, Y_train, X_validation, Y_validation, k_vals, distance_function
            )

            row = f"| {distance_function.__name__} | "
            for accuracy in accuracy_vals:
                row += f"{accuracy:.2f}% | "
            print(row, file=out)


def get_accuracy_best_kNN():
    kNN_test_itr_accuracy = []
    for itr in range(n_iter):
        print(f"---Test---Iteration {itr + 1}---")
        input_file = f"test_itr_{itr}.in"
        X_test, Y_test = get_X_Y_from(input_file)
        accuracy_vals = kNN_performance_evaluation(
            X_train, Y_train, X_test, Y_test, [best_k], best_dist_func
        )
        kNN_test_itr_accuracy.append(accuracy_vals[0])
    return kNN_test_itr_accuracy


# file: <<<<<<<< kNN

# file: >>>>>>>> naive_bayes_document_classification


class ClassWordCounter:
    def __init__(self, n_docs, n_words, word_counter):
        self.n_docs = n_docs
        self.n_words = n_words
        self.word_counter = word_counter

    def __str__(self):
        return f"n_docs: {self.n_docs}\nn_words: {self.n_words}\nword_counter: {self.word_counter}"

    def __repr__(self):
        return f"ClassWordCounter({self.n_docs},{self.n_words},{self.word_counter})"


def get_class_word_counters(X_train, Y_train) -> ClassWordCounter:
    class_word_counters = dict()
    for cls, doc in zip(Y_train, X_train):
        if cls not in class_word_counters:
            class_word_counters[cls] = ClassWordCounter(0, 0, Counter())
        class_word_counters[cls].n_docs += 1
        class_word_counters[cls].n_words += len(doc)
        class_word_counters[cls].word_counter += Counter(doc)
    # print(class_word_counters)
    return class_word_counters


def naive_bayes_predict(document, class_word_counters, n_docs, n_unique_words, alpha):
    probabilities = dict()
    for word in document:
        for cls, class_word_counter in class_word_counters.items():
            if cls not in probabilities:
                # print(f"prior {cls}: P({cls}) = {class_word_counter.n_docs}/{n_docs}")
                # prior: P(cls)
                probabilities[cls] = class_word_counter.n_docs / n_docs

            n_word_cls = class_word_counter.word_counter[word]
            n_cls = class_word_counter.n_words

            # zero problem
            # print(f"P({word}|{cls}) *= {n_word_cls}/{n_cls}")

            # with smoothin factor
            # print(f"P({word}|{cls}) *= ({n_word_cls} + {alpha})/({n_cls} + {alpha}*{n_unique_words})")
            probabilities[cls] *= (n_word_cls + alpha) / (
                n_cls + (alpha * n_unique_words)
            )

    # print("Probabilities: ",probabilities)
    prediction = max(probabilities, key=lambda key: probabilities[key])
    return prediction


def NB_performance_evaluation(X_train, Y_train, X_test, Y_test, alpha_vals):
    # get the proper input format
    class_word_counters = get_class_word_counters(X_train, Y_train)
    n_docs = len(X_train)
    unique_words = {word for doc in X_train for word in doc}
    n_unique_words = len(unique_words)

    stat = []
    for alpha in alpha_vals:
        print(f"---alpha = {alpha:.2f}------")
        total, correct, cur = len(X_test), 0, 0
        interval = max(total // 5, 1)
        for doc, actual_class in zip(X_test, Y_test):
            prediction = naive_bayes_predict(
                doc, class_word_counters, n_docs, n_unique_words, alpha
            )

            if prediction == actual_class:
                correct += 1

            cur += 1
            if cur % interval == 0:
                print(f"Completed: {cur*100/total:.1f}%")

        print("--------------------")
        print(f"Correct : {correct}")
        print(f"Total   : {total}")
        print(f"Accuracy: {(correct*100)/(total):.2f}%")
        print("--------------------")
        stat.append(correct * 100 / total)
    return stat


def naive_bayes_validation():
    # get performance
    stats = NB_performance_evaluation(
        X_train, Y_train, X_validation, Y_validation, alpha_vals
    )

    # generate stat in markdown for NB
    with open("NB_stat.md", "w") as out:
        print("# Naive Bayes", file=out)
        print("\n## Topics", file=out)
        for topic in set(Y_train):
            print(f"- {topic}", file=out)

        print(f"\n## Training Data Size\n- **{len(Y_train)}** documents", file=out)
        print(
            f"\n## Validation Data Size\n- **{len(Y_validation)}** documents", file=out
        )

        print("\n## Accuracy for Different Smoothing Factors ($\\alpha$)\n", file=out)
        print("| Serial | alpha | Accuracy |", file=out)
        print("| --- | --- | --- |", file=out)

        for i, alpha, accuracy in zip(range(len(alpha_vals)), alpha_vals, stats):
            print(f"| {i+1} | {alpha:.2f} | {accuracy:.2f}% |", file=out)


def get_accuracy_best_NB():
    NB_test_itr_accuracy = []
    for itr in range(n_iter):
        print(f"---Test---Iteration {itr + 1}---")

        input_file = f"test_itr_{itr}.in"
        X_test, Y_test = get_X_Y_from(input_file)

        accuracy_vals = NB_performance_evaluation(
            X_train, Y_train, X_test, Y_test, [best_alpha]
        )
        NB_test_itr_accuracy.append(accuracy_vals[0])
    return NB_test_itr_accuracy


# file: <<<<<<<< naive_bayes_document_classification

# file: >>>>>>>> t_test


def compare_with_t_test():
    t_stat, p_value = stats.ttest_rel(kNN_test_itr_accuracy, NB_test_itr_accuracy)
    print(f"stat {t_stat} p_value {p_value}")

    # Markdown Iteration
    with open("t_test.md", "w") as out:
        print("\n# kNN vs. NB Accuracy\n", file=out)
        print("| Serial | kNN   | NB    |", file=out)
        print("| :---:  | :---: | :---: |", file=out)
        i = 1
        for kNN_a, NB_a in zip(kNN_test_itr_accuracy, NB_test_itr_accuracy):
            print(f"| {i} | {kNN_a:.2f}% | {NB_a:.2f}% |", file=out)
            i += 1

        print("\n# T-test\n", file=out)
        print(
            "| Significance Level  | t Statistics | t Critical Value | Result |",
            file=out,
        )
        print("| :---:  | :---: | :---: | :---: |", file=out)

        for sig in significances:
            p = 1 - sig
            df = n_iter - 1
            t_critical = stats.t.ppf(p, df)
            t_critical = -t_critical if t_stat < 0 else t_critical

            if t_stat > 0:
                result = "**kNN** better" if t_stat < t_critical else "No Difference"
            else:
                result = "**NB** better" if t_stat < t_critical else "No Difference"
            print(
                f"| {sig:.3f} | {t_stat:.6f} | {t_critical:.6f} | {result} |", file=out
            )


# file: <<<<<<<< t_test


# Parameters: Data Path
DATA_DIR = os.path.join(os.path.join(os.getcwd(), "Data"), "Training")
# print(DATA_DIR)

topics_txt = os.path.join(os.path.join(os.getcwd(), "Data"), "topics.txt")
# print(topics_txt)

# Parameters: Preprocessing
n_train = 50
n_validation = 20
n_itr = 50
n_test = n_itr * 10

# Run: pre processing
# pre_processing()


# Parameters: input validation paths
train_input_file = os.path.join(os.getcwd(), "train.in")
validation_input_file = os.path.join(os.getcwd(), "validation.in")

# get data
X_train, Y_train = get_X_Y_from(train_input_file)
# print("input",len(X_train), len(Y_train))

X_validation, Y_validation = get_X_Y_from(validation_input_file)
# print("validation", len(X_validation), len(Y_validation))

# hyper parameters
k_vals = [1, 3, 5]
functions = [hamming_distance, euclidean_distance, cosine_similarity]

# Run: kNN Validation
# kNN_validation()

# Parameters: best kNN
best_k = 5
best_dist_func = hamming_distance  # testing ,NOT FINAL
n_iter = 50

# Run: test the best performing kNN
kNN_test_itr_accuracy = get_accuracy_best_kNN()
print("kNN_test_itr_accuracy", kNN_test_itr_accuracy)

# Parameters: NB alphas
# hyper parameters
alpha_vals = np.linspace(0.1, 1.0, num=10)
# print("alpha_vals", alpha_vals)

# Run: NB validation
# naive_bayes_validation()

# Parameters: best NB alpha
best_alpha = 0.5  # testing not final

# Run: best NB
NB_test_itr_accuracy = get_accuracy_best_NB()
print("NB_test_itr_accuracy", NB_test_itr_accuracy)

# Parameters: significances
significances = [0.005, 0.01, 0.05]

# Run: T-Test
compare_with_t_test()
