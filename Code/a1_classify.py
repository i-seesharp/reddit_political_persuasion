#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import argparse
import os
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# set the random state for reproducibility 
import numpy as np
np.random.seed(401)

def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    row, col = C.shape
    total = 0
    correct = 0
    for i in range(row):
        for j in range(col):
            total = total + C[i,j]
            if i==j:
                correct = correct + C[i,j]
    return float(correct)/float(total)

def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    recall_vals = []
    row, col = C.shape
    for i in range(row):
        correct_classified = 0
        total_classified = 0
        for j in range(col):
            if i==j:
                correct_classified = correct_classified + C[i,j]
            total_classified = total_classified + C[i,j]
        recall_vals.append(float(correct_classified)/float(total_classified))
    return recall_vals


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    precision_vals = []
    row, col = C.shape
    for j in range(col):
        correct_predicted = 0.0
        total_predicted = 0.0
        for i in range(row):
            if i==j:
                correct_predicted = correct_predicted + C[i,j]
            total_predicted = total_predicted + C[i,j]
        precision_vals.append(float(correct_predicted)/float(total_predicted))
    return precision_vals

name_to_idx = {
        "SGDClassifier" : 0,
        "GaussianNB": 1,
        "RandomForestClassifier": 2,
        "MLPClassifier": 3,
        "AdaBoostClassifier": 4}
idx_to_name = {
    0: "SGDClassifier",
    1: "GaussianNB",
    2: "RandomForestClassifier",
    3: "MLPClassifier",
    4: "AdaBoostClassifier"
}
classifiers = {
        "SGDClassifier" : SGDClassifier(),
        "GaussianNB": GaussianNB(),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=10, max_depth=5),
        "MLPClassifier": MLPClassifier(alpha=0.05),
        "AdaBoostClassifier": AdaBoostClassifier()
    }

def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:      
       i: int, the index of the supposed best classifier
    '''
    results = {}
    for name in classifiers:
        classifier = classifiers[name]
        pipeline = make_pipeline(StandardScaler(), classifier)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        conf_matrix = confusion_matrix(y_test, y_pred)
        result = {
            "accuracy": accuracy(conf_matrix),
            "recall": recall(conf_matrix),
            "precision": precision(conf_matrix),
            "C": conf_matrix
        }
        results[name] = result
    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        # For each classifier, compute results and write the following output:
        #     outf.write(f'Results for {classifier_name}:\n')  # Classifier name
        #     outf.write(f'\tAccuracy: {accuracy:.4f}\n')
        #     outf.write(f'\tRecall: {[round(item, 4) for item in recall]}\n')
        #     outf.write(f'\tPrecision: {[round(item, 4) for item in precision]}\n')
        #     outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')
        for name in results:
            result = results[name]
            outf.write(f'Results for {name}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {result["accuracy"]:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in result["recall"]]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in result["precision"]]}\n')
            outf.write(f'\tConfusion Matrix: \n{result["C"]}\n\n')
  
    i = 0
    for name in classifiers:
        name_to_idx[name] = i
        idx_to_name[i] = name
        i = i + 1
    iBest = 0
    for name in results:
        curr_accuracy = results[name]["accuracy"]
        best_accuracy = results[idx_to_name[iBest]]["accuracy"]
        if(curr_accuracy > best_accuracy):
            iBest = name_to_idx[name]
    return iBest

def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    X_1k, y_1k, _, _ = train_test_split(X_train, y_train, train_size = 1000)
    X_5k, y_5k, _, _ = train_test_split(X_train, y_train, train_size = 5000)
    X_10k, y_10k, _, _ = train_test_split(X_train, y_train, train_size = 10000)
    X_15k, y_15k, _, _ = train_test_split(X_train, y_train, train_size = 15000)
    X_20k, y_20k, _, _ = train_test_split(X_train, y_train, train_size = 20000)

    best_classifier = classifiers[idx_to_name[iBest]]


    
    trains = {
        "1k": (X_1k, y_1k),
        "5k": (X_5k, y_5k),
        "10k": (X_10k, y_10k),
        "15k": (X_15k, y_15k),
        "20k": (X_20k, y_20k)
    }
    accuracies = {}
    for size in trains:
        X_t, y_t = trains[size]
        pipeline = make_pipeline(StandardScaler(), best_classifier)
        pipeline.fit(X_t, y_t)
        y_pred = pipeline.predict(X_test)
        accuracies[size] = accuracy(confusion_matrix(y_test, y_pred))
    
    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # For each number of training examples, compute results and write
        # the following output:
        #     outf.write(f'{num_train}: {accuracy:.4f}\n'))
        for size in accuracies:
            outf.write(f'{size}: {accuracies[size]:.4f}\n')

    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    possible = [5, 50]
    selector5 = SelectKBest(f_classif, k=5)
    selector50 = SelectKBest(f_classif, k=50)
    X_train_copy, X_train_copy2 = X_train.copy(), X_train.copy()
    y_train_copy, y_train_copy2 = y_train.copy(), y_train.copy()

    X_5t, y_5t = selector5.fit_transform(X_train_copy, y_train_copy)
    X_50t, y_50t = selector50.fit_transform(X_train_copy2, y_train_copy2)
    pp = {}
    #Store the corresponding p-values in a HashMap
    pp[5] = selector5.pvalues_
    pp[50] = selector50.pvalues_

    best_classifier = classifiers[idx_to_name[i]]
    

    trains = {
        "1k": selector5.transform(X_1k.copy(), y_1k.copy()),
        "32k": selector5.transform(X_train.copy(), y_train.copy())
    }
    accuracies = {}

    for size in trains:
        X_t, y_t = trains[size]
        pipeline = make_pipeline(StandardScaler(), best_classifier)
        pipeline.fit(X_t, y_t)
        y_pred = pipeline.predict(X_test)
        accuracies[size] = accuracy(confusion_matrix(y_test, y_pred))
    
    #New selector for determining best features in 1k and best features in 32k
    selector1k_5 = SelectKBest(f_classif, k=5)
    selector32k_5 = SelectKBest(f_classif, k=5)

    selector1k_5.fit(X_1k.copy(), y_1k.copy())
    selector32k_5.fit(X_train.copy(), y_train.copy())

    p1k_values = selector1k_5.pvalues_
    p32k_values = selector32k_5.pvalues_

    #Assuming p1k_values and p32k_values contain p_values of all the 173 features
    indices_1k = set(np.argpartition(p1k_values, 5)[:5])
    indices_32k = set(np.argpartition(p32k_values, 5)[:5])

    feature_intersection = list(indices_1k.intersection(indices_32k))


    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.
        
        # for each number of features k_feat, write the p-values for
        # that number of features:
            # outf.write(f'{k_feat} p-values: {[round(pval, 4) for pval in p_values]}\n')
        for k in pp:
            p_values = pp[k]
            outf.write(f'{k} p-values: {[format(pval) for pval in p_values]}\n')
        
        outf.write(f'Accuracy for 1k: {accuracies["1k"]:.4f}\n')
        outf.write(f'Accuracy for full dataset: {accuracies["32k"]:.4f}\n')
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        outf.write(f'Top-5 at higher: {indices_32k}\n')


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  

        '''
    
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    kf = KFold(n_splits=5, shuffle=True)
    accuracies = {}

    for name in classifiers:
        accuracies[name] = []

    for train_index, test_index in kf.split(X):
        X_tr, X_ts = X[train_index], X[test_index]
        y_tr, y_ts = y[train_index], y[test_index]

        for name in classifiers:
            classifier = classifiers[name]
            pipeline = make_pipeline(StandardScaler(), classifier)
            pipeline.fit(X_tr, y_tr)

            y_pred = pipeline.predict(X_ts)
            accuracies[name].append(accuracy(confusion_matrix(y_ts, y_pred)))
    
    fold_accuracies = []
    for fold in range(5):
        d = {}
        for name in classifiers:
            d[name] = accuracies[name][fold]
        fold_accuracies.append(d)
    
    best_classifier_name = idx_to_name[i]
    best_vector = accuracies[best_classifier_name]
    p_values = []
    for index in range(len(classifiers)): #5 total classifiers
        if i==index:
            continue
        val = ttest_rel(best_vector, accuracies[idx_to_name[index]])
        p_values.append(val)


    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        # for each fold:
        #     outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
        # outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')
        for fold in range(5):
            kfold_accuracies = []
            for index in range(len(classifiers)):
                kfold_accuracies.append(fold_accuracies[fold][idx_to_name[index]])
            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
        outf.write(f'p-values: {[format(pval) for pval in p_values]}\n')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()
    
    # TODO: load data and split into train and test.
    npz_data = np.load(args.input)
    data = npz_data['arr_0']
    #print(data.shape)
    X = data[:, :173] #Retreive the 173 features for each instance
    y = data[:, -1] #Obtain the label value for each instance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # TODO : complete each classification experiment, in sequence.
    # Perform experiment 3.1
    class31(args.output_dir, X_train, X_test, y_train, y_test)
