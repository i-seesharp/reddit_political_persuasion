#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import numpy as np
import argparse
import json
import string
from csv import DictReader as reader
from os import path as dir_plus_file

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}

def bga_helper():
    path = "/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv"
    bgl_file = open(path, "r")
    statistics = {}
    for data in reader(bgl_file):
        statistics[data["WORD"]] = data
    bgl_file.close()
    return statistics

bgl_metrics = bga_helper()

def wa_helper():
    path = "/u/cs401/Wordlists/Ratings_Warriner_et_al.csv"
    wa_file = open(path, "r")
    statistics = {}
    for data in reader(wa_file):
        statistics[data["Word"]] = data
    wa_file.close()
    return statistics

wa_metrics = wa_helper()

def first_person_pronouns(worded_sentences):
    s = 0
    for words in worded_sentences:
        for lst in words:
            if lst[0].lower() in FIRST_PERSON_PRONOUNS:
                s = s + 1
    return s


def second_person_pronouns(worded_sentences):
    s = 0
    for words in worded_sentences:
        for lst in words:
            if lst[0].lower() in SECOND_PERSON_PRONOUNS:
                s = s + 1
    return s


def third_person_pronouns(worded_sentences):
    s = 0
    for words in worded_sentences:
        for lst in words:
            if lst[0].lower() in THIRD_PERSON_PRONOUNS:
                s = s + 1
    return s


def coordinating_conjunctions(worded_sentences):
    s = 0
    tags = ["CC"]
    possible = set(t.lower() for t in tags)
    for words in worded_sentences:
        for lst in words:
            if lst[1] and lst[1].lower() in possible:
                s = s + 1
    return s


def past_tense_verbs(worded_sentences):
    s = 0
    tags = ["VBD"]
    possible = set(t.lower() for t in tags)
    for words in worded_sentences:
        for lst in words:
            if lst[1] and lst[1].lower() in possible:
                s = s + 1
    return s


def future_tense_verbs(worded_sentences):
    s = 0
    possible = ["'ll", "will", "gonna"]
    for words in worded_sentences:
        for i in range(len(words)):
            word, tag = words[i]
            if word.lower() in possible:
                s = s + 1
            if tag == "VB" and i > 0:
                new_word, new_tag = words[i-1]
                if new_word == "going-to":  # Since we hyphenated going-to in preprocessing
                    s = s + 1
    return s


def num_commas(worded_sentences):
    s = 0
    tags = [","]
    possible = set(t.lower() for t in tags)
    for words in worded_sentences:
        for lst in words:
            if lst[1] and lst[1].lower() in possible:
                s = s + 1
    return s


def num_mulchar_punc(worded_sentences):
    s = 0
    for words in worded_sentences:
        for lst in words:
            if(len(lst[0]) > 1):
                x = True
                for c in lst[0]:
                    x = x and c in set(string.punctuation)
                if x:
                    s = s + 1
    return s


def num_common_nouns(worded_sentences):
    s = 0
    tags = ["NN", "NNS"]
    possible = set(t.lower() for t in tags)
    for words in worded_sentences:
        for lst in words:
            if lst[1] and lst[1].lower() in possible:
                s = s + 1
    return s


def num_proper_nouns(worded_sentences):
    s = 0
    tags = ["NNP", "NNPS"]
    possible = set(t.lower() for t in tags)
    for words in worded_sentences:
        for lst in words:
            if lst[1] and lst[1].lower() in possible:
                s = s + 1
    return s


def num_adverbs(worded_sentences):
    s = 0
    tags = ["RB", "RBR", "RBS"]
    possible = set(t.lower() for t in tags)
    for words in worded_sentences:
        for lst in words:
            if lst[1] and lst[1].lower() in possible:
                s = s + 1
    return s


def num_wh_words(worded_sentences):
    s = 0
    tags = ["WDT", "WP", "WP$", "WRB"]
    possible = set(t.lower() for t in tags)
    for words in worded_sentences:
        for lst in words:
            if lst[1] and lst[1].lower() in possible:
                s = s + 1
    return s


def num_slang(worded_sentences):
    s = 0
    for words in worded_sentences:
        for lst in words:
            if lst[0].lower() in SLANG:
                s = s + 1
    return s


def avg_length_sentences(worded_sentences):
    avg = 0
    for words in worded_sentences:
        for lst in words:
            avg = avg + len(lst[0])
    if len(worded_sentences) > 0:
        return avg/float(len(worded_sentences))
    return 0


def avg_length_tokens(worded_sentences):
    s = 0
    count = 0
    for words in worded_sentences:
        for lst in words:
            x = False
            for c in lst[0]:
                x = x or (c not in set(string.punctuation))
            if x:
                s = s + len(lst[0])
                count = count + 1
    if count == 0:
        return count
    return s/float(count)


def num_sentences(worded_sentences):
    return len(worded_sentences)


def average_aoa(worded_sentences):
    average = []
    query = "AoA (100-700)"

    values = bgl_metrics
    for words in worded_sentences:
        for lst in words:
            try:
                average.append(float(values[lst[0].lower()][query]))
            except:
                pass
    if len(average) > 0:
        return sum(average)/float(len(average))
    return 0


def average_img(worded_sentences):
    average = []
    query = "IMG"

    values = bgl_metrics
    for words in worded_sentences:
        for lst in words:
            try:
                average.append(float(values[lst[0].lower()][query]))
            except:
                pass
    if len(average) > 0:
        return sum(average)/float(len(average))
    return 0


def average_fam(worded_sentences):
    average = []
    query = "FAM"

    values = bgl_metrics
    for words in worded_sentences:
        for lst in words:
            try:
                average.append(float(values[lst[0].lower()][query]))
            except:
                pass
    if len(average) > 0:
        return sum(average)/float(len(average))
    return 0


def std_dev_aoa(worded_sentences):
    metrics = []
    query = "AoA (100-700)"

    values = bgl_metrics
    for words in worded_sentences:
        for lst in words:
            try:
                metrics.append(float(values[lst[0].lower()][query]))
            except:
                pass
    if len(metrics) > 0:
        return np.std(metrics)
    return 0


def std_dev_img(worded_sentences):
    metrics = []
    query = "IMG"

    values = bgl_metrics
    for words in worded_sentences:
        for lst in words:
            try:
                metrics.append(float(values[lst[0].lower()][query]))
            except:
                pass
    if len(metrics) > 0:
        return np.std(metrics)
    return 0


def std_dev_fam(worded_sentences):
    metrics = []
    query = "FAM"

    values = bgl_metrics
    for words in worded_sentences:
        for lst in words:
            try:
                metrics.append(float(values[lst[0].lower()][query]))
            except:
                pass
    if len(metrics) > 0:
        return np.std(metrics)
    return 0


def average_vmeansum(worded_sentences):
    average = []
    query = "V.Mean.Sum"
   
    values = wa_metrics
    for words in worded_sentences:
        for lst in words:
            try:
                average.append(float(values[lst[0].lower()][query]))
            except:
                pass
    if len(average) > 0:
        return sum(average)/float(len(average))
    return 0


def average_ameansum(worded_sentences):
    average = []
    query = "A.Mean.Sum"

    values = wa_metrics
    for words in worded_sentences:
        for lst in words:
            try:
                average.append(float(values[lst[0].lower()][query]))
            except:
                pass
    if len(average) > 0:
        return sum(average)/float(len(average))
    return 0


def average_dmeansum(worded_sentences):
    average = []
    query = "D.Mean.Sum"

    values = wa_metrics
    for words in worded_sentences:
        for lst in words:
            try:
                average.append(float(values[lst[0].lower()][query]))
            except:
                pass
    if len(average) > 0:
        return sum(average)/float(len(average))
    return 0


def std_dev_vmeansum(worded_sentences):
    metrics = []
    query = "V.Mean.Sum"

    values = wa_metrics
    for words in worded_sentences:
        for lst in words:
            try:
                metrics.append(float(values[lst[0].lower()][query]))
            except:
                pass
    if len(metrics) > 0:
        return np.std(metrics)
    return 0

def std_dev_ameansum(worded_sentences):
    metrics = []
    query = "A.Mean.Sum"

    values = wa_metrics
    for words in worded_sentences:
        for lst in words:
            try:
                metrics.append(float(values[lst[0].lower()][query]))
            except:
                pass
    if len(metrics) > 0:
        return np.std(metrics)
    return 0

def std_dev_dmeansum(worded_sentences):
    metrics = []
    query = "D.Mean.Sum"

    values = wa_metrics
    for words in worded_sentences:
        for lst in words:
            try:
                metrics.append(float(values[lst[0].lower()][query]))
            except:
                pass
    if len(metrics) > 0:
        return np.std(metrics)
    return 0

def num_uppercase_tokens(worded_sentences):
    s = 0
    for words in worded_sentences:
        for lst in words:
            if lst[0].isupper() and len(lst[0]) > 3:
                s = s + 1
    return s
def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''    
    # TODO: Extract features that rely on capitalization.
    # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
    # TODO: Extract features that do not rely on capitalization.

    #Initialize an empty features numpy array to be populated later
    feature_size = 173
    module_func_num = 29
    features = np.zeros(feature_size, dtype=np.float64)

    sentences = []
    worded_sentences = []
    #Separate eaach sentence from the overall comment based on newline characters
    #and add them to sentences list.
    for sentence in comment.split("\n"):
        sentences.append(sentence.strip())
    
    #Populate worded_sentences with each tag word extracted from sentence in sentences.
    for sentence in sentences:
        words = []
        for word in sentence.split(" "):
            index = -1
            for i in range(len(word)):
                if word[i] == "/":
                    index = i
            #At this point index contains the index of the last "/" seen
            if index == -1:  # "/" does not exist
                words.append([word, None])
            else:
                words.append([word[:index], word[index+1:]])
        worded_sentences.append(words)
    feature_funcs = [
            num_uppercase_tokens,
            first_person_pronouns,
            second_person_pronouns,
            third_person_pronouns,
            coordinating_conjunctions,
            past_tense_verbs,
            future_tense_verbs,
            num_commas,
            num_mulchar_punc,
            num_common_nouns,
            num_proper_nouns,
            num_adverbs,
            num_wh_words,
            num_slang,
            avg_length_sentences,
            avg_length_tokens,
            num_sentences,
            average_aoa,
            average_img,
            average_fam,
            std_dev_aoa,
            std_dev_img,
            std_dev_fam,
            average_vmeansum,
            average_ameansum,
            average_dmeansum,
            std_dev_vmeansum,
            std_dev_ameansum,
            std_dev_dmeansum
    ]

    values = []
    for func in feature_funcs:
        values.append(func(worded_sentences))
    
    features[:module_func_num] = np.array(values)
    return features
    
def extract2(feat, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feat: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feat : numpy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    ''' 
    classes = ["Alt", "Center", "Left", "Right"]
    feat[29:] = lookup_statistics[comment_id]

    return feat

def extract2_helper():
    liwc_api_path = "/u/cs401/A1/feats"
    lookup = {}
    classes = ["Alt", "Center", "Left", "Right"]
    for cat in classes:
        cat_path = liwc_api_path+"/"+cat+"_IDs.txt"
        cat_file = open(cat_path, "r")
        features = np.load(liwc_api_path+"/"+cat+"_feats.dat.npy")

        lines = cat_file.readlines()
        for i in range(len(lines)):
            processed = lines[i].strip()
            lookup[processed] = features[i]
        cat_file.close()
    return lookup

lookup_statistics = extract2_helper()    

def main(args):
    #Declare necessary global variables here. 

    #Load data
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))

    # TODO: Call extract1 for each datatpoint to find the first 29 features. 
    # Add these to feats.
    classes = {
        "Alt" : 0,
        "Center": 1,
        "Left": 2,
        "Right": 3
    }
    for index in range(len(data)):
        features = extract1(data[index]["body"])
        features = extract2(features, data[index]["cat"], data[index]["id"])
        feats[index, :173] = features
        feats[index, 173] = classes[data[index]["cat"]]
    # TODO: Call extract2 for each feature vector to copy LIWC features (features 30-173)
    # into feats. (Note that these rely on each data point's class,
    # which is why we can't add them in extract1).

    np.savez_compressed(args.output, feats)

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()        

    main(args)

