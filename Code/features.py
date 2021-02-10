import numpy as np
import argparse
import json
import string
from csv import DictReader as reader


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

    def helper_function():
        path = "/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv"
        bgl_file = open(path, "r")
        statistics = {}
        for data in reader(bgl_file):
            statistics[data["WORD"]] = data
        bgl_file.close()
        return statistics
    values = helper_function()
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

    def helper_function():
        path = "/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv"
        bgl_file = open(path, "r")
        statistics = {}
        for data in reader(bgl_file):
            statistics[data["WORD"]] = data
        bgl_file.close()
        return statistics
    values = helper_function()
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

    def helper_function():
        path = "/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv"
        bgl_file = open(path, "r")
        statistics = {}
        for data in reader(bgl_file):
            statistics[data["WORD"]] = data
        bgl_file.close()
        return statistics
    values = helper_function()
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

    def helper_function():
        path = "/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv"
        bgl_file = open(path, "r")
        statistics = {}
        for data in reader(bgl_file):
            statistics[data["WORD"]] = data
        bgl_file.close()
        return statistics
    values = helper_function()
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

    def helper_function():
        path = "/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv"
        bgl_file = open(path, "r")
        statistics = {}
        for data in reader(bgl_file):
            statistics[data["WORD"]] = data
        bgl_file.close()
        return statistics
    values = helper_function()
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

    def helper_function():
        path = "/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv"
        bgl_file = open(path, "r")
        statistics = {}
        for data in reader(bgl_file):
            statistics[data["WORD"]] = data
        bgl_file.close()
        return statistics
    values = helper_function()
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

    def helper_function():
        path = "/u/cs401/Wordlists/Ratings_Warriner_et_al.csv"
        wa_file = open(path, "r")
        statistics = {}
        for data in reader(wa_file):
            statistics[data["Word"]] = data
        wa_file.close()
        return statistics
   
    values = helper_function()
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

    def helper_function():
        path = "/u/cs401/Wordlists/Ratings_Warriner_et_al.csv"
        wa_file = open(path, "r")
        statistics = {}
        for data in reader(wa_file):
            statistics[data["Word"]] = data
        wa_file.close()
        return statistics
    values = helper_function()
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

    def helper_function():
        path = "/u/cs401/Wordlists/Ratings_Warriner_et_al.csv"
        wa_file = open(path, "r")
        statistics = {}
        for data in reader(wa_file):
            statistics[data["Word"]] = data
        wa_file.close()
        return statistics
    values = helper_function()
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

    def helper_function():
        path = "/u/cs401/Wordlists/Ratings_Warriner_et_al.csv"
        wa_file = open(path, "r")
        statistics = {}
        for data in reader(wa_file):
            statistics[data["Word"]] = data
        wa_file.close()
        return statistics
    values = helper_function()
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

    def helper_function():
        path = "/u/cs401/Wordlists/Ratings_Warriner_et_al.csv"
        wa_file = open(path, "r")
        statistics = {}
        for data in reader(wa_file):
            statistics[data["Word"]] = data
        wa_file.close()
        return statistics
    values = helper_function()
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

    def helper_function():
        path = "/u/cs401/Wordlists/Ratings_Warriner_et_al.csv"
        wa_file = open(path, "r")
        statistics = {}
        for data in reader(wa_file):
            statistics[data["Word"]] = data
        wa_file.close()
        return statistics
    values = helper_function()
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