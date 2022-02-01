import os
import re
from pathlib import Path
import numpy as np
import spacy
import nltk

nltk.download('wordnet')
nltk.download('punkt')

nlp = spacy.load("en_core_web_md")

import string
from sklearn.feature_extraction.text import TfidfVectorizer as ti
import pandas as pd
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):
        return [self.wnl.lemmatize(t, pos='v') for t in word_tokenize(articles)]


class LogPreprocessing:
    def __init__(self, path):
        self.path = path;
        self.regex = '(([0-9]+(-[0-9]+)+) ([0-9]+(:[0-9]+)+),[0-9]+\s[a-zA-z]+ \[[^\]]*] [a-zA-Z0-9.a-zA-Z0-9.a-zA-Z0-9]+.:)'
        self.normal_log_file_names = ['application_1445087491445_0005', 'application_1445087491445_0007',
                                      'application_1445175094696_0005',
                                      'application_1445062781478_0011',
                                      'application_1445062781478_0016',
                                      'application_1445062781478_0019',
                                      'application_1445076437777_0002',
                                      'application_1445076437777_0005',
                                      'application_1445144423722_0021',
                                      'application_1445144423722_0024',
                                      'application_1445182159119_0012'
                                      ]

    def isLog(self, x):
        if len(x) > 0:
            return True;
        else:
            return False

    def get_log_text(self, file, folder):
        txt = Path(self.path + "/" + folder + "/" + file).read_text()
        txt = re.sub(self.regex, "splitonthis", txt);
        txt = re.sub(r'[0-9]+', "number", txt);
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        txt = regex.sub(' ', txt)
        txt = re.sub("\n", "", txt)
        logs_list = re.split(r'splitonthis', txt);
        return logs_list

    def get_normal_log(self):
        self.logs = []
        self.all_normal_logs = []
        dir = os.listdir(self.path)
        first = True;
        for folder in dir:
            if folder != '.DS_Store' and folder != 'abnormal_label.txt' and self.normal_log_file_names.__contains__(
                    folder):
                textFiles = os.listdir(self.path + "/" + folder);
                for file in textFiles:
                    if file != '.DS_Store':
                        logs_list = self.get_log_text(file, folder)
                        final_list = [];
                        for log in filter(self.isLog, logs_list):
                            if not self.all_normal_logs.__contains__(log):
                                final_list.append(log);
                                self.all_normal_logs.append(log);
                        nd_array = np.column_stack((final_list, np.ones(len(final_list))))
                        if first:
                            self.logs = nd_array
                            first = False
                        else:
                            self.logs = np.append(self.logs, nd_array, axis=0);

    def all_log_data(self):
        all_log = []
        dir = os.listdir(self.path);
        for folder in dir:
            if folder != '.DS_Store' and folder != 'abnormal_label.txt' and not self.normal_log_file_names.__contains__(
                    folder):
                textFiles = os.listdir(self.path + "/" + folder);
                for file in textFiles:
                    if file != '.DS_Store':
                        logs_list = self.get_log_text(file, folder)
                        final_list = []
                        label = []
                        for log in filter(self.isLog, logs_list):
                            if not self.all_normal_logs.__contains__(log):
                                if not all_log.__contains__(log):
                                    final_list.append(log)
                                    label.append(0)
                                    all_log.append(log)
                        nd_array = np.column_stack((final_list, label))
                        self.logs = np.append(self.logs, nd_array, axis=0)


logPreprocessing = LogPreprocessing("/Users/arpanjeetsandhu/Desktop/Hadoop")
logPreprocessing.get_normal_log()
logPreprocessing.all_log_data()

similarity_arr = []
number_of_word = []
# number_of_similar logs
# number of words in logs
doc_1 = nlp(str(logPreprocessing.logs[1][0]))
for log in logPreprocessing.logs:
    word = str(log[0])
    number_of_word.append(len(word.split(" ")))
    doc_2 = nlp(word)
    a = doc_1.similarity(doc_2)
    similarity_arr.append(a)

unique_number = list(set(similarity_arr))
dict = {}
for num in unique_number:
    count = 0
    for similarity in similarity_arr:
        if num == similarity:
            count = count + 1
    dict[num] = count

final_arr = []

for i in similarity_arr:
    final_arr.append(dict.get(i))

logPreprocessing.logs = np.c_[logPreprocessing.logs, final_arr]
logPreprocessing.logs = np.c_[logPreprocessing.logs, number_of_word]

log_list = list(logPreprocessing.logs[0:len(logPreprocessing.logs), 0])


# TF_IDF MATRIX
tokenizer = LemmaTokenizer()
tf_idf = ti(stop_words='english', analyzer="word", tokenizer=tokenizer)
tf_idf_matrix = tf_idf.fit_transform(log_list)
tfidf_df = pd.DataFrame(tf_idf_matrix.toarray())



print("done")
