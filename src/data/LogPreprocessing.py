import os
import re
from pathlib import Path
import numpy as np


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
        txt = re.sub(self.regex, "++++++", txt);
        txt = re.sub(r'[0-9]+', "number", txt);
        txt = re.sub("\n", "", txt)
        logs_list = re.split(r'[++++++]', txt);
        return logs_list

    def get_normal_log(self):
        self.logs = []
        self.all_normal_logs= []
        dir = os.listdir(self.path)
        first = True;
        for folder in dir:
            if folder != '.DS_Store' and folder != 'abnormal_label.txt' and self.normal_log_file_names.__contains__(folder):
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
            if folder != '.DS_Store' and folder != 'abnormal_label.txt' and not self.normal_log_file_names.__contains__(folder):
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
print("done")