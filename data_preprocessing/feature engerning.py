import os
from urllib.request import urlopen
import re
import shutil
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import csv

location_apis = '(getLatitude|getLongitude|getAddress)'
device_apis = '(getDeviceId|getSubscriberId|getSimSerialNumber|getLine1Number|sendTextMessage|getCurrentUser|getSimOperatorName|getSimCountryIso|getNetworkOperatorName|getNetworkType|getPhoneType)'
suspicious_permissions = '(SEND_SMS|READ_CONTACTS|WRITE_CONTACTS|CALL_PHONE|ADD_VOICEMAIL|READ_CALENDAR|WRITE_CALENDAR|ACCESS_FINE_LOCATION|ACCESS_COARSE_LOCATION|RECORD_AUDIO)'
touchEvent_apis = '(addFlags|getDisplayMetrics|getDefaultDisplay|heightPixels|widthPixels|dispatchTouchEvent|onClick)'
reflective_calls = '(class\.getMethod|class\.getDeclaredMethod|invoke|java\.lang\.reflect|getClass|getMethods|getDeclaredField)'
new_receiver = '(new( ){1,}BroadcastReceiver|extends( ){1,}BroadcastReceiver)'
new_service = '(new( ){1,}Service|extends( ){1,}Service)'
onCreate_apis = '(onCreate|run|exec|start)'


def read_cvs():
    csv.field_size_limit(100000000)
    csvFile = open("./Semantic Network/features.csv", "r", encoding='utf-8')
    reader = csv.reader(csvFile)
    feature = []
    weight = []
    for item in reader:
        feature.append(item[0])
        weight.append(float(item[1]))
    csvFile.close()
    return feature, weight



def find():
    feature, weight = read_cvs()

    diff_old_path = './research_project_summary-master/Diffs_old'
    diff_new_path = './research_project_summary-master/Diffs_new'
    repo_list = [diff_old_path,diff_new_path]

    for c in range(len(repo_list)):
        commit_name = repo_list[c]

        vector_set = []
        path_set = []
        weight_set = []
        add_set = []
        min_set = []

        for repo_name in next(os.walk(commit_name))[1]:
            if "." in repo_name:
                continue
            elif 'Telegram' in repo_name:
                continue
            num = 0
            for file_name in os.listdir(commit_name+ '/' + repo_name + "/Commits"):
                num += 1
                path =  commit_name + '/' + repo_name + "/Commits" + "/" + file_name
                if num % 50 == 0:
                    print(str(num) + '  ' + path)
                file = open(path, "r")
                #owner = file_name.split(':')[0]
                #fork_name = file_name.split(':')[1]
                #hash_val = file_name.split(':')[2]
                file_content = file.read()
                file.close()
                content_list = file_content.split('\n')

                vector = np.zeros(len(feature))
                file_weight = 0
                file_path = path
                add_lines = 0
                min_lines = 0

                for content_line in content_list:
                    if content_line == '':  # 空行
                        continue
                    match = re.match('\+( )*//.*', content_line)  # 注释行
                    if match:
                        continue
                    if content_line[0] == '+':
                        add_lines += 1
                    elif content_line[0] == '-':
                        min_lines += 1

                    for f in range(len(feature)):
                        matchObj = re.match('\+.*( +|\.)' + feature[f], content_line) #+ '( )*(\(|\{).*'
                        if matchObj:
                            vector[f] += 1
                            file_weight += weight[f]
                            break

                vector_set.append(vector)
                path_set.append(file_path)
                weight_set.append(file_weight)
                add_set.append(add_lines)
                min_set.append(min_lines)

            print(str(num) + '  ' + path)

        np.savez_compressed(commit_name.split('/')[-1] + '.npz', name=path_set, weight=weight_set, feature=vector_set, add_lines=add_set, min_lines=min_set)


if __name__ == '__main__':
    find()