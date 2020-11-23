import os
from urllib.request import urlopen
import re
import shutil
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

location_apis = '(getLatitude|getLongitude|getAddress)'
device_apis = '(getDeviceId|getSubscriberId|getSimSerialNumber|getLine1Number|sendTextMessage|getCurrentUser|getSimOperatorName|getSimCountryIso|getNetworkOperatorName|getNetworkType|getPhoneType)'
suspicious_permissions = '(SEND_SMS|READ_CONTACTS|WRITE_CONTACTS|CALL_PHONE|ADD_VOICEMAIL|READ_CALENDAR|WRITE_CALENDAR|ACCESS_FINE_LOCATION|ACCESS_COARSE_LOCATION|RECORD_AUDIO)'
touchEvent_apis = '(addFlags|getDisplayMetrics|getDefaultDisplay|heightPixels|widthPixels|dispatchTouchEvent|onClick)'
reflective_calls = '(class\.getMethod|class\.getDeclaredMethod|invoke|java\.lang\.reflect|getClass|getMethods|getDeclaredField)'
new_receiver = '(new( ){1,}BroadcastReceiver|extends( ){1,}BroadcastReceiver)'
new_service = '(new( ){1,}Service|extends( ){1,}Service)'
onCreate_apis = '(onCreate|run|exec|start)'
# getSystemService

# weight_dic = {}
# weight_dic["location"] = 20
# weight_dic["device_info"] = 20
# weight_dic["permission"] = 20
# weight_dic["receiver"] = 5
# weight_dic["service"] = 5
# weight_dic["long_file"] = 100
# weight_dic["touchEvent"] = 5
# weight_dic["onCreate"] = 2
# weight_dic["reflection"] = 5
weight_dic = {}
weight_dic["location"] = 1
weight_dic["device_info"] = 1
weight_dic["permission"] = 1
weight_dic["receiver"] = 1
weight_dic["service"] = 1
weight_dic["long_file"] = 1
weight_dic["touchEvent"] = 1
weight_dic["onCreate"] = 1
weight_dic["reflection"] = 1


def cluster():
    data_set = []
    commit_set = []
    for repo_name in next(os.walk('.'))[1]:
        if "." in repo_name:
            continue
        for file_name in os.listdir("./" + repo_name + "/Commits"):
            file = open("./" + repo_name + "/Commits" + "/" + file_name, "r")

            file_content = file.read()
            file.close()
            content_list = file_content.split('\n')



    data_set = np.array(data_set)
    np.save("./vectors2.npy", data_set)
    with open("./commits2.pkl", "wb") as f:
        pickle.dump(commit_set, f)

    kmeans = KMeans(n_clusters=15).fit(data_set)
    y_kmeans = kmeans.predict(data_set)
    res = {}
    for i in range(len(commit_set)):
        if y_kmeans[i] in res:
            res[y_kmeans[i]].append((commit_set[i], data_set[i]))
        else:
            res[y_kmeans[i]] = []
            res[y_kmeans[i]].append((commit_set[i], data_set[i]))

    for key in res:
        print(key)
        for commit_vec in res[key]:
            print(commit_vec[0], commit_vec[1])


def main():
    cluster()


main()

