import os

import csv


def cluster():
    data_set = []
    commit_set = []
    csvFile = open("instance.csv", "w", encoding='utf-8')
    writer = csv.writer(csvFile,lineterminator='\n')
    writer.writerow(['text'])
    for repo_name in next(os.walk('.'))[1]:
        if "." in repo_name:
            continue

        for file_name in os.listdir("./" + repo_name):

            print(file_name)
            file = open("./" + repo_name + "/" + file_name, "r", encoding='utf-8')

            file_content = file.read()
            file.close()
            file_content = file_content.strip().replace('\n', ';')
            content_list = file_content.strip().replace('\r', ';')
            writer.writerow([content_list])

    csvFile.close()
    print('done')


def main():
    cluster()


main()
csv.field_size_limit(100000000)
csvFile = open("instance.csv", "r", encoding='utf-8')
reader = csv.reader(csvFile)
i=0
for item in reader:
    i+=1
    print(i)
csvFile.close()

