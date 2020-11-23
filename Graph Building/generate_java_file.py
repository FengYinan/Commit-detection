import os
import re
import requests
from numba import jit

def generate_java_file_from_commit_name(path_commit, path_file):
    # obtain the file name list
    commit_list = os.listdir(path_commit)
    # assign the diff name
    for i in range(1, len(commit_list)):
        print(i)
        try:
            # get the commit name
            diff = commit_list[i]
            diff_list = diff.split("_")  # extract more information
            # read the diff file's context
            with open(path_commit + diff, 'r', encoding="utf-8") as r:
                lines_diff = r.readlines()
            tag = False  # we assume no java.file in the commit
            new_location = []
            old_location = []
            for line in lines_diff:
                if ("diff --git" in line) and (".java" in line):
                    # obtain the location of file
                    new_location = re.findall(r"git a(.+?) b", line)[0]
                    old_location = re.findall(r" b/(.+?)$", line)  # further analysis
                    tag = True
                    break  # stop this loop, one java file is enough
            if not tag:
                continue
                # if this file does not have java.file, we continue to next loop
            if not old_location:
                continue
                # if this file is new created, we don't consider it, because there is only 1 graph
            else:
                old_location = old_location[0]
            # print(new_location)

            # generate the new url
            url_new = "https://raw.Githubusercontent.com/" + diff_list[0] + '/' + diff_list[1] + '/' + diff_list[
                2] + new_location
            r = requests.get(url_new)
            path_java = path_file + str(i)  # define the old file path
            folder = os.path.exists(path_java)
            if not folder:
                os.makedirs(path_java)
            with open(path_java + '/new.java', 'w', encoding='utf-8') as w_new:
                w_new.write(r.text)
            # finish creat old java

            # find the parent file
            url_commit = "https://github.com/" + diff_list[0] + '/' + diff_list[1] + '/commit/' + diff_list[2]
            r = requests.get(url_commit)
            # write the content to a file
            with open(path_java + '/html.html', 'w', encoding='utf-8') as f_html:
                f_html.write(r.text)
            # open the html file
            with open(path_java + '/html.html', 'r', encoding='utf-8') as r:
                lines_web = r.readlines()
            old_git = []
            for line in lines_web:
                if "data-hotkey=\"p\"" in line:
                    old_git = re.findall(r"href=\"/(.*)\">", line)[0]
                    old_git = old_git.replace("/commit", "")
                    # print(old_git)

            if not old_git:
                continue
                # if no parent, exist
            # define the url of old java file
            url_old = "https://raw.Githubusercontent.com/" + old_git + '/' + new_location
            # obtain the file context
            r = requests.get(url_old)
            with open(path_java + '/old.java', 'w', encoding='utf-8') as w_new:
                w_new.write(r.text)
            # finish creat old java
            # delete the html file to save space
            path_html = path_java + '/html.html'
            if os.path.exists(path_html):
                os.remove(path_html)

            # print something
            print("successful creat #", i, "file")
        except:
            print("there are something wrong")
        continue


def generate_java_file_from_url(path_commit, path_file):
    # obtain the file name list
    with open(path_commit, 'r') as r:
        commit_list = r.readlines()
    # assign the diff name
    for i in range(len(commit_list)):
        print(i)
        try:
            # get the commit url
            url_diff = commit_list[i]
            url_diff = url_diff.replace("\n", "")
            # obtain the context of diff file
            r = requests.get(url_diff + ".patch")
            path_diff = path_file.replace(path_file.split('/')[-2], 'diff')  # define the file path, diff, old_java, new_java, html
            folder = os.path.exists(path_diff)
            if not folder:
                os.makedirs(path_diff)
            with open(path_diff + '/diff_' + str(i) + '.txt', 'w', encoding='utf-8') as w_new:
                w_new.write(r.text)
            # finish creat the diff file

            # read the diff file's context
            with open(path_diff + '/diff_' + str(i) + '.txt', 'r', encoding="utf-8") as r:
                lines_diff = r.readlines()
            tag = False  # we assume no java.file in the commit
            new_location = []
            old_location = []
            for line in lines_diff:
                if ("diff --git" in line) and (".java" in line):
                    # obtain the location of file
                    new_location = re.findall(r"git a(.+?) b", line)[0]
                    old_location = re.findall(r" b/(.+?)$", line)  # further analysis
                    tag = True
                    #break  # stop this loop, one java file is enough
            if not tag:
                continue
                # if this file does not have java.file, we continue to next loop
            if not old_location:
                continue
                # if this file is new created, we don't consider it, because there is only 1 graph
            else:
                old_location = old_location[0]
            # print(new_location)

            # generate the new url
            diff_list = re.match(".*github.com/(.*)/(.*)/commit/(.*)$", url_diff)
            """
            for the url: 'https://github.com/ChenKaiJung/facebook-android-sdk/commit/0690322310a7c9d7d486686255e6b1fdd87acb9c'
            diff_list[1] = 'ChenKaiJung'
            diff_list[2] = 'facebook-android-sdk'
            diff_list[3] = '0690322310a7c9d7d486686255e6b1fdd87acb9c'
            """
            url_new = "https://raw.Githubusercontent.com/" + diff_list[1] + '/' + diff_list[2] + '/' + diff_list[3]\
                      + new_location
            # download the new java file
            r = requests.get(url_new)
            path_java = path_file + str(i)  # define the file path, diff, old_java, new_java, html
            folder = os.path.exists(path_java)
            if not folder:
                os.makedirs(path_java)
            with open(path_java + '/new.java', 'a+', encoding='utf-8') as w_new:
                w_new.write(r.text)
            # finish creat old java

            # find the parent file
            r = requests.get(url_diff)
            # write the content to a file
            with open(path_java + '/html.html', 'w', encoding='utf-8') as f_html:
                f_html.write(r.text)
            # open the html file
            with open(path_java + '/html.html', 'r', encoding='utf-8') as r:
               lines_web = r.readlines()
            old_git = []
            for line in lines_web:
                if "data-hotkey=\"p\"" in line:
                    old_git = re.findall(r"href=\"/(.*)\">", line)[0]
                    old_git = old_git.replace("/commit", "")
                    # print(old_git)

            if not old_git:
                continue
                # if no parent, exist
            # define the url of old java file
            url_old = "https://raw.Githubusercontent.com/" + old_git + '/' + new_location
            # obtain the file context
            r = requests.get(url_old)
            with open(path_java + '/old.java', 'a+', encoding='utf-8') as w_new:
                w_new.write(r.text)
            # finish creat old java

            # delete the html file to save space
            # """
            path_html = path_java + '/html.html'
            if os.path.exists(path_html):
                os.remove(path_html)
            # """


            # print something
            print("successful creat #", i, "file")
        except:
            print("there are something wrong")
        continue


def generate_java_file_from_commit_txt(path_commit, path_file):
    # obtain the file name list
    commit_list = os.listdir(path_commit)
    # assign the diff name
    #if len(commit_list) > 200:
        #number_loop = 200
    #else:
    number_loop = len(commit_list)
    # write the commit_list for test
    if not os.path.exists(path_file):
        os.makedirs(path_file)
    with open(path_file + 'commits_name.txt', 'w') as w:
        for line in commit_list[:number_loop]:
            w.write(line + '\n')
    # finish writing
    for i in range(number_loop):
        if (i+1) % 50 == 0:
            print('{}, {}'.format(i+1, path_commit.split('/')[-3]))
        try:
            # get the commit name
            diff = commit_list[i]
            project_name = re.match(r".*/(.*)/Commits/$", path_commit)
            project_name = project_name[1]  # extract the string
            diff_list = re.match(r"^(.*)_(.*).txt", diff)  # extract more information
            # read the diff file's context
            with open(path_commit + diff, 'r', encoding="utf-8") as r:
                lines_diff = r.readlines()
            tag = False  # we assume no java.file in the commit
            new_locations = []
            old_location = []
            for line in lines_diff:
                if ("diff --git" in line) and (".java" in line):
                    # obtain the location of file
                    new_locations.append(re.findall(r"git a(.+?) b", line)[0])
                    old_location.append(re.findall(r" b/(.+?)$", line))  # further analysis
                    tag = True
                    #break  # stop this loop, one java file is enough
            if not tag:
                continue
                # if this file does not have java.file, we continue to next loop
            if not old_location:
                continue
                # if this file is new created, we don't consider it, because there is only 1 graph
            # print(new_location)

            """ generate the new url
            diff_list[1] is the project_owner
            project_name is the project_name
            diff_list[2] is the commit hash
            """
            for java in range(len(new_locations)):
                new_location = new_locations[java]

                url_new = "https://raw.Githubusercontent.com/" + diff_list[1] + '/' + project_name + '/' + diff_list[
                    2] + new_location
                r = requests.get(url_new)
                path_java = path_file + str(i)  # define the old file path
                folder = os.path.exists(path_java)
                if not folder:
                    os.makedirs(path_java)
                with open(path_java + '/new.java', 'a+', encoding='utf-8') as w_new:
                    w_new.write(r.text+ '\n')
                # finish creat old java

                # find the parent file
                url_commit = "https://github.com/" + diff_list[1] + '/' + project_name + '/commit/' + diff_list[2]
                r = requests.get(url_commit)
                # write the content to a file
                with open(path_java + '/html.html', 'w', encoding='utf-8') as f_html:
                    f_html.write(r.text)
                # open the html file
                with open(path_java + '/html.html', 'r', encoding='utf-8') as r:
                    lines_web = r.readlines()
                old_git = []
                for line in lines_web:
                    if "data-hotkey=\"p\"" in line:
                        old_git = re.findall(r"href=\"/(.*)\">", line)[0]
                        old_git = old_git.replace("/commit", "")
                        # print(old_git)

                if not old_git:
                    continue
                    # if no parent, exist
                # define the url of old java file
                url_old = "https://raw.Githubusercontent.com/" + old_git + '/' + new_location
                # obtain the file context
                r = requests.get(url_old)
                with open(path_java + '/old.java', 'a+', encoding='utf-8') as w_new:
                    w_new.write(r.text+ '\n')
                # finish creat old java
                # delete the html file to save space
                path_html = path_java + '/html.html'
                if os.path.exists(path_html):
                    os.remove(path_html)


            # print something
            print("successful creat #", i, "file")
        except:
            print("there are something wrong")
        continue
"""
We divide the java file generation into three types
First: generate from the format:
    owner_of_fork:fork_name:hash_value
    such as Ekt0s:android-saripaar:86e0b3366df7cfd83943488aa2f6902f84b225f9
Second: generate from txt file:
    owner_of_fork:hash_value.txt
    such as Keanu:89cf2cac0ac1b0138f0c403c7af1ed49959a049c.txt
Third: generate from url:
    This is most easy way
"""


def main_test(path_original):
    """
    # Generate from commit
    path_commit = "E:/1-Android/Graph-Generated/Commits/Commits_DIFF/"
    path_file = "E:/1-Android/Graph-Generated/Commits/Commits_File_new/"
    generate_java_file_from_commit_name(path_commit, path_file)
    """

    """
    Generate from url"
    path_commit = "E:/1-Android/Graph-Generated/GoogleDoc/Commits_Draft/commit_url.txt"
    path_file = "E:/1-Android/Graph-Generated/GoogleDoc/Commits_Draft/Commits_File/"
    generate_java_file_from_url(path_commit, path_file)
    """

    # Generate from txt file
    path_dir = os.listdir(path_original)
    if '_old' in path_original:
        path_dir = path_dir[39:]
    for path in path_dir:
        if 'Telegram' in path:
            continue
        path_commit = path_original + path + '/Commits/'
        path_file = 'C:/Users/yinan/Desktop/python_code/' + path_original.split('/')[-2] + "_result/" + path + '/Commits_File/'
        generate_java_file_from_commit_txt(path_commit, path_file)

if __name__ == '__main__':
    main_test()










