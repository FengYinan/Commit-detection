import re

flag = False


# The comment of file will be deleted if exist lines[i]
def handle_single_comment(lines, i):
    k =lines[i]

    ends = re.finditer('//', lines[i])
    for end in ends:
        start = end.start()
        cend = end.end()
        if start-6 > 0:
            if 'http' in lines[i][start-6: cend]:
                continue
        lines[i] = lines[i][0:end.start()]
        lines[i] += '\r\n'


# @return -1:the Line is Comment Line,should delete this line
# @return -2:Only begin Comment found in this Line
# @return  0:Not find Comment
def handle_document_comment(lines, i):
    global flag
    while True:
        if not flag:
            index = lines[i].find("/*")
            if index != -1:
                flag = True
                index2 = lines[i].find("*/", index + 2)
                if index2 != -1:
                    lines[i] = lines[i][0:index]+lines[i][index2+2:]
                    flag = False  # continue look for comment
                else:
                    lines[i] = lines[i][0:index]  # only find "begin comment
                    lines[i] += '\r\n'
                    return -2
            else:
                return 0  # not find
        else:
            index2 = lines[i].find("*/")
            if index2 != -1:
                flag = False
                lines[i] = lines[i][index2+2:]  # continue look for comment
            else:
                return -1  # should delete this


# At last print the handled result
def remove_comment(file):
    global flag
    f = open(file, "r")
    lines = f.readlines()
    f.close()
    length = len(lines)
    i = 0
    while i < length:
        ret = handle_document_comment(lines, i)
        if ret == -1:
            if not flag:
                print("There must be some wrong")
            del lines[i]
            i -= 1
            length -= 1
        elif ret == 0:
            handle_single_comment(lines, i)
        else:
            pass
        i += 1
    # output_file(lines)
    # new_file = file.split(".java")[0] + "_no_com" + ".java"
    # write_result(new_file, lines)
    return lines


def write_result(file, lines):
    f = open(file, "w")
    for line in lines:
        if line == '':  # if find blank, continue this loop
            continue
        f.write(line)
    f.close()


def output_file(lines):
    for line in lines:
        if line == '':
            continue
        print(line)


if __name__ == '__main__':
    path = "D:/1-Android/0103-Examples/02-facebook-android-sdk/AuthorizationClient.java"
    remove_comment(path)
