#  find th method name and locations for each java
import numpy as np
import re
import csv

def in_str(start, match_str):
    result = 0
    for match in match_str:
        if start.start() >= match[0] and start.end() <= match[1]:
            result += 1
            break
        else:
            continue
    return result

def find_node_name_location(codes):
    method_pattern = re.compile(
        r"((void|public|private|protected|static|final|native|synchronized|abstract|transient)+\s[\$_\w\<\>\w\s\[\]\s]*[\$_\w]+\([^\)]*\)?\s*)")
    class_pattern = re.compile(r"(class\s[\$_\w]+)")
    str_parrern = re.compile(r"((?<=[\"|\']).*?(?=[\"|\']))")

    trigger = 0

    node_line = []
    node_list = []
    stack = []
    for index in range(len(codes)):
        line = codes[index]
        match_method = method_pattern.findall(line)
        match_class = class_pattern.findall(line)
        match_str = str_parrern.finditer(line)
        match_str = [[match.start(), match.end()] for match in match_str]

        added = False


        if (match_method or match_class):
            if line.strip()[-1] != ';':
                added = True
                stack.append([match_method, match_class, index,line])
                if '{' not in line :
                    trigger = 1
            else:
                if match_class:  # in this case, there is a class
                    node_name = match_class[0].split(" ")[-1]
                    node_line.append([node_name,index, index+1])
                    # generate the hash value for the node
                    node_list.append(node_name)
                if match_method:  # in this case, there is a method
                    node_name = match_method[0][0].split('(')[0].split(' ')[-1]
                    node_line.append([node_name,index, index+1])
                    # generate the hash value for the node
                    node_list.append(node_name)
                continue
        else:
            starts = re.finditer('{', line)
            for start in starts:
                if in_str(start, match_str):
                    continue
                if not trigger:
                    added = True
                    stack.append([match_method, match_class, index, line])
                    continue
                else:
                    trigger = 0
                    continue
            # this means open a block, enter level 1 (class)
        ends = re.finditer('}', line)
        for end in ends:
            if in_str(end, match_str):
                continue
            if end.end() < len(line) - 2 and added:
                match_method, match_class, start_location, _ = stack.pop(-2)
            else:
                match_method, match_class, start_location, _ = stack.pop()
                # try:
                #     match_method, match_class, start_location, _ = stack.pop()
                # except:
                #     print(line, index, codes[index-5:index+5])
            if match_class:  # in this case, there is a class
                node_name = match_class[0].split(" ")[-1]
                node_line.append([node_name, start_location, index + 1])
                # generate the hash value for the node
                node_list.append(node_name)
            if match_method:  # in this case, there is a method
                node_name = match_method[0][0].split('(')[0].split(' ')[-1]
                node_line.append([node_name, start_location, index + 1])
                # generate the hash value for the node
                node_list.append(node_name)


    return node_line, node_list

def read_cvs():
    csv.field_size_limit(100000000)
    csvFile = open("D:/UVA_RESEARCH/COMMIT/data/Semantic Network/features.csv", "r", encoding='utf-8')
    reader = csv.reader(csvFile)
    feature = []
    weight = []
    dic={}
    for item in reader:
        feature.append(item[0])
        weight.append(float(item[1]))
        dic[item[0]] = float(item[1])
    csvFile.close()
    return feature, weight, dic


def read_apilist():
    # api_list = np.load('api_list.npy')
    # feature = api_list.tolist()

    feature = ["WNetAddConnection", "WNetAddConnection2", "WNetAddConnection3", "WNetCancelConnection", "WNetCancelConnection2", "WNetCloseEnum", "WNetConnectionDialog", "WNetDisconnectDialog", "WNetEnumResource", "WNetGetConnection", "WNetGetLastError", "WNetGetUniversalName", "WNetGetUser", "WNetOpenEnum", "BroadcastSystemMessage", "GetMessagePos", "GetMessageTime", "PostMessage", "PostThreadMessage", "RegisterWindowMessage", "ReplyMessage", "SendMessage", "SendMessageCallback", "SendMessageTimeout", "SendNotifyMessage", "CloseHandle", "CompareFileTime", "CopyFile", "CreateDirectory", "CreateFile", "CreateFileMapping", "DeleteFile", "DeviceIoControl", "DosDateTimeToFileTime", "FileTimeToDosDateTime", "FileTimeToLocalFileTime", "FileTimeToSystemTime", "FindClose", "FindFirstFile", "FindNextFile", "FlushFileBuffers", "FlushViewOfFile", "GetBinaryType", "GetCompressedFileSize", "GetCurrentDirectory", "GetDiskFreeSpace", "GetDiskFreeSpaceEx", "GetDriveType", "GetExpandedName", "GetFileAttributes", "GetFileInformationByHandle", "GetFileSize", "GetFileTime", "GetFileType", "GetFileVersionInfo", "GetFileVersionInfoSize", "GetFullPathName", "GetLogicalDrives", "GetLogicalDriveStrings", "GetOverlappedResult", "GetPrivateProfileInt", "GetPrivateProfileSection", "GetPrivateProfileString", "GetProfileInt", "GetProfileSection", "GetProfileString", "GetShortPathName", "GetSystemDirectory", "GetTempFileName", "GetTempPath", "GetVolumeInformation", "GetWindowsDirectory", "hread", "hwrite", "lclose", "lcreat", "llseek", "LockFile", "LockFileEx", "lopen", "lread", "lwrite", "LZClose", "LZCopy", "LZInit", "LZOpenFile", "LZRead", "LZSeek", "MapViewOfFile", "MoveFile", "OpenFile", "OpenFileMapping", "QueryDosDevice", "ReadFile", "ReadFileEx", "RegCloseKey", "RegConnectRegistry", "RegCreateKey", "RegCreateKeyEx", "RegDeleteKey", "RegDeleteValue", "RegEnumKey", "RegEnumKeyEx", "RegEnumValue", "RegFlushKey", "RegGetKeySecurity", "RegLoadKey", "RegNotifyChangeKeyValue", "RegOpenKey", "RegOpenKeyEx", "RegQueryInfoKey", "RegQueryValue", "RegQueryValueEx", "RegReplaceKey", "RegRestoreKey", "RegSaveKey", "RegSetKeySecurity", "RegSetValue", "RegSetValueEx", "RegUnLoadKey", "RemoveDirectory", "SearchPath", "SetCurrentDirectory", "SetEndOfFile", "SetFileAttributes", "SetFilePointer", "SetFileTime", "SetHandleCount", "SetVolumeLabel", "SystemTimeToFileTime", "UnlockFile", "UnlockFileEx", "UnmapViewOfFile", "VerFindFile", "VerInstallFile", "VerLanguageName", "VerQueryValue", "WriteFile", "WriteFileEx", "WritePrivateProfileSection", "WritePrivateProfileString", "WriteProfileSection", "WriteProfileString", "ActivateKeyboardLayout", "Beep", "CharToOem", "ClipCursor", "ConvertDefaultLocale", "CreateCaret", "DestroyCaret", "EnumCalendarInfo", "EnumDateFormats", "EnumSystemCodePages", "EnumSystemLocales", "EnumTimeFormats", "ExitWindowsEx", "ExpandEnvironmentStrings", "FreeEnvironmentStrings", "GetACP", "GetAsyncKeyState", "GetCaretBlinkTime", "GetCaretPos", "GetClipCursor", "GetCommandLine", "GetComputerName", "GetCPInfo", "GetCurrencyFormat", "GetCursor", "GetCursorPos", "GetDateFormat", "GetDoubleClickTime", "GetEnvironmentStrings", "GetEnvironmentVariable", "GetInputState", "GetKBCodePage", "GetKeyboardLayout", "GetKeyboardLayoutList", "GetKeyboardLayoutName", "GetKeyboardState", "GetKeyboardType", "GetKeyNameText", "GetKeyState", "GetLastError", "GetLocaleInfo", "GetLocalTime", "GetNumberFormat", "GetOEMCP", "GetQueueStatus", "GetSysColor", "GetSystemDefaultLangID", "GetSystemDefaultLCID", "GetSystemInfo", "GetSystemMetrics", "GetSystemPowerStatus", "GetSystemTime", "GetSystemTimeAdjustment", "GetThreadLocale", "GetTickCount", "GetTimeFormat", "GetTimeZoneInformation", "GetUserDefaultLangID", "GetUserDefaultLCID", "GetUserName", "GetVersion", "GetVersionEx", "HideCaret", "IsValidCodePage", "IsValidLocale", "keybd_event", "LoadKeyboardLayout", "MapVirtualKey", "MapVirtualKeyEx", "MessageBeep", "mouse_event", "OemKeyScan", "OemToChar", "SetCaretBlinkTime", "SetCaretPos", "SetComputerName", "SetCursor", "SetCursorPos", "SetDoubleClickTime", "SetEnvironmentVariable", "SetKeyboardState", "SetLocaleInfo", "SetLocalTime", "SetSysColors", "SetSystemCursor", "SetSystemTime", "SetSystemTimeAdjustment", "SetThreadLocale", "SetTimeZoneInformation", "ShowCaret", "ShowCursor", "SwapMouseButton", "SystemParametersInfo", "SystemTimeToTzSpecificLocalTime", "ToAscii", "ToUnicode", "UnloadKeyboardLayout", "VkKeyScan", "CancelWaitableTimer", "CallNamedPipe", "ConnectNamedPipe", "CreateEvent", "CreateMailslot", "CreateNamedPipe", "CreatePipe", "CreateProcess", "CreateSemaphore", "CreateWaitableTimer", "DisconnectNamedPipe", "DuplicateHandle", "ExitProcess", "FindCloseChangeNotification", "FindExecutable", "FindFirstChangeNotification", "FindNextChangeNotification", "FreeLibrary", "GetCurrentProcess", "GetCurrentProcessId", "GetCurrentThread", "GetCurrentThreadId", "GetExitCodeProces", "GetExitCodeThread", "GetHandleInformation", "GetMailslotInfo", "GetModuleFileName", "GetModuleHandle", "GetPriorityClass", "GetProcessShutdownParameters", "GetProcessTimes", "GetProcessWorkingSetSize", "GetSartupInfo", "GetThreadPriority", "GetTheardTimes", "GetWindowThreadProcessId", "LoadLibrary", "LoadLibraryEx", "LoadModule", "MsgWaitForMultipleObjects", "SetPriorityClass", "SetProcessShutdownParameters", "SetProcessWorkingSetSize", "SetThreadPriority", "ShellExecute", "TerminateProcess", "WinExec", "AdjustWindowRect", "AnyPopup", "ArrangeIconicWindows", "AttachThreadInput", "BeginDeferWindowPos", "BringWindowToTop", "CascadeWindows", "ChildWindowFromPoint", "ClientToScreen", "CloseWindow", "CopyRect", "DeferWindowPos", "DestroyWindow", "DrawAnimatedRects", "EnableWindow", "EndDeferWindowPos", "EnumChildWindows", "EnumThreadWindows", "EnumWindows", "EqualRect", "FindWindow", "FindWindowEx", "FlashWindow", "GetActiveWindow", "GetCapture", "GetClassInfo", "GetClassLong", "GetClassName", "GetClassWord", "GetClientRect", "GetDesktopWindow", "GetFocus", "GetForegroundWindow", "GetLastActivePopup", "GetParent", "GetTopWindow", "GetUpdateRect", "GetWindow", "GetWindowContextHelpId", "GetWindowLong", "GetWindowPlacement", "GetWindowRect", "GetWindowText", "GetWindowTextLength", "GetWindowWord", "InflateRect", "IntersectRect", "InvalidateRect", "IsChild", "IsIconic", "IsRectEmpty", "IsWindow", "IsWindowEnabled", "IsWindowUnicode", "IsWindowVisible", "IsZoomed", "LockWindowUpdate", "MapWindowPoints", "MoveWindow", "OffsetRect", "OpenIcon", "PtInRect", "RedrawWindow", "ReleaseCapture", "ScreenToClient", "ScrollWindow", "ScrollWindowEx", "SetActiveWindow", "SetCapture", "SetClassLong", "SetClassWord", "SetFocusAPI", "SetForegroundWindow", "SetParent", "SetRect", "SetRectEmpty", "SetWindowContextHelpId", "SetWindowLong", "SetWindowPlacement", "SetWindowPos", "SetWindowText", "SetWindowWord", "ShowOwnedPopups", "ShowWindow", "ShowWindowAsync", "SubtractRect", "TileWindows", "UnionRect", "UpdateWindow", "ValidateRect", "WindowFromPoint"]

    dic ={}
    for f in range(len(feature)):
        dic[feature[f]] = f
    return feature, dic


def find_node_feature(codes, node_line):
    node_feature = {}

    _, weight, weight_dic = read_cvs()
    feature, dic = read_apilist()

    feature_len = len(feature)

    # Assign the weight
    for node in range(len(node_line)):
        vector = np.zeros(feature_len, dtype=float)

        for line in codes[node_line[node][1]: node_line[node][2]]:
            api = re.findall(r'^.*\.(.*)\(.*$', line)
            if not api:
                api = re.findall(r"^.* (.*)\(.*$", line)
            for atok in api:
                if atok in dic.keys():
                    vector[dic[atok]] += 1
                    # if atok in weight_dic.keys():
                    #     vector[dic[atok]] += weight_dic[atok]
            # finish the scanning
        node_feature[node] = vector
    # return the features
    return node_feature


#%%
# we need to merge feature and graph
if __name__ == '__main__':
    feature = ["WNetAddConnection", "WNetAddConnection2", "WNetAddConnection3", "WNetCancelConnection", "WNetCancelConnection2", "WNetCloseEnum", "WNetConnectionDialog", "WNetDisconnectDialog", "WNetEnumResource", "WNetGetConnection", "WNetGetLastError", "WNetGetUniversalName", "WNetGetUser", "WNetOpenEnum", "BroadcastSystemMessage", "GetMessagePos", "GetMessageTime", "PostMessage", "PostThreadMessage", "RegisterWindowMessage", "ReplyMessage", "SendMessage", "SendMessageCallback", "SendMessageTimeout", "SendNotifyMessage", "CloseHandle", "CompareFileTime", "CopyFile", "CreateDirectory", "CreateFile", "CreateFileMapping", "DeleteFile", "DeviceIoControl", "DosDateTimeToFileTime", "FileTimeToDosDateTime", "FileTimeToLocalFileTime", "FileTimeToSystemTime", "FindClose", "FindFirstFile", "FindNextFile", "FlushFileBuffers", "FlushViewOfFile", "GetBinaryType", "GetCompressedFileSize", "GetCurrentDirectory", "GetDiskFreeSpace", "GetDiskFreeSpaceEx", "GetDriveType", "GetExpandedName", "GetFileAttributes", "GetFileInformationByHandle", "GetFileSize", "GetFileTime", "GetFileType", "GetFileVersionInfo", "GetFileVersionInfoSize", "GetFullPathName", "GetLogicalDrives", "GetLogicalDriveStrings", "GetOverlappedResult", "GetPrivateProfileInt", "GetPrivateProfileSection", "GetPrivateProfileString", "GetProfileInt", "GetProfileSection", "GetProfileString", "GetShortPathName", "GetSystemDirectory", "GetTempFileName", "GetTempPath", "GetVolumeInformation", "GetWindowsDirectory", "hread", "hwrite", "lclose", "lcreat", "llseek", "LockFile", "LockFileEx", "lopen", "lread", "lwrite", "LZClose", "LZCopy", "LZInit", "LZOpenFile", "LZRead", "LZSeek", "MapViewOfFile", "MoveFile", "OpenFile", "OpenFileMapping", "QueryDosDevice", "ReadFile", "ReadFileEx", "RegCloseKey", "RegConnectRegistry", "RegCreateKey", "RegCreateKeyEx", "RegDeleteKey", "RegDeleteValue", "RegEnumKey", "RegEnumKeyEx", "RegEnumValue", "RegFlushKey", "RegGetKeySecurity", "RegLoadKey", "RegNotifyChangeKeyValue", "RegOpenKey", "RegOpenKeyEx", "RegQueryInfoKey", "RegQueryValue", "RegQueryValueEx", "RegReplaceKey", "RegRestoreKey", "RegSaveKey", "RegSetKeySecurity", "RegSetValue", "RegSetValueEx", "RegUnLoadKey", "RemoveDirectory", "SearchPath", "SetCurrentDirectory", "SetEndOfFile", "SetFileAttributes", "SetFilePointer", "SetFileTime", "SetHandleCount", "SetVolumeLabel", "SystemTimeToFileTime", "UnlockFile", "UnlockFileEx", "UnmapViewOfFile", "VerFindFile", "VerInstallFile", "VerLanguageName", "VerQueryValue", "WriteFile", "WriteFileEx", "WritePrivateProfileSection", "WritePrivateProfileString", "WriteProfileSection", "WriteProfileString", "ActivateKeyboardLayout", "Beep", "CharToOem", "ClipCursor", "ConvertDefaultLocale", "CreateCaret", "DestroyCaret", "EnumCalendarInfo", "EnumDateFormats", "EnumSystemCodePages", "EnumSystemLocales", "EnumTimeFormats", "ExitWindowsEx", "ExpandEnvironmentStrings", "FreeEnvironmentStrings", "GetACP", "GetAsyncKeyState", "GetCaretBlinkTime", "GetCaretPos", "GetClipCursor", "GetCommandLine", "GetComputerName", "GetCPInfo", "GetCurrencyFormat", "GetCursor", "GetCursorPos", "GetDateFormat", "GetDoubleClickTime", "GetEnvironmentStrings", "GetEnvironmentVariable", "GetInputState", "GetKBCodePage", "GetKeyboardLayout", "GetKeyboardLayoutList", "GetKeyboardLayoutName", "GetKeyboardState", "GetKeyboardType", "GetKeyNameText", "GetKeyState", "GetLastError", "GetLocaleInfo", "GetLocalTime", "GetNumberFormat", "GetOEMCP", "GetQueueStatus", "GetSysColor", "GetSystemDefaultLangID", "GetSystemDefaultLCID", "GetSystemInfo", "GetSystemMetrics", "GetSystemPowerStatus", "GetSystemTime", "GetSystemTimeAdjustment", "GetThreadLocale", "GetTickCount", "GetTimeFormat", "GetTimeZoneInformation", "GetUserDefaultLangID", "GetUserDefaultLCID", "GetUserName", "GetVersion", "GetVersionEx", "HideCaret", "IsValidCodePage", "IsValidLocale", "keybd_event", "LoadKeyboardLayout", "MapVirtualKey", "MapVirtualKeyEx", "MessageBeep", "mouse_event", "OemKeyScan", "OemToChar", "SetCaretBlinkTime", "SetCaretPos", "SetComputerName", "SetCursor", "SetCursorPos", "SetDoubleClickTime", "SetEnvironmentVariable", "SetKeyboardState", "SetLocaleInfo", "SetLocalTime", "SetSysColors", "SetSystemCursor", "SetSystemTime", "SetSystemTimeAdjustment", "SetThreadLocale", "SetTimeZoneInformation", "ShowCaret", "ShowCursor", "SwapMouseButton", "SystemParametersInfo", "SystemTimeToTzSpecificLocalTime", "ToAscii", "ToUnicode", "UnloadKeyboardLayout", "VkKeyScan", "CancelWaitableTimer", "CallNamedPipe", "ConnectNamedPipe", "CreateEvent", "CreateMailslot", "CreateNamedPipe", "CreatePipe", "CreateProcess", "CreateSemaphore", "CreateWaitableTimer", "DisconnectNamedPipe", "DuplicateHandle", "ExitProcess", "FindCloseChangeNotification", "FindExecutable", "FindFirstChangeNotification", "FindNextChangeNotification", "FreeLibrary", "GetCurrentProcess", "GetCurrentProcessId", "GetCurrentThread", "GetCurrentThreadId", "GetExitCodeProces", "GetExitCodeThread", "GetHandleInformation", "GetMailslotInfo", "GetModuleFileName", "GetModuleHandle", "GetPriorityClass", "GetProcessShutdownParameters", "GetProcessTimes", "GetProcessWorkingSetSize", "GetSartupInfo", "GetThreadPriority", "GetTheardTimes", "GetWindowThreadProcessId", "LoadLibrary", "LoadLibraryEx", "LoadModule", "MsgWaitForMultipleObjects", "SetPriorityClass", "SetProcessShutdownParameters", "SetProcessWorkingSetSize", "SetThreadPriority", "ShellExecute", "TerminateProcess", "WinExec", "AdjustWindowRect", "AnyPopup", "ArrangeIconicWindows", "AttachThreadInput", "BeginDeferWindowPos", "BringWindowToTop", "CascadeWindows", "ChildWindowFromPoint", "ClientToScreen", "CloseWindow", "CopyRect", "DeferWindowPos", "DestroyWindow", "DrawAnimatedRects", "EnableWindow", "EndDeferWindowPos", "EnumChildWindows", "EnumThreadWindows", "EnumWindows", "EqualRect", "FindWindow", "FindWindowEx", "FlashWindow", "GetActiveWindow", "GetCapture", "GetClassInfo", "GetClassLong", "GetClassName", "GetClassWord", "GetClientRect", "GetDesktopWindow", "GetFocus", "GetForegroundWindow", "GetLastActivePopup", "GetParent", "GetTopWindow", "GetUpdateRect", "GetWindow", "GetWindowContextHelpId", "GetWindowLong", "GetWindowPlacement", "GetWindowRect", "GetWindowText", "GetWindowTextLength", "GetWindowWord", "InflateRect", "IntersectRect", "InvalidateRect", "IsChild", "IsIconic", "IsRectEmpty", "IsWindow", "IsWindowEnabled", "IsWindowUnicode", "IsWindowVisible", "IsZoomed", "LockWindowUpdate", "MapWindowPoints", "MoveWindow", "OffsetRect", "OpenIcon", "PtInRect", "RedrawWindow", "ReleaseCapture", "ScreenToClient", "ScrollWindow", "ScrollWindowEx", "SetActiveWindow", "SetCapture", "SetClassLong", "SetClassWord", "SetFocusAPI", "SetForegroundWindow", "SetParent", "SetRect", "SetRectEmpty", "SetWindowContextHelpId", "SetWindowLong", "SetWindowPlacement", "SetWindowPos", "SetWindowText", "SetWindowWord", "ShowOwnedPopups", "ShowWindow", "ShowWindowAsync", "SubtractRect", "TileWindows", "UnionRect", "UpdateWindow", "ValidateRect", "WindowFromPoint"]
    print(len(feature))









