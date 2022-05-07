# -*- coding: utf-8 -*-
import timestamp.is_xxx as is_xxx
import re
import string


# 根据匹配到的不同正则表达是对数据进行分割得到年月日
def get_ymd(word):
    date_list = [-1, -1, -1]
    if re.match('^\\d{8}$', word) is None and re.match('^\\d{6}$', word) is None:
        date_list = re.split(r'[-\./]', word)
        if len(date_list[0]) == 2 and len(date_list) == 3:
            date_list[0] = int(date_list[0]) + 2000
    if re.match('^\\d{8}$', word) is not None:
        date_list[0] = word[0:4]
        date_list[1] = word[4:6]
        date_list[2] = word[6:8]

    if re.match('^\\d{6}$', word) is not None:
        date_list[0] = int(word[0:2]) + 2000
        date_list[1] = word[2:4]
        date_list[2] = word[4:6]

    # 有些格式中不包含年份，所以列表的长度可能是2耶可能是3
    if len(date_list) == 3:
        if is_xxx.is_year(date_list[0]) is False or is_xxx.is_month(date_list[1]) is False or is_xxx.is_day(
                date_list[2]) is False:
            date_list = [-1, -1, -1]

    if len(date_list) == 2:
        date_list.insert(0, -1)  # 插入年份为-1
        if is_xxx.is_month(date_list[1]) is False or is_xxx.is_day(date_list[2]) is False:
            date_list = [-1, -1, -1]
    return date_list


def get_hmsm(word):
    time_list = [-1, -1, -1, -1]
    if re.match('^\\d{6}$', word) is None:
        time_list = re.split(r'[\.:,]', word)
    else:
        time_list[0] = word[0:2]
        time_list[1] = word[2:4]
        time_list[2] = word[4:6]
    if is_xxx.is_hour(time_list[0]) is False or is_xxx.is_minute_second(
            time_list[1]) is False or is_xxx.is_minute_second(time_list[2]) is False:
        time_list = [-1, -1, -1, -1]
        return time_list
    if len(time_list) == 3:
        time_list.append(-1)
    return time_list


def get_all(word):
    all_list = []
    date_list = [-1, -1, -1]
    time_list = [-1, -1, -1]
    all_list1 = re.split('[_]', word)
    length = len(all_list)
    # length=1表示连接符不是下划线
    if length == 0:
        all_list = re.split('[-]', word)
        length = len(all_list)
        # length=2表示日期信息中没有中划线
        if length == 2:
            date_list = get_ymd(all_list[0])
            time_list = get_hmsm(all_list[1])
            all_list = date_list + time_list
        # length=3表示日期部分也有中划线并且没有年份信息
        if length == 3:
            time_list = get_hmsm(all_list[2])
            if is_xxx.is_month(all_list[0]) and is_xxx.is_day(all_list[1]):
                date_list[1] = all_list[0]
                date_list[2] = all_list[1]
                all_list = date_list + time_list
            else:
                return [-1, -1, -1, -1, -1, -1, -1]
        # length=4表示日期部分也有中划线并且有年份信息
        if length == 4:
            time_list = get_hmsm(all_list[3])
            if is_xxx.is_year(all_list[0]) and is_xxx.is_month(all_list[1]) and is_xxx.is_day(all_list[2]):
                date_list[0] = all_list[0]
                date_list[1] = all_list[1]
                date_list[2] = all_list[2]
                all_list = date_list + time_list
            else:
                return [-1, -1, -1, -1, -1, -1, -1]
    # 用下划线分割后有两项第一项日期第二项时间
    else:
        date_list = get_ymd(all_list[0])
        time_list = get_hmsm(all_list[1])
    return all_list


def filter(text):
    # lowers = text.lower()
    # remove the punctuation using the character deletion step of translate
    '''remove_punctuation_map = dict((ord(char), ' ') for char in ['|'])
    no_punctuation = text.translate(remove_punctuation_map)'''
    trantab = string.maketrans('|', ' ')
    no_punctuation = text.translate(trantab)
    return no_punctuation
