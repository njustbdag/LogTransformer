# -*- coding: utf-8 -*-
# 字典定义

import re
import timestamp.get as get
import time
from random import random
import timestamp.is_xxx as is_xxx


def timestamp(content_index, file):
    dates = [r"^\d{4}-\d{2}-\d{2}$", r"^\d{2}-\d{2}$", r"^\d{2}\.\d{2}$", r"^\d{4}\.\d{2}\.\d{2}$",
             r"^\d{4}/\d{2}/\d{2}$",
             r"^\d{2}/\d{2}/\d{2}$", r"^\d{8}$", r"^\d{6}$"]
    times = [r"^\d{2}:\d{2}:\d{2}$", r"^\d{2}:\d{2}:\d{2}\.(\d{3}|\d{6})$", r"^\d{2}\.\d{2}\.\d{2}\.(\d{3}|\d{6})$",
             r"^\d{2}:\d{2}:\d{2}:(\d{3}|\d{6})$", r"^\d{6}$", r"^\d{2}:\d{2}:\d{2},(\d{3}|\d{6})$",
             r"^\d{2}\.\d{2}\.\d{2},(\d{3}|\d{6})$"]
    dates2 = [r"\d{4}-\d{2}-\d{2}", r"\d{2}-\d{2}", r"\d{2}\.\d{2}", r"\d{4}\.\d{2}\.\d{2}", r"^\d{4}/\d{2}/\d{2}",
              r"\d{2}/\d{2}/\d{2}", r"\d{8}", r"\d{6}"]
    times2 = [r"\d{2}:\d{2}:\d{2}", r"\d{2}:\d{2}:\d{2}\.(\d{3}|\d{6})", r"\d{2}\.\d{2}\.\d{2}\.(\d{3}|\d{6})",
              r"\d{2}:\d{2}:\d{2}:(\d{3}|\d{6})", r"\d{6}", r"\d{2}:\d{2}:\d{2},(\d{3}|\d{6})",
              r"\d{2}\.\d{2}\.\d{2},(\d{3}|\d{6})", r"(\d{1}|\d{2}):(\d{1}|\d{2}):(\d{1}|\d{2}):(\d{1}|\d{2}|\d{3})"]
    connection = ['_', '-']
    comb = []
    for date in dates2:
        for con in connection:
            for t in times2:
                str_1 = '^' + date + con + t + '$'
                comb.append(str_1)
    # print(len(comb),len(dates)*len(times)*len(connection))
    months = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8,
              'September': 9, 'October': 10, 'November': 11, 'December': 12,
              'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11,
              'Dec': 12
              }
    weeks = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday ': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7,
             'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu ': 4, 'Fri': 5, 'Sat': 6, 'Sun': 7,
             }

    # data = readfile.read_file(file)
    i = 0
    tokens_dict = []  # 保存可能为时间戳的token位置和字段
    try:
        lines = [line for line in open(file) if random() <= .01]
        tokens = []
        for line_index, line in enumerate(lines):
            j = 0
            tokens_dict.append({})
            special = r'[\s|=]'
            '''
            key_words = list(filter(lambda x: x != '', re.split(special, line.strip())))[0:content_index]
            key_words = list(filter(lambda x: x != '', re.split(special, line.strip())))
            '''
            # 分割[]内的不定长token
            multi = re.findall(r'\[(.*?)\]', line.strip())
            line = re.sub(r'\[(.*?)\]', 'multi', line.strip())
            key_words = list(filter(lambda x: x != '', re.split(special, line.strip())))
            multi_count = 0
            for idx, item in enumerate(key_words):
                if item == 'multi':
                    key_words[idx] = multi[multi_count]
                    multi_count += 1

            for word in range(len(key_words)):
                key_words[word] = key_words[word].strip(',[]')
            for word in key_words:
                if (re.search(r'[a-z]', word) or re.search(r'[A-Z]',
                                                           word)) and word in months.keys() or word in weeks.keys():
                    tokens_dict[i].update({j: word})
                if not (re.search(r'[a-z]', word) or re.search(r'[A-Z]', word)) and re.search(r'\d', word):
                    tokens_dict[i].update({j: word})
                j += 1
            i += 1
        index = []  # 时间戳出现的token位置
        index = list(tokens_dict[0].keys())
        for item in tokens_dict:
            for id in index:
                if id not in item.keys():
                    index.remove(id)

        for line in lines:
            year = -1
            month = -1
            day = -1
            week = -1
            hour = -1
            minute = -1
            second = -1
            msecond = -1
            month_ = -1
            week_ = -1
            day_ = -1
            special = r'[\[\]\s|=]'
            special = r'[\s|=]'
            # 分割[]内的不定长token
            multi = re.findall(r'\[(.*?)\]', line.strip())
            line = re.sub(r'\[(.*?)\]', 'multi', line.strip())
            key_words = list(filter(lambda x: x != '', re.split(special, line.strip())))
            multi_count = 0
            for i, item in enumerate(key_words):
                if item == 'multi':
                    key_words[i] = '['+multi[multi_count]+']'
                    multi_count += 1
            month_flag = False
            day_flag = False
            for word in index:
                # key_words[word] = key_words[word].strip(',[]')
                # flag用于控制循环
                flag = False
                '''month_flag = False
                day_flag = False'''
                # 根据显式的关键字得到月
                if months.get(key_words[word]) is not None:
                    month_ = months[key_words[word]]
                    continue
                # 根据显式的关键字的得到星期几
                if weeks.get(key_words[word]) is not None:
                    week_ = weeks[key_words[word]]
                    continue
                # 匹配单独的年份
                if re.match('^\\d{4}$', key_words[word]) is not None and year == -1:
                    if is_xxx.is_year(int(key_words[word])):
                        year = int(key_words[word])
                if re.match('^\\d{1,2}$', key_words[word]) is not None and word > 0 and months.get(
                        key_words[word - 1]) is not None:
                    day_ = int(key_words[word])  # 针对 Jan 21 的格式
                    continue
                #  Unix时间戳
                if (month_flag is False or day_flag is False) and re.match('^\\d{10}$', key_words[word]):

                    all = time.gmtime(int(key_words[word]))

                    if is_xxx.is_year(all[0]):
                        year, month, day, hour, minute, second, week, _, _ = all
                        month_flag = True
                        day_flag = True
                # 匹配年月日
                if month_flag is False:
                    for i in range(len(dates)):
                        if re.match(dates[i], key_words[word]) is not None:
                            date_list = get.get_ymd(key_words[word])
                            if is_xxx.is_valid(date_list):
                                year, month, day = date_list
                                flag = True
                                month_flag = True
                                break
                if flag:
                    continue

                # 匹配时间
                if day_flag is False:
                    for i in range(len(times)):
                        if re.match(times[i], key_words[word]) is not None:
                            time_list = get.get_hmsm(key_words[word])
                            if is_xxx.is_valid(time_list):
                                hour, minute, second, msecond = time_list
                                flag = True
                                day_flag = True
                                break
                if flag:
                    continue
                if month_flag is False or day_flag is False:
                    for i in range(len(comb)):
                        if re.match(comb[i], key_words[word]) is not None:
                            all_list = get.get_all(key_words[word])
                            if is_xxx.is_valid(all_list):
                                year, month, day, hour, minute, second, msecond = all_list
                                break

            if month_ != -1:
                month = month_
            if week_ != -1:
                week = week_
            if day_ != -1:
                day = day_

            '''print(str(year) + '/' + str(month) + '/' + str(day) + ' week:' + str(week) + ' time:' + str(
                hour) + ':' + str(
                minute) + ':' + str(second) + '.' + str(msecond))
            '''
            tokens.append(key_words)
        return tokens, index
    except IOError as e:
        print(e)
