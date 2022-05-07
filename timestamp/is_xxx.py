# -*- coding: utf-8 -*-
import time


def is_year(a):
    if int(a) < 2000 or int(a) > time.gmtime(int(time.time()))[0]:  # 判断年份是否在2000到当前年份之间
        return False
    else:
        return True


def is_month(a):
    if int(a) < 0 or int(a) > 12:
        return False
    else:
        return True


def is_day(a):
    if int(a) < 0 or int(a) > 31:
        return False
    else:
        return True


def is_hour(a):
    if int(a) < 0 or int(a) > 24:
        return False
    else:
        return True


def is_minute_second(a):
    if int(a) < 0 or int(a) > 59:
        return False
    else:
        return True


# 判断返回的列表是否为有效列表如果全为-1就是无效
def is_valid(list):
    count = 0
    for i in range(len(list)):
        if int(list[i]) == -1:
            count += 1
    if count == len(list):
        return False
    return True
