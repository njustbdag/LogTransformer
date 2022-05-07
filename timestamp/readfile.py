# -*- coding: utf-8 -*-
import random


def read_file(path):
    file = open(path)
    data = file.readlines()
    # 随机读取日志文件中的日志行
    lines = []
    for num, aline in enumerate(data, 2):
        if random.randrange(num):
            continue
        line = aline
        lines.append(line)
    return lines
