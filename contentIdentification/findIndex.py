# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
import level
import timestamp.timestamp as timestamp

input_dir = '../logs/'  # The input directory of log file

benchmark_settings = {
    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
        'minEventCount': 2,
        'merge_percent': 0.5
    },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_2k.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        'minEventCount': 2,
        'merge_percent': 0.4
    },

    'Spark': {
        'log_file': 'Spark/Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
        'minEventCount': 2,
        'merge_percent': 0.4
    },

    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?'],
        'minEventCount': 2,
        'merge_percent': 0.4
    },

    'BGL': {
        'log_file': 'BGL/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'regex': [r'core\.\d+'],
        'minEventCount': 2,
        'merge_percent': 0.5
    },

    'HPC': {
        'log_file': 'HPC/HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'regex': [r'=\d+'],
        'minEventCount': 5,
        'merge_percent': 0.4
    },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        'minEventCount': 2,
        'merge_percent': 0.4
    },

    'Windows': {
        'log_file': 'Windows/Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'regex': [r'0x.*?\s'],
        'minEventCount': 2,
        'merge_percent': 0.4
    },

    'Linux': {
        'log_file': 'Linux/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}'],
        'minEventCount': 2,
        'merge_percent': 0.6
    },

    'Andriod': {
        'log_file': 'Andriod/Andriod_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b'],
        'minEventCount': 2,
        'merge_percent': 0.6
    },

    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_2k.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'regex': [],
        'minEventCount': 2,
        'merge_percent': 0.6
    },

    'Apache': {
        'log_file': 'Apache/Apache_2k.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        'minEventCount': 2,
        'merge_percent': 0.4
    },

    'Proxifier': {
        'log_file': 'Proxifier/Proxifier_2k.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
        'regex': [r'<\d+\s?sec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B'],
        'minEventCount': 2,
        'merge_percent': 0.4
    },

    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],
        'minEventCount': 10,
        'merge_percent': 0.7
    },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\d+'],
        'minEventCount': 6,
        'merge_percent': 0.5
    },

    'Mac': {
        'log_file': 'Mac/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'regex': [r'([\w-]+\.){2,}[\w-]+'],
        'minEventCount': 2,
        'merge_percent': 0.6
    }
}

class FindIndex:
    def __init__(self):
        pass

    def find_index(self):
        index_setting = {}
        for dataset, setting in benchmark_settings.items():
            index_setting[dataset] = {}
            index_setting[dataset]['timestamp_index'] = []
            index_setting[dataset]['level_index'] = []
            print('\n===generate the logformat of %s===' % dataset)
            indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
            log_file = os.path.basename(setting['log_file'])
            # 对于每个日志进行10次抽样
            all_index = []
            for x in range(10):
                tokens, timestamp_index = timestamp.timestamp(2, os.path.join(indir, log_file))
                # 需要修改
                # index_setting[dataset]['timestamp_index'] = timestamp_index
                index_setting[dataset]['timestamp_index'].append(timestamp_index)
                df = self.log_to_df(tokens)
                count, level_index = self.tokenSetCount(df, dataset, timestamp_index)
                # 需要修改
                # index_setting[dataset]['level_index'] = level_index
                index_setting[dataset]['level_index'].append(level_index)
                # 输出所有index中最右一个值为0的index+1,即为<Content>的首个index
                index = 0
                for k in count.keys():
                    if count[k] == 0 and k > index:
                        index = k
                # print index + 1
                all_index.append(index + 1)
            index_setting[dataset]['timestamp_index'] = max(index_setting[dataset]['timestamp_index'],
                                                            key=index_setting[dataset]['timestamp_index'].count)
            index_setting[dataset]['level_index'] = max(index_setting[dataset]['level_index'],
                                                        key=index_setting[dataset]['level_index'].count)

            content_index = max(all_index, key=all_index.count)

            # timestamp.timestamp(content_index, os.path.join(indir, log_file))
            print (content_index)
            index_setting[dataset]['content_index'] = content_index + 1
        return index_setting

    def log_to_df(self, tokens):
        '''lines = [line for line in open(log_file) if random() <= .01]
        special = r'[#|^|\'|+|,|<|>|=|@|)|(|\\||~|\s]'
        special = r'[\[|\]|\s|=]'
        special = r'[\s|=]'
        '''
        # 原始日志行以DataFrame的格式存储
        df = pd.DataFrame()
        for l in tokens:
            # 使用抽样的第一行构造DataFrame的初始column
            if df.empty is True:
                columns = []
                for i in range(len(l)):
                    columns.append(i)
                df = pd.DataFrame(columns=columns)
                df.loc[len(df)] = l
            # 当tokens的数量大于column的长度时，增加新的column
            elif df.shape[1] < len(l):
                for i in range(df.shape[1], len(l)):
                    columns.append(i)
                    df[i] = ''
                df.loc[len(df)] = l
            else:
                for i in range(len(l), df.shape[1]):
                    l.append('')
                df.loc[len(df)] = l
        return df

    def tokenSetCount(self, df, dataset, timestamp_index):
        count = {}
        level_index = -1
        for indexs in df.columns:
            if indexs in timestamp_index:
                count[indexs] = 0
                continue
            tokens = df[indexs]
            length = []
            ch = []
            level_count = 0
            str_3gram = pd.DataFrame()
            # 判断该列是否都非空，若存在空字符，则必为<Content>
            if len([i for i in df[indexs].tolist() if i != '']) < df.shape[0]:
                count[indexs] = -1
                break
            else:
                column_set = set(df[indexs].tolist())
                count[indexs] = len(column_set)
                for item in column_set:
                    # 排除不同level的影响
                    if item in level.level[dataset]:
                        level_count += df[indexs].tolist().count(item)
                # 当该列都为level时，记录值为0
                if level_count == len(df[indexs].tolist()):
                    count[indexs] = 0
                    level_index = indexs
                    continue
            # 判断该列是否所有的token长度相同，以及判断是否所有的token都包含相同的符号组合
            for i, v in tokens.iteritems():
                # if len(v) not in length:
                length.append(len(v))
                token = list(filter(lambda x: x != '', re.split(r'[a-zA-Z0-9]', v)))
                if token not in ch:
                    ch.append(token)
            # 过滤相同长度的token
            maxCount_length = max(length, key=length.count)
            if length.count(maxCount_length) >= len(df[indexs].tolist()) * 0.8:
                count[indexs] = 0
                continue
            # 过滤包含相同符号的token
            if len(ch) == 1 and len(ch[0]) != 0:
                count[indexs] = 0
                continue
        return count, level_index

FindIndex().find_index()
