import sys

sys.path.append('../')

import os
import pandas as pd

input_dir = '../logs/'  # The input directory of log file
output_dir = 'LogTrf_result/'  # The output directory of parsing results
from algorithm import evaluator, parseTree


benchmark_settings = {
    'Andriod': {
        'log_file': 'Andriod/Andriod_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'st': 0.75,
        'span': 2,
        'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b'],

    },

    'Apache': {
        'log_file': 'Apache/Apache_2k.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        'st': 0.55,
        'span': 1,
        'regex': [r'(\d+\.){3}\d+'],

    },

    'BGL': {
        'log_file': 'BGL/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'st': 0.55,
        'span': 1,
        'regex': [r'core\.\d+'],

    },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_2k.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
        'st': 0.65,
        'span': 1,
        'regex': [r'(\d+\.){3}\d+'],

    },

    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'st': 0.55,
        'span': 1,
        'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
    },

    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_2k.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'st': 0.55,
        'span': 1,
        'regex': [],

    },

    'HPC': {
        'log_file': 'HPC/HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'st': 0.56,
        'span': 1,
        'regex': [r'=\d+'],

    },

    'Linux': {
        'log_file': 'Linux/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'st': 0.72,
        'span': 1,
        'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}'],

    },

    'Mac': {
        'log_file': 'Mac/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'st': 0.74,
        'span': 1,
        'regex': [r'([\w-]+\.){2,}[\w-]+'],

    },

    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'st': 0.74,
        'span': 1,
        'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],

    },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'st': 0.75,
        'span': 4,
        'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\d+'],

    },

    'Proxifier': {
        'log_file': 'Proxifier/Proxifier_2k.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
        'st': 0.55,
        'span': 6,
        'regex': [r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'\b[KGTM]?B\b'],

    },

    'Spark': {
        'log_file': 'Spark/Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>',
        'st': 0.55,
        'span': 1,
        'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],

    },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'st': 0.72,
        'span': 1,
        'regex': [r'(\d+\.){3}\d+'],

    },

    'Windows': {
        'log_file': 'Windows/Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'st': 0.76,
        'span': 1,
        'regex': [r'0x.*?\s'],

    },

    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'st': 0.63,
        'span': 2,
        'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?'],
    }
}

benchmark_result = []
df_compareParameters = None
for dataset, setting in benchmark_settings.items():
    try:
        print('\n=== Evaluation on %s ===' % dataset)
        indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
        log_file = os.path.basename(setting['log_file'])
        parser = parseTree.Parser(setting['log_format'], indir, output_dir, setting['st'], setting['regex'],
                                  setting['span'])
        time = parser.parse(log_file)
        precision, recall, F1_measure, randIndex, accuracy = evaluator.evaluate(
            groundtruth=os.path.join(indir, log_file + '_structured.csv'),
            parsedresult=os.path.join(output_dir, log_file + '_structured.csv')
        )
        benchmark_result.append(
            [log_file, precision, recall, F1_measure, randIndex, accuracy, setting['st'], setting['span'], time])
    except Exception as e:
        print(e)

print('\n=== Overall evaluation results ===')
df_result = pd.DataFrame(benchmark_result,
                         columns=['Dataset', 'Precision', 'Recall', 'F1_measure', 'RandIndex', 'Accuracy', 'st', 'span',
                                  'Time'])
df_result.set_index('Dataset', inplace=True)
print(df_result)
