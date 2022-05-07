import sys

sys.path.append('../')
import os
import pandas as pd

input_dir = '../logs/'  # The input directory of log file
output_dir = 'LogTransformer_result/'  # The output directory of parsing results
from algorithm import parseTree, evaluator

benchmark_settings = {
    'Andriod': {
        'unevolved_log_file': 'Andriod/Andriod_unevolved.log',
        'evolved_log_file': 'Andriod/Andriod_evolved.log',
        'log_file': 'Andriod/Andriod_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'st': 0.75,
        'span': 2,
    },

    'Apache': {
        'unevolved_log_file': 'Apache/Apache_unevolved.log',
        'evolved_log_file': 'Apache/Apache_evolved.log',
        'log_file': 'Apache/Apache_2k.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        'st': 0.4,
        'span': 1,
    },

    'BGL': {
        'unevolved_log_file': 'BGL/BGL_unevolved.log',
        'evolved_log_file': 'BGL/BGL_evolved.log',
        'log_file': 'BGL/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'st': 0.5,
        'span': 1,
    },

    'Hadoop': {
        'unevolved_log_file': 'Hadoop/Hadoop_unevolved.log',
        'evolved_log_file': 'Hadoop/Hadoop_evolved.log',
        'log_file': 'Hadoop/Hadoop_2k.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
        'st': 0.72,
        'span': 1,
    },

    'HDFS': {
        'unevolved_log_file': 'HDFS/HDFS_unevolved.log',
        'evolved_log_file': 'HDFS/HDFS_evolved.log',
        'log_file': 'HDFS/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'st': 0.4,
        'span': 1,
    },

    'HealthApp': {
        'unevolved_log_file': 'HealthApp/HealthApp_unevolved.log',
        'evolved_log_file': 'HealthApp/HealthApp_evolved.log',
        'log_file': 'HealthApp/HealthApp_2k.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'st': 0.5,
        'span': 1,
    },

    'HPC': {
        'unevolved_log_file': 'HPC/HPC_unevolved.log',
        'evolved_log_file': 'HPC/HPC_evolved.log',
        'log_file': 'HPC/HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'st': 0.56,
        'span': 1,
    },

    'Linux': {
        'unevolved_log_file': 'Linux/Linux_unevolved.log',
        'evolved_log_file': 'Linux/Linux_evolved.log',
        'log_file': 'Linux/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'st': 0.72,
        'span': 1,
    },

    'Mac': {
        'unevolved_log_file': 'Mac/Mac_unevolved.log',
        'evolved_log_file': 'Mac/Mac_evolved.log',
        'log_file': 'Mac/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'st': 0.69,
        'span': 1,
    },

    'OpenSSH': {
        'unevolved_log_file': 'OpenSSH/OpenSSH_unevolved.log',
        'evolved_log_file': 'OpenSSH/OpenSSH_evolved.log',
        'log_file': 'OpenSSH/OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'st': 0.74,
        'span': 1,
    },

    'OpenStack': {
        'unevolved_log_file': 'OpenStack/OpenStack_unevolved.log',
        'evolved_log_file': 'OpenStack/OpenStack_evolved.log',
        'log_file': 'OpenStack/OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'st': 0.75,
        'span': 4,
    },

    'Proxifier': {
        'unevolved_log_file': 'Proxifier/Proxifier_unevolved.log',
        'evolved_log_file': 'Proxifier/Proxifier_evolved.log',
        'log_file': 'Proxifier/Proxifier_2k.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
        'st': 0.55,
        'span': 6,
    },

    'Spark': {
        'unevolved_log_file': 'Spark/Spark_unevolved.log',
        'evolved_log_file': 'Spark/Spark_evolved.log',
        'log_file': 'Spark/Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>',
        'st': 0.4,
        'span': 1,
    },

    'Thunderbird': {
        'unevolved_log_file': 'Thunderbird/Thunderbird_unevolved.log',
        'evolved_log_file': 'Thunderbird/Thunderbird_evolved.log',
        'log_file': 'Thunderbird/Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'st': 0.61,
        'span': 1,
    },

    'Windows': {
        'unevolved_log_file': 'Windows/Windows_unevolved.log',
        'evolved_log_file': 'Windows/Windows_evolved.log',
        'log_file': 'Windows/Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'st': 0.75,
        'span': 1,
    },

    'Zookeeper': {
        'unevolved_log_file': 'Zookeeper/Zookeeper_unevolved.log',
        'evolved_log_file': 'Zookeeper/Zookeeper_evolved.log',
        'log_file': 'Zookeeper/Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'st': 0.63,
        'span': 2,
    }
}

benchmark_settings = {
    'HDFS': {
        'unevolved_log_file': 'HDFS/HDFS_unevolved.log',
        'evolved_log_file': 'HDFS/HDFS_evolved.log',
        'log_file': 'HDFS/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'st': 0.4,
        'span': 1,
    },
}

bechmark_result = []
df_compareParameters = None
for dataset, setting in benchmark_settings.items():
    try:
        print('\n=== Evaluation on %s ===' % dataset)
        indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
        log_file = os.path.basename(setting['log_file'])
        parser = parseTree.Parser(setting['log_format'], indir, output_dir, setting['st'])
        time = parser.parse(log_file, setting['span'])
        precision, recall, F1_measure, randIndex, accuracy, correct_events, df_compareParameters = evaluator.evaluate(
            groundtruth=os.path.join(indir, log_file + '_structured.csv'),
            parsedresult=os.path.join(output_dir, log_file + '_structured.csv')
        )
        bechmark_result.append(
            [log_file, precision, recall, F1_measure, randIndex, accuracy, setting['st'], setting['span'], time])
    except Exception as e:
        print(e)


print('\n=== Overall evaluation results ===')
df_result = pd.DataFrame(bechmark_result,
                         columns=['Dataset', 'Precision', 'Recall', 'F1_measure', 'RandIndex', 'Accuracy', 'st', 'span',
                                  'Time'])
df_result.set_index('Dataset', inplace=True)
print(df_result)
df_result.to_csv('LogTransformer_benchmark_result.csv')
