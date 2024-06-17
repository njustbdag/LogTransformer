# LogTransformer 
 
 
 
This is the basic implementation of our submission in TNSM: **LogTransformer: Transforming IT System Logs into Events using Tree-based Approach**.
- [LogTransformer](#LogTransformer)
  * [Description](#description)
  * [Datasets](#datasets)
  * [Running](#Running)

## Description

`LogTransformer` is an online event extraction approach based on a tree structure. 
Our approach equips a novel tree-style log and event representation for a log message and an event template, respectively, which can accurately differentiate static tokens and dynamic parameter tokens. 
We also design a tree-based parsing method, which fastens the parsing process by reducing the number of candidate event templates filtered by the length span, and accurately identifies the right event template by the tree-based similarity measure.
We implement `CALSAD` and evaluate it on real-world benchmark datasets. The results demonstrate that LogTransformer can effectively cope with the issues of variable-length logs and evolving log templates.


## Datasets

We implemented `LogTransformer` on 16 open source log datasets from [LogHub](https://github.com/logpai/loghub).

### Running

Run `benchmark/benchmark_LogTrf.py`

