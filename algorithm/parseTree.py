import os
import re
from datetime import datetime
import pandas as pd
import numpy as np
from pympler import asizeof
from algorithm import logTree
import hashlib
import operator
import nltk
import math


class Logcluster:
    def __init__(self, eventTree=None, logIDL=None):
        self.eventTree = eventTree
        if logIDL is None:
            logIDL = []
        self.logIDL = logIDL


class ParseTreeNode:
    def __init__(self, childD=None, depth=0, digitOrtoken=None):
        if childD is None:
            childD = dict()
        self.childD = childD
        self.fastChild = None
        self.depth = depth
        self.digitOrtoken = digitOrtoken


class Parser:
    def __init__(self, log_format='', indir='./', outdir='./result/', st=0, rgx=[], span=1):
        self.log_format = log_format
        self.path = indir
        self.savePath = outdir
        self.logName = None
        self.df_log = None
        self.st = st
        self.lcs1 = [],
        self.lcs2 = [],
        self.new_et_vec = {}
        self.regex = rgx
        self.span = span

    def parse(self, logName):
        print('Parsing file: ' + os.path.join(self.path, logName))
        start_time = datetime.now()
        logID = None
        match_count = 0
        logline_count = 0
        update_count = 0
        try:
            self.logName = logName
            rootNode = ParseTreeNode()
            logCluL = []
            self.load_data()
            for idx, line in self.df_log.iterrows():
                logID = line['LineId']
                start = datetime.now()
                logmessageL = line['Content'].strip().split()
                start = datetime.now()
                logmessageL = self.splitColomn(logmessageL)
                lTree = logTree.LogTree(logmessageL).logToTree()
                start = datetime.now()
                flag, matchCluster, count = self.parseTreeSearch(rootNode, lTree)
                match_count += count
                start = datetime.now()
                if matchCluster is None:
                    if self.tokenIsDigit(lTree.getIndicator()) is True:
                        lTree.nodes[0][lTree.getIndicatorIdx()] = ['<*>']
                    self.checkVerb(lTree)
                    paramLen = self.getParmLen(lTree)
                    newCluster = Logcluster(eventTree=lTree, logIDL=[logID])
                    logCluL.append(newCluster)
                    indicator = lTree.size
                    self.updateParseTree(rootNode, newCluster, indicator, indicator, flag, logCluL)
                if matchCluster is not None:
                    newE = self.getEventTree(lTree, matchCluster.eventTree)
                    matchCluster.logIDL.append(logID)
                    matchCluster.eventTree = newE
                    update_count += 1
                logline_count += 1
                if logline_count % 1000 == 0 or logline_count == len(self.df_log):
                    print('Processed {0:.1f}% of log lines.'.format(logline_count * 100.0 / len(self.df_log)))
        except Exception as e:
            print("ParseError:" + str(e) + "|ErrorLogID:" + str(logID))
        finally:
            self.outputResult(logCluL)
            time = datetime.now() - start_time
            print('Parsing done. [Time taken: {!s} s]'.format(time))
            return time.total_seconds()

    def outputResult(self, logCluL):
        if not os.path.exists(self.savePath):
            os.mkdir(self.savePath)
        logID = None
        try:
            log_templates = [0] * self.df_log.shape[0]
            log_templateids = [0] * self.df_log.shape[0]
            log_parameters = [0] * self.df_log.shape[0]
            df_events = []

            for logCluster in logCluL:
                # print(logCluster.logIDL)
                template_str = self.etToEvents(logCluster.eventTree)
                occurence = len(logCluster.logIDL)
                template_id = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
                for logID in logCluster.logIDL:
                    logID -= 1
                    log_templates[logID] = template_str
                    log_templateids[logID] = template_id
                df_events.append([template_id, template_str, occurence])

            df_event = pd.DataFrame(df_events, columns=['EventId', 'EventTemplate', 'Occurences'])
            self.df_log['EventId'] = log_templateids
            self.df_log['EventTemplate'] = log_templates
            self.df_log['ParameterList'] = log_parameters
            self.df_log["ParameterList"] = self.df_log.apply(self.get_parameter_list, axis=1)
            self.df_log.to_csv(os.path.join(self.savePath, self.logName + '_structured.csv'), index=False)

            occ_dict = dict(self.df_log['EventTemplate'].value_counts())
            df_event = pd.DataFrame()
            df_event['EventTemplate'] = self.df_log['EventTemplate'].unique()
            df_event['EventId'] = df_event['EventTemplate'].map(
                lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()[0:8])
            df_event['Occurrences'] = df_event['EventTemplate'].map(occ_dict)
            df_event.to_csv(os.path.join(self.savePath, self.logName + '_templates.csv'), index=False,
                            columns=["EventId", "EventTemplate", "Occurrences"])
            print('find events:' + str(len(logCluL)))
            print('output events:' + str(len(occ_dict)))
        except Exception as e:
            print("OutputError:" + str(e) + "|ErrorLogID:" + str(logID))

    def mergeEvent(self, newE, rn, matchCluster):
        indicator = newE.getIndicator()
        newTemplate = self.etToEvents(newE)
        if rn.childD.__contains__(indicator):
            nodeP = rn.childD[indicator]
            logClusters = nodeP.childD
            for logCluster in logClusters:
                if logCluster == matchCluster:
                    continue
                template = self.etToEvents(logCluster.eventTree)
                if operator.eq(newTemplate, template) is True:
                    return logCluster
        return None

    def etToEvents(self, eventTree):
        template_str = eventTree.printTree()
        return template_str

    def checkVerb(self, lt):
        tokens, positions = self.getStaticNodes(lt)
        tags = nltk.pos_tag(tokens)
        for index in positions:
            if 'rdd_' in tags[index][0]: continue
            if 'V' in tags[index][1] or 'GET' in tags[index][0] or 'DELETE' in tags[index][0] or 'POST' in tags[index][
                0] or 'boot' in tags[index][0] or 'wait' in tags[index][0]:
                flag = True
                for currentRex in self.regex:
                    if re.search(currentRex, tags[index][0]) is not None:
                        flag = False
                        break
                if flag:
                    lt.nodes[1][index].append('V')

    def splitColomn(self, logmessageL):
        offset = 0
        for i in range(0, len(logmessageL)):
            token = logmessageL[i + offset]
            if ':' in token and token[len(token) - 1] != ':':
                tokens = list(filter(lambda x: x != '', token.split(':')))
                # don't split timestamp, such as 00:01
                flag = True
                for t in tokens:
                    flag = t.isdigit()
                    if flag is False:
                        break
                if flag is True:
                    continue
                logmessageL.remove(token)
                for j in range(0, len(tokens)):
                    if j < len(tokens) - 1:
                        logmessageL.insert(i + offset, tokens[j] + ':')
                        logmessageL, i, offset = self.splitByEqual(tokens[j] + ':', logmessageL, i, offset)
                    else:
                        logmessageL.insert(i + offset, tokens[j])
                        logmessageL, i, offset = self.splitByEqual(tokens[j], logmessageL, i, offset)
                    i += 1
                offset += len(tokens) - 1
                continue
            logmessageL, i, offset = self.splitByEqual(token, logmessageL, i, offset)
        return logmessageL

    def splitByEqual(self, token, logmessageL, i, offset):
        if '=' in token and token[len(token) - 1] != '=':
            tokens = list(filter(lambda x: x != '', token.split('=')))
            logmessageL.remove(token)
            for j in range(0, len(tokens)):
                if j < len(tokens) - 1:
                    logmessageL.insert(i + offset, tokens[j] + '=')
                else:
                    logmessageL.insert(i + offset, tokens[j])
                i += 1
            offset += len(tokens) - 1
        return logmessageL, i, offset

    def printEventTree(self, rt):
        count = 0
        for childD1 in rt.childD.items():
            childD2 = childD1[1]
            for i in range(len(childD2.childD)):
                print("EventTree" + str(count + 1) + ":", end='')
                count += 1
                tree = childD2.childD[i].eventTree
                tree.printTree()
                print('')

    def getEventTree(self, lt, et):
        verbList = []
        for key, item in et.nodes[1].items():
            if item is None: continue
            if len(item) == 2:
                verbList.append(item[0])
        new_et = logTree.Tree(et.size)
        new_et.addNode(value=et.getIndicator(), index=et.getIndicatorIdx(), depth=0)
        if lt.getIndicatorIdx() == 0:
            offset = 1
        else:
            offset = 0
        lt_lcs = self.lcs1
        et_lcs = self.lcs2
        gap_count = len(lt_lcs)

        for i in range(-1, gap_count):
            if i == -1:
                if -1 in et.nodes[1].keys() or lt_lcs[0][1] != et_lcs[0][1] or 0 in self.new_et_vec.keys():
                    new_et.addNode(value=None, index=-1, depth=1)
                if i + 1 in self.new_et_vec.keys():
                    new_et.addNode(value=self.new_et_vec[i + 1], index=i, depth=2)
            else:
                if lt_lcs[i][0] in verbList:
                    lt_lcs[i][0] = [lt_lcs[i][0], 'V']
                new_et.addNode(value=lt_lcs[i][0], index=i + offset, depth=1)
                if i + 1 in self.new_et_vec.keys():
                    new_et.addNode(value=self.new_et_vec[i + 1], index=i + offset, depth=2)
        return new_et

    def updateParseTree(self, rn, logClust, old_indicator, indicator, flag, logCluL):
        # flag = False: append the new logClust
        # flag = True: remove the old logClust, add the new logClust
        if flag is False:
            if indicator not in rn.childD:
                firstLayerNode = ParseTreeNode(depth=1, digitOrtoken=indicator)
                rn.childD[indicator] = firstLayerNode
            else:
                firstLayerNode = rn.childD[indicator]
            if len(firstLayerNode.childD) == 0:
                firstLayerNode.childD = [logClust]
            else:
                firstLayerNode.childD.append(logClust)

    def getStaticTokens(self, tree):
        tokens = []
        for index, item in tree.nodes[1].items():
            if index == -1: continue
            if len(item) == 1:
                tokens.extend(item)
            if len(item) == 2: tokens.append(item[0])
        return tokens

    def getStaticNodes(self, tree):
        positions = []
        tokens = []
        for index, item in tree.nodes[1].items():
            if index == -1: continue
            if len(item) == 1:
                tokens.extend(item)
                positions.append(index)
            if len(item) == 2:
                tokens.append(item[0])
                positions.append(index)
        return tokens, positions

    def getDigLen(self, tree):
        tokens = self.getStaticTokens(tree)
        count = 0
        for token in tokens:
            if len(re.findall('[0-9]+', token)) > 0:
                count += 1
        return count

    def getParmLen(self, tree):
        tokens = self.getStaticTokens(tree)
        count = 0
        for token in tokens:
            if len(re.findall('[a-zA-Z]', token)) == len(token):
                count += 1
        return count

    def randomSearch(self, tree1, tree2):
        str1 = self.getStaticTokens(tree1)
        str2 = self.getStaticTokens(tree2)
        count = 0
        flag = False
        for token in str1:
            if count >= math.ceil(len(str1) / 3):
                flag = True
                break
            if token in str2:
                count += 1
        if count >= math.ceil(len(str1) / 3):
            flag = True
        return flag

    def fastSearchEvent(self, logClusters, lt):
        contain_verb_clusters = []
        without_verb_clusters = []
        for logClust in logClusters:
            result, contain_verb = self.preCalSim(lt, logClust.eventTree)
            if not result:
                continue
            else:
                flag = self.randomSearch(logClust.eventTree, lt)
                if flag:
                    flag = self.randomSearch(lt, logClust.eventTree)
                    if flag:
                        if contain_verb:
                            contain_verb_clusters.append(logClust)
                        else:
                            without_verb_clusters.append(logClust)
        if len(contain_verb_clusters) > 0:
            ret_logClusters = contain_verb_clusters
        else:
            ret_logClusters = without_verb_clusters + contain_verb_clusters
        return ret_logClusters

    def parseTreeSearch(self, rn, lt):
        """ check the current logTree if or not belongs to an existed logCluster
        """
        retLogClust = None
        flag = False
        if len(rn.childD) == 0:
            return flag, retLogClust, 0
        size = lt.size
        count = 1
        maxSim = -1
        for key in rn.childD.keys():
            # size：the size of the log tree, key：the size of the event tree
            if abs(key - size) < self.span:
                nodeP = rn.childD[key]
                logClusters = nodeP.childD
                # Speed up searching with simple matching
                logClusters = self.fastSearchEvent(logClusters, lt)
                tmp_retLogClust, sim = self.findMatchCluster(logClusters, lt, maxSim)
                if sim > maxSim:
                    retLogClust = tmp_retLogClust
                    maxSim = sim
                count += len(logClusters)
                if retLogClust is not None:
                    flag = True
        return flag, retLogClust, count

    def findMatchCluster(self, logClusters, lt, maxSim):
        """ find the logCluster that the current logTree belongs to
        """
        retLogClust = None
        maxClust = None
        logClust = None
        count = 0
        for logClust in logClusters:
            count += 1
            curSim, lcs11, lcs22, new_et_vec = self.calSimilarity(lt, logClust.eventTree)
            if curSim > maxSim:
                maxClust = logClust
                maxSim = curSim
                self.lcs1, self.lcs2 = lcs11, lcs22
                self.new_et_vec = new_et_vec
        if maxSim > self.st:
            retLogClust = maxClust
        return retLogClust, maxSim

    def calSimilarity(self, lt, et):
        """ calculate the similarity between the current logTree and the logCluster's eventTree
        """
        # lcs1,lcs2: LCS sequences; seq1,seq2: sequences without '<*>'; nonCommon1,nonCommon2: non-LCS sequences
        lcs1, lcs2, seq1, seq2, nonCommon1, nonCommon2 = self.LCStoLIS(lt.nodes[1], et.nodes[1])
        count = len(lcs1)
        minLength = min(len(seq1), len(seq2))
        if minLength == 0 or len(lcs1) == 0:
            simS = 0
        else:
            simS = count / minLength
        if simS == 0:
            new_et_vec = [0.0, 0.0, 0.0, 0.0]
            return simS, lcs1, lcs2, new_et_vec
        simD, new_et_vec = self.calDynamicSim(lt, et, lcs1, lcs2, nonCommon1, nonCommon2)
        beta = 0.5
        sim = (1 + beta * beta) * simS * simD / (beta * beta * simS + simD)
        return sim, lcs1, lcs2, new_et_vec

    # Pre-similarity calculation is performed by whether the current log contains the verb constant in the event
    def preCalSim(self, lt, et):
        flag = True
        contain_verb = False
        for et_key, et_item in et.nodes[1].items():
            if et_item is None: continue
            if len(et_item) == 2:
                contain_verb = True
                flag = False
                for lt_key, lt_item in lt.nodes[1].items():
                    if lt_item is None: continue
                    if et_item[0] in lt_item[0]:
                        flag = True
                if flag is False:
                    return False, contain_verb
        if flag is False:
            return False, contain_verb
        else:
            return True, contain_verb

    def calDynamicSim(self, lt, et, lt_lcs, et_lcs, lt_nonCom, et_nonCom):
        gap_count = len(lt_lcs)
        lt_vec = {}
        et_vec = {}

        # Calculate the gap vector of lt and et respectively
        start = 0
        for i in range(gap_count + 1):
            if i == gap_count:
                lt_index = len(lt.nodes[1]) + 1
            else:
                lt_index = lt_lcs[i][1]
            lt_vec[i] = [0.0, 0.0, 0.0, 0.0]
            lt_vec[i] = self.calDynamicVec(lt, start, lt_index, lt_nonCom, lt_vec[i])
            if i != gap_count: start = lt_lcs[i][1]

        if -1 in et.nodes[1].keys():
            start = -1
        else:
            start = 0

        for i in range(gap_count + 1):
            if i == gap_count:
                et_index = len(et.nodes[1]) + 1
            else:
                et_index = et_lcs[i][1]
            et_vec[i] = [0.0, 0.0, 0.0, 0.0]
            if start == -1:
                et_vec[i] = self.calDynamicVec(et, -1, 0, et_nonCom, et_vec[i])
            et_vec[i] = self.calDynamicVec(et, start, et_index, et_nonCom, et_vec[i])
            if i != gap_count: start = et_lcs[i][1]

        # Compute cosine similarity of gap vectors of trees
        delta_vec = {}
        sum = 0
        new_et_vec = {}
        for i in range(gap_count + 1):
            if np.linalg.norm(lt_vec[i]) == np.linalg.norm(et_vec[i]) and np.linalg.norm(lt_vec[i]) == 0:
                continue
            if np.linalg.norm(lt_vec[i]) == 0 or np.linalg.norm(et_vec[i]) == 0:
                new_et_vec[i] = (lt_vec[i] + et_vec[i]) / 2
                delta_vec[i] = 0
            else:
                delta_vec[i] = lt_vec[i].dot(et_vec[i]) / (np.linalg.norm(lt_vec[i]) * np.linalg.norm(et_vec[i]))
                sum += delta_vec[i]
                new_et_vec[i] = (lt_vec[i] + et_vec[i]) / 2
        if len(delta_vec) != 0:
            simd = sum / len(delta_vec)
        else:
            simd = 1
        return simd, new_et_vec

    def calDynamicVec(self, tree, start, index, nonCom, tree_vec):
        for j in range(start, index):
            if j in nonCom.keys():
                tmp = nonCom[j]
                if not isinstance(nonCom[j], str):
                    tmp = nonCom[j][0]
                vec = self.wordToVect(tmp)
                if isinstance(tree_vec, list):
                    tree_vec += vec
                else:
                    tree_vec = (tree_vec + vec) / 2
            if 2 not in tree.nodes.keys():
                continue
            if j in tree.nodes[2].keys():
                vec = tree.nodes[2][j]
                if isinstance(tree_vec, list):
                    tree_vec += vec
                else:
                    tree_vec = (tree_vec + vec) / 2
        if isinstance(tree_vec, list):
            tree_vec = np.array(tree_vec)
        return tree_vec

    # Adopt the LIS(Optimization algorithm of LCS) to obtain the common tokens
    def LCStoLIS(self, seq1, seq2):
        greedy_dict = {}
        for index, token in seq2.items():
            if index == -1: continue
            if token[0] not in greedy_dict.keys():
                greedy_dict[token[0]] = [index]
            else:
                greedy_dict[token[0]].insert(0, index)
        greedy_seq = []
        for index, token in seq1.items():
            if token[0] in greedy_dict.keys():
                greedy_seq.extend(greedy_dict[token[0]])
        if len(greedy_seq) == 0: return [], [], seq1, seq2, {}, {}
        subSeq = self.calLIS(greedy_seq, seq2)
        LCS = []
        result1 = []
        result2 = []
        nonCommon1 = {}
        nonCommon2 = {}
        index_subSeq = 0

        for index in subSeq:
            LCS.append(seq2[index][0])
        for index, token in seq2.items():
            if index == -1: continue
            if index in subSeq:
                for i in range(index_subSeq, len(seq1)):
                    if token[0] == seq1[i][0]:
                        result1.append([token[0], i])
                        index_subSeq = i + 1
                        break
                    else:
                        nonCommon1[i] = seq1[i][0]
                result2.append([token[0], index])
            else:
                nonCommon2[index] = token[0]
        while index_subSeq < len(seq1):
            nonCommon1[index_subSeq] = seq1[index_subSeq]
            index_subSeq += 1

        return result1, result2, seq1, seq2, nonCommon1, nonCommon2

    def calLIS(self, greedy_seq, seq2):
        subSeq = [greedy_seq[0]]
        index_sub = 0
        for count, index in enumerate(greedy_seq[1:]):
            if count >= 1:
                if seq2[index][0] == seq2[greedy_seq[count]][0]:
                    if index > greedy_seq[count - 1]:
                        subSeq[index_sub] = index
                    continue

            if index > subSeq[index_sub]:
                subSeq.append(index)
                index_sub += 1
            else:
                # binary search
                left = 0
                right = index
                while left < right:
                    mid = int((left + right) / 2)
                    if len(subSeq) == 1:
                        right = mid
                        continue
                    if mid >= len(subSeq):
                        right = len(subSeq) - 1
                        continue
                    if index > subSeq[mid]:
                        left = mid + 1
                    else:
                        right = mid
                subSeq[left] = index
        return subSeq

    # [lower, upper, digit, other]
    def wordToVect(self, word):
        retVal = [0, 0, 0, 0]
        for c in word:
            if c.islower():
                retVal[0] += 1
            elif c.isupper():
                retVal[1] += 1
            elif c.isdigit():
                retVal[2] += 1
            else:
                retVal[3] += 1

        if all(i == 0 for i in retVal):
            return retVal

        if retVal[1] == 0 and retVal[2] == 0 and retVal[3] == 0:
            return np.array([0, 0, 0, 0])
            print('belongs word')

        retVal = np.array(retVal)
        # Euclidean distance
        retVal = retVal / np.linalg.norm(retVal)
        return retVal

    def vecDist(self, vec1, vec2):
        return np.linalg.norm(vec1 - vec2)

    def tokenIsDigit(self, token):
        for c in token:
            if not c.isdigit(): return False
        return True

    def LCV(self, vec1, vec2):
        if (len(vec1) or len(vec2)) == 0:
            return []
        v1 = []
        v2 = []
        for key, value in vec1.items():
            v1.append([key, value])
        for key, value in vec2.items():
            v2.append([key, value])

        lengths = [[0 for j in range(len(v2) + 1)] for i in range(len(v1) + 1)]
        # row 0 and column 0 are initialized to 0 already
        for i in range(len(v1)):
            for j in range(len(v2)):
                if self.vecDist(v1[i][1], v2[j][1]) <= self.LCVt:
                    lengths[i + 1][j + 1] = lengths[i][j] + 1
                else:
                    lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])

        # read the substring out from the matrix
        result = []
        lenOfSeq1, lenOfSeq2 = len(v1), len(v2)
        while lenOfSeq1 != 0 and lenOfSeq2 != 0:
            if lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1 - 1][lenOfSeq2]:
                lenOfSeq1 -= 1
            elif lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1][lenOfSeq2 - 1]:
                lenOfSeq2 -= 1
            else:
                assert self.vecDist(v1[lenOfSeq1 - 1][1], v2[lenOfSeq2 - 1][1]) <= self.LCVt
                result.insert(0, v1[lenOfSeq1 - 1])
                lenOfSeq1 -= 1
                lenOfSeq2 -= 1
        return result

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)

    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """ Function to transform log file to dataframe
        """
        log_messages = []
        linecount = 0
        with open(log_file, 'r', encoding='UTF-8') as fin:
            for line in fin.readlines():
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf

    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def get_parameter_list(self, row):
        parameter_list = []
        eventTemplate = row['EventTemplate'].split(' ')
        content = row['Content'].split(' ')
        for token in content:
            if token not in eventTemplate:
                parameter_list.append(token)
        return parameter_list
