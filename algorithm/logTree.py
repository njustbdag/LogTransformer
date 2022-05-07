import numpy as np


class Tree:
    def __init__(self, size, depth=0):
        self.size = float(size)
        self.nodes = {}
        self.depth = depth

    def addNode(self, value, index, depth):
        if depth not in self.nodes:
            self.nodes[depth] = {}
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list):
            value = value
        self.nodes[depth][index] = value
        if self.depth < depth:
            self.depth = depth

    def getIndicator(self):
        indicatorIdx = list(self.nodes[0].keys())[0]
        return self.nodes[0][indicatorIdx][0]

    def getIndicatorIdx(self):
        indicatorIdx = list(self.nodes[0].keys())[0]
        return indicatorIdx

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

        retVal = np.array(retVal)

        # Euclidean distance
        retVal = retVal / np.linalg.norm(retVal)

        return retVal

    def printTree(self):
        template_str = ''
        if -1 in self.nodes[1].keys():
            offset = -1
        else: offset = 0
        for j in range(offset, len(self.nodes[1])+offset):
            if j != -1:
                template_str += (self.nodes[1][j][0]+' ')
            if self.depth < 2: continue
            if j in self.nodes[2].keys():
                template_str += ('<*>'+' ')
        return template_str.strip()

class LogTree:
    def __init__(self, logMessage):
        self.logMessage = logMessage

    def logToTree(self):
        """  Function to transform logLine to logTree
        """
        lTree = Tree(size=len(self.logMessage))
        segment = []
        logMessage = self.logMessage
        rnIdx = -2
        lTree.addNode(value=logMessage[0], index=-2, depth=0)

        # based on ',' divide logMsg into segment
        pos = 1
        if rnIdx == -2:
            pos = 0
        for i in range(pos, len(logMessage), 1):
            token = logMessage[i]
            if token[len(token) - 1] == ',':
                segment.append(logMessage[pos:i + 1:])
                pos = i + 1
            elif i == len(logMessage) - 1:
                segment.append(logMessage[pos:i + 1:])

        staticIndex = 1
        if rnIdx == -2:
            staticIndex = 0

        for seg in segment:
            currentDep = 1
            if len(seg) == 1:
                token = seg[0]
                lTree.addNode(value=token, index=staticIndex, depth=1)
                staticIndex += 1
            else:
                vec = [0, 0, 0, 0]
                vecIndex = 0
                for i in range(len(seg)):
                    token = seg[i]
                    if '<*>' in token:
                        token = '<*>'
                    if currentDep == 1:
                        lTree.addNode(value=token, index=staticIndex, depth=currentDep)
                        if token[len(token) - 1] == ':' or token[len(token) - 1] == '=':
                            if i+1 < len(seg):
                                if not seg[i+1].isalpha():
                                    vecIndex = staticIndex
                                    currentDep = 2
                        staticIndex += 1
                        continue
                    if currentDep == 2:
                        if isinstance(vec, list): vec = lTree.wordToVect(token)
                        else: vec = (vec + lTree.wordToVect(token))/2
                        if token[len(token) - 1] == ':':
                            currentDep == 2
                        else:
                            lTree.addNode(value=vec, index=vecIndex, depth=currentDep)
                            currentDep = 1
        return lTree
