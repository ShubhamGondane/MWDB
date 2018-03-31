from copy import deepcopy
import math


class Tree:
    def __init__(self):
        self.attr = None
        self.vals = []
        self.subTrees = []
        self.final_label = None

    def chooseBest(self, label_count):
        best = list(label_count)[0]
        for lbl in label_count:
            if label_count[lbl] > label_count[best]:
                best = lbl
        self.final_label = best

    def getRelevantSubTree(self, val):
        for i in range(len(self.vals)):
            if val == self.vals[i]:
                return self.subTrees[i]
        return None


class decisionTree:

    def __init__(self):
        self.data = []
        self.labels = []
        self.root = None

    def train(self, data, labels):
        self.data = data
        self.labels = labels
        numAtrributes = len(data[0])

        pAttr = list(range(numAtrributes))
        rlist = list(range(len(data)))
        self.root = Tree()
        self.createTree(self.root, pAttr, rlist)
        self.data = []
        self.labels = []

    def createTree(self, tree, pAttr, rlist):
        first_label = self.labels[rlist[0]]
        for r in range(1, len(rlist)):
            i = rlist[r]
            lbl = self.labels[i]
            if lbl != first_label:
                break
            if r == (len(rlist) - 1):
                tree.final_label = first_label
                return

        if len(pAttr) == 0:
            tree.chooseBest(self.getLabelCount(rlist))
            return

        gains = [0 for i in range(len(pAttr))]
        for i in  range(len(pAttr)):
            a = pAttr[i]
            gains[i] = self.gain(a, rlist)

        maxGain = 0
        for i in range(len(gains)):
            if gains[i] > gains[maxGain]:
                maxGain = i

        if gains[maxGain] == 0:
            tree.chooseBest(self.getLabelCount(rlist))

        tree.attr = pAttr[maxGain]
        del pAttr[maxGain]

        vals_dist = self.valDistribution(tree.attr, rlist)

        tree.vals = list(vals_dist)
        tree.subTrees = [Tree() for i in  range(len(tree.vals))]

        for i in range(len(tree.vals)):
            self.createTree(tree.subTrees[i], deepcopy(pAttr), vals_dist[tree.vals[i]])

    def gain(self, a, rlist):
        return self.entropy(rlist) + self.expectedEntropy(a, rlist)

    def entropy(self, rlist):
        labels = self.labels
        total = 0.0
        label_count = self.getLabelCount(rlist)

        for lbl in  label_count:
            p = self.getLabelProportion(lbl, label_count, len(rlist))
            total += -1 * p #* math.log(p, 2)
        return total

    def expectedEntropy(self, a, rlist):
        total = 0.0
        vals = self.valDistribution(a, rlist)

        for val in vals:
            sub = vals[val]
            s_v = len(sub)
            sub_entropy = self.entropy(sub)
            total += -1 * s_v / len(rlist) * sub_entropy

        return total

    def valDistribution(self, a, rlist):
        vals = {}

        for i in rlist:
            p = self.data[i]
            val = p[a]
            if not val in vals:
                vals[val] = []
            vals[val].append(i)

        return vals

    def getLabelCount(self, rlist):
        label_count = {}

        for i in rlist:
            lbl = self.labels[i]
            if not lbl in label_count:
                label_count[lbl] = 0
            label_count[lbl] += 1

        return label_count

    def getLabelProportion(self, lbl, label_count, numPoints):
        if not lbl in label_count:
            return 0.0
        return label_count[lbl] / numPoints

    def classify(self, x):
        tree = self.root
        while True:
            if tree.final_label is not None:
                return tree.final_label
            x_val = x[tree.attr]
            tree = tree.getRelevantSubTree(x_val)