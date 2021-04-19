from transition.Action import *

class Node:
    def __init__(self, nuclear=-1, label=-1, edu_start=-1, edu_end=-1, is_validate=False):
        self.nuclear = nuclear
        self.label = label
        self.edu_start = edu_start
        self.edu_end = edu_end
        self.is_validate = is_validate

        self.str = ''

    def clear(self):
        self.nuclear = -1
        self.label = -1
        self.edu_start = -1
        self.edu_end = -1
        self.is_validate = False

        self.str = ''

    def nuclear_str(self):
        if self.nuclear == NUCLEAR.NN:
            return "NN"
        elif self.nuclear == NUCLEAR.SN:
            return "SN"
        elif self.nuclear == NUCLEAR.NS:
            return "NS"
        return nullkey

    def relation_str(self, vocab):
        if self.label == -1:
            return nullkey
        else:
            return vocab._id2rel[self.label]

class AtomFeat:
    def __init__(self):
        self.s0 = Node()
        self.s1 = Node()
        self.s2 = Node()
        self.q0 = Node()

    def getFeat(self):
        return self.s0, self.s1, self.s2, self.q0

