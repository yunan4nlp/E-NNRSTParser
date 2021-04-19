from enum import Enum
from data.Vocab import *


class CODE(Enum):
    REDUCE = 0
    SHIFT = 1
    POP_ROOT = 2
    NO_ACTION = 3

class NUCLEAR(Enum):
    NN = 0
    NS = 1
    SN = 2

nullkey="-NULL-"

class Action:
    ## for hash
    __hash__ = object.__hash__

    def __init__(self, code=-1, nuclear=-1, label=-1, label_str=nullkey):
        self.code = code
        self.nuclear = nuclear
        self.label = label
        self.label_str = label_str

    def set(self, code=-1, nuclear=-1, label=-1, label_str=nullkey):
        self.code = code
        self.nuclear = nuclear
        self.label = label
        self.label_str = label_str

    ## for dic key
    def __hash__(self):
        return hash(str(self.code) + str(self.nuclear) + str(self.label))

    def is_reduce(self):
        return self.code == CODE.REDUCE

    def is_shift(self):
        return self.code == CODE.SHIFT

    def is_finish(self):
        return self.code == CODE.POP_ROOT

    def is_none(self):
        return self.code == CODE.NO_ACTION

    def __eq__(self, other):
        return other.code == self.code and \
               other.nuclear == self.nuclear and \
               other.label == self.label

    def str(self, vocab):
        if self.is_shift():
            return "shift"
        elif self.is_reduce():
            if self.nuclear == NUCLEAR.NN:
                return 'reduce_NN_' + vocab._id2rel[self.label]
            if self.nuclear == NUCLEAR.NS:
                return 'reduce_NS_' + vocab._id2rel[self.label]
            if self.nuclear == NUCLEAR.SN:
                return 'reduce_SN_' + vocab._id2rel[self.label]
        elif self.is_finish():
            return "pop_root"
        else:
            return "no_action"
