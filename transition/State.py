from transition.Action import *
from transition.AtomFeat import *
import torch
from torch.autograd import Variable
import numpy as np
from data.Discourse import *
from  copy import deepcopy

max_length = 512


class State:
    def __init__(self):
        self._stack = []
        for idx in range(max_length):
            self._stack.append(Node())
        self._stack_size = 0
        self._edu_size = 0
        self._next_index = 0
        self._word_size = 0
        self._is_start = True
        self._is_gold = True
        self._inst = None
        self._pre_state = None
        self._atom_feat = AtomFeat()
        self._pre_action = Action(CODE.NO_ACTION)

    def ready(self, doc):
        self._inst = doc
        self._edu_size = len(self._inst.EDUs)

    def clear(self):
        self._next_index = 0
        self._stack_size = 0
        self._inst = None
        self._is_gold = True
        self._is_start = True
        self._pre_state = None
        self._pre_action = Action(CODE.NO_ACTION)
        self.done_mark()

    def done_mark(self):
        self._stack[self._stack_size].clear()

    def allow_shift(self):
        if self._next_index >= self._edu_size:
            return False
        else:
            return True

    def allow_pop_root(self):
        if self._stack_size == 1 and self._next_index == self._edu_size:
            return True
        else:
            return False

    def allow_reduce(self):
        if self._stack_size >= 2:
            return True
        else:
            return False

    def shift(self, next_state, doc, vocab):
        assert self._next_index < self._edu_size
        next_state._stack_size = self._stack_size + 1
        next_state._next_index = self._next_index + 1
        self.copy_state(next_state)
        top = next_state._stack[next_state._stack_size - 1]
        top.clear()
        top.is_validate = True
        top.edu_start = self._next_index
        top.edu_end = self._next_index
        next_state.done_mark()
        next_state._pre_action.set(CODE.SHIFT)

        nuclear_str = 't'
        label_str = 'leaf'
        start = doc.EDUs[top.edu_start].start
        end = doc.EDUs[top.edu_end].end
        top.str = '( ' + label_str + ' ' + nuclear_str + ' ' + str(start) + ' ' + str(end) + ' )'

    def reduce(self, next_state, nuclear, label, doc, vocab):
        next_state._stack_size = self._stack_size - 1
        next_state._next_index = self._next_index
        self.copy_state(next_state)
        top0 = next_state._stack[self._stack_size - 1]
        top1 = next_state._stack[self._stack_size - 2]
        assert top0.is_validate == True and top1.is_validate == True
        assert top0.edu_start == top1.edu_end + 1
        top1.edu_end = top0.edu_end
        top1.nuclear = nuclear
        top1.label = label

        if nuclear == NUCLEAR.NN:
            nuclear_str = 'c'
        if nuclear == NUCLEAR.NS:
            nuclear_str = 'l'
        if nuclear == NUCLEAR.SN:
            nuclear_str = 'r'
        label_str = vocab.id2rel(label)
        top1.str = '( ' + label_str + ' ' + nuclear_str + ' ' + str(top1.str) + ' ' + str(top0.str) + ' )'
        top0.clear()
        next_state.done_mark()
        next_state._pre_action.set(CODE.REDUCE, nuclear=nuclear, label=label)

    def pop_root(self, next_state):
        assert  self._stack_size == 1 and self._next_index == self._edu_size
        next_state._next_index = self._edu_size
        next_state._stack_size = 0
        self.copy_state(next_state)
        top0 = next_state._stack[self._stack_size - 1]
        assert top0.is_validate == True
        assert top0.edu_start == 0 and top0.edu_end + 1 == len(self._inst.EDUs)
        top0.clear()
        next_state.done_mark()
        next_state._pre_action.set(CODE.POP_ROOT)

    def move(self, next_state, action, doc, vocab):
        next_state._is_start = False
        next_state._is_gold = False
        if action.is_shift():
            self.shift(next_state, doc, vocab)
        elif action.is_reduce():
            self.reduce(next_state, action.nuclear, action.label, doc, vocab)
        elif action.is_finish():
            self.pop_root(next_state)
        else:
            print(" error state ")

    def get_candidate_actions(self, vocab):
        mask = np.array([False]*vocab.ac_size)
        if self.allow_reduce():
            mask = mask | vocab.mask_reduce
        if self.is_end():
            mask = mask | vocab.mask_no_action
        if self.allow_shift():
            mask = mask | vocab.mask_shift
        if self.allow_pop_root():
            mask = mask | vocab.mask_pop_root
        return ~mask

    def copy_state(self, next_state):
        next_state._stack[0:self._stack_size] = deepcopy(self._stack[0:self._stack_size])
        next_state._edu_size = self._edu_size
        next_state._inst = self._inst
        next_state._pre_state = self

    def is_end(self):
        if self._pre_action.is_finish():
            return True
        else:
            return False

    def get_result(self, vocab):
        result = Result()
        state_iter = self
        while not state_iter._pre_state._is_start:
            action = state_iter._pre_action
            pre_state = state_iter._pre_state
            if action.is_reduce():
                assert pre_state._stack_size >= 2
                right_node = pre_state._stack[pre_state._stack_size - 1]
                left_node = pre_state._stack[pre_state._stack_size - 2]

                la = LabelledAttachment(action.nuclear,
                                        vocab._id2rel[action.label],
                                        left_node.edu_start,
                                        right_node.edu_end)
                result.labelled_attachments.append(la)
                '''
                left_subtree = SubTree()
                right_subtree = SubTree()

                left_subtree.edu_start = left_node.edu_start
                left_subtree.edu_end = left_node.edu_end

                right_subtree.edu_start = right_node.edu_start
                right_subtree.edu_end = right_node.edu_end
                if action.nuclear == NUCLEAR.NN:
                    left_subtree.nuclear = nuclear_str
                    right_subtree.nuclear = nuclear_str
                    left_subtree.relation = vocab._id2rel[action.label]
                    right_subtree.relation = vocab._id2rel[action.label]
                elif action.nuclear == NUCLEAR.SN:
                    left_subtree.nuclear = satellite_str
                    right_subtree.nuclear = nuclear_str
                    left_subtree.relation = vocab._id2rel[action.label]
                    right_subtree.relation = span_str
                elif action.nuclear == NUCLEAR.NS:
                    left_subtree.nuclear = nuclear_str
                    right_subtree.nuclear = satellite_str
                    left_subtree.relation = span_str
                    right_subtree.relation = vocab._id2rel[action.label]
                result.subtrees.insert(0, right_subtree)
                result.subtrees.insert(0, left_subtree)
                '''
            state_iter = state_iter._pre_state
        return result

    def prepare_index(self):
        if self._stack_size > 0:
            self._atom_feat.s0 = self._stack[self._stack_size - 1]
        else:
            self._atom_feat.s0 = None
        if self._stack_size > 1:
            self._atom_feat.s1 = self._stack[self._stack_size - 2]
        else:
            self._atom_feat.s1 = None
        if self._stack_size > 2:
            self._atom_feat.s2 = self._stack[self._stack_size - 3]
        else:
            self._atom_feat.s2 = None
        if self._next_index >= 0 and self._next_index < self._edu_size:
            self._atom_feat.q0 = self._next_index
        else:
            self._atom_feat.q0 = None

        return self._atom_feat
