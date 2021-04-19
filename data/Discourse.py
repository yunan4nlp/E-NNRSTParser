from transition.Action import *

nuclear_str = "NUCLEAR"
satellite_str = "SATELLITE"
span_str = "SPAN"

class Discourse:
    def __init__(self, origin_sentences, sentences, sentences_tags, sent_types, total_words, total_tags):
        self.origin_sentences = origin_sentences
        self.sentences = sentences
        self.sentences_tags = sentences_tags

        self.words = []
        for sentence in self.sentences:
            for word in sentence:
                self.words.append(word)

        self.tags = []
        for sentence_tag in self.sentences_tags:
            for tag in sentence_tag:
                self.tags.append(tag)

        self.sent_types = sent_types
        self.total_words = total_words
        self.total_tags = total_tags
        self.EDUs = []
        self.gold_actions = []
        self.result = Result()
    

    def evaluate_labelled_attachments(self, other, S, N, R, F):
        S.overall_label_count += len(self.result.labelled_attachments)
        S.predicated_label_count += len(other.labelled_attachments)
        for p_la in other.labelled_attachments:
            for g_la in self.result.labelled_attachments:
                if p_la.spanEqual(g_la):
                    S.correct_label_count += 1
                    break
        N.overall_label_count += len(self.result.labelled_attachments)
        N.predicated_label_count += len(other.labelled_attachments)
        for p_la in other.labelled_attachments:
            for g_la in self.result.labelled_attachments:
                if p_la.nuclearEqual(g_la):
                    N.correct_label_count += 1
                    break
        R.overall_label_count += len(self.result.labelled_attachments)
        R.predicated_label_count += len(other.labelled_attachments)
        for p_la in other.labelled_attachments:
            for g_la in self.result.labelled_attachments:
                if p_la.relationEqual(g_la):
                    R.correct_label_count += 1
                    break
        F.overall_label_count += len(self.result.labelled_attachments)
        F.predicated_label_count += len(other.labelled_attachments)
        for p_la in other.labelled_attachments:
            for g_la in self.result.labelled_attachments:
                if p_la.fullEqual(g_la):
                    F.correct_label_count += 1
                    break

    def parseTree(self, tree_str):
        buffer = tree_str.strip().split(" ")
        buffer_size = len(buffer)
        step = 0
        subtree_stack = [] # edu index
        op_stack = []
        relation_stack = []
        action_stack = []
        while True:
            assert step <= buffer_size
            if step == buffer_size:
                break
            if buffer[step] == "(":
                op_stack.append(buffer[step])
                relation_stack.append(buffer[step + 1])
                action_stack.append(buffer[step + 2])
                if buffer[step + 1] == 'leaf' and buffer[step + 2] == 't':
                    start = int(buffer[step + 3])
                    end = int(buffer[step + 4])
                    step += 2
                step += 3
            elif buffer[step] == ")":
                action = action_stack[-1]
                if action == 't':
                    for sent_type in self.sent_types:
                        assert len(sent_type) == 3
                        if start >= sent_type[0] and end <= sent_type[1]:
                            e = EDU(start, end, sent_type[2])
                            edu_start = len(self.EDUs)
                            edu_end = len(self.EDUs)
                            subtree_stack.append([edu_start, edu_end])
                            self.EDUs.append(e)
                            assert relation_stack[-1] == "leaf"
                            ac = Action(CODE.SHIFT, -1, -1, relation_stack[-1])
                            self.gold_actions.append(ac)
                            break
                elif action == 'l' or action == 'r' or action == 'c':
                    if action == 'l':
                        nuclear = NUCLEAR.NS
                    if action == 'r':
                        nuclear = NUCLEAR.SN
                    if action == 'c':
                        nuclear = NUCLEAR.NN
                    code = CODE.REDUCE
                    ac = Action(code, nuclear, -1, relation_stack[-1])
                    self.gold_actions.append(ac)

                    assert len(subtree_stack) >= 2
                    l_index = subtree_stack[-2]
                    r_index = subtree_stack[-1]
                    assert l_index[1] + 1 == r_index[0]
                    #left_subtree = SubTree(nullkey, nullkey, l_index[0], l_index[1])
                    #right_subtree = SubTree(nullkey, nullkey, r_index[0], r_index[1])

                    la = LabelledAttachment(nuclear, relation_stack[-1], l_index[0], r_index[1])
                    self.result.labelled_attachments.append(la)

                    '''
                    if action == "l": #NS
                        left_subtree.nuclear = nuclear_str
                        left_subtree.relation = span_str
                        right_subtree.nuclear = satellite_str
                        right_subtree.relation = ac.label_str
                    if action == "r": #SN
                        left_subtree.nuclear = satellite_str
                        left_subtree.relation = ac.label_str
                        right_subtree.nuclear = nuclear_str
                        right_subtree.relation = span_str
                    if action == "c": #NN
                        left_subtree.nuclear = nuclear_str
                        left_subtree.relation = ac.label_str
                        right_subtree.nuclear = nuclear_str
                        right_subtree.relation = ac.label_str
                    self.result.subtrees.append(left_subtree)
                    self.result.subtrees.append(right_subtree)
                    '''
                    l_index[1] = r_index[1]
                    subtree_stack.pop()

                relation_stack.pop()
                op_stack.pop()
                action_stack.pop()

                step += 1
        ac = Action(CODE.POP_ROOT)
        self.gold_actions.append(ac)
        assert len(subtree_stack) == 1
        root = subtree_stack[0]
        assert root[0] == 0 and root[1] == len(self.EDUs) - 1
        subtree_stack.pop() # pop root

        #### check stack, all stack empty
        assert op_stack == [] and relation_stack == [] and action_stack == [] and subtree_stack == []
        #### check edu index
        for idx in range(len(self.EDUs)):
            edu = self.EDUs[idx]
            assert edu.start >= 0 and edu.end < len(self.total_words)
            assert edu.start <= edu.end
            if idx < len(self.EDUs) - 1:
                assert edu.end + 1 == self.EDUs[idx + 1].start
        #### initialize edu word and tag
        sum = 0
        for edu in self.EDUs:
            for idx in range(edu.start, edu.end + 1):
                if self.total_tags[idx] != nullkey:
                    edu.words.append(self.total_words[idx])
                    edu.tags.append(self.total_tags[idx])
            sum += len(edu.words)
        assert sum == len(self.words)
        #### check subtree
        #for subtree in self.result.subtrees:
            #assert subtree.relation != nullkey and subtree.nuclear != nullkey


class EDU:
    def __init__(self, start, end, type):
        self.start = start
        self.end = end
        self.type = type
        self.words = []
        self.tags = []

class LabelledAttachment:
    def __init__(self, nuclear=nullkey, relation=nullkey, edu_start=-1, edu_end=-1):
        self.nuclear = nuclear
        self.relation = relation
        self.edu_start = edu_start
        self.edu_end = edu_end

    def spanEqual(self, other):
        return self.edu_start == other.edu_start and \
               self.edu_end == other.edu_end

    def nuclearEqual(self, other):
        return self.edu_start == other.edu_start and \
               self.edu_end == other.edu_end and \
               self.nuclear == other.nuclear

    def relationEqual(self, other):
        return self.edu_start == other.edu_start and \
               self.edu_end == other.edu_end and \
               self.relation == other.relation

    def fullEqual(self, other):
        return self.edu_start == other.edu_start and \
               self.edu_end == other.edu_end and \
               self.nuclear == other.nuclear and \
               self.relation == other.relation
'''
class SubTree:
    def __init__(self, nuclear=nullkey, relation=nullkey, edu_start=-1, edu_end=-1):
        self.nuclear = nuclear
        self.relation = relation
        self.edu_start = edu_start
        self.edu_end = edu_end

    def spanEqual(self, tree):
        return self.edu_start == tree.edu_start and \
               self.edu_end == tree.edu_end

    def nuclearEqual(self, tree):
        return self.edu_start == tree.edu_start and \
               self.edu_end == tree.edu_end and \
               self.nuclear == tree.nuclear

    def relationEqual(self, tree):
        return self.edu_start == tree.edu_start and \
               self.edu_end == tree.edu_end and \
               self.relation == tree.relation

    def fullEqual(self, tree):
        return self.edu_start == tree.edu_start and \
               self.edu_end == tree.edu_end and \
               self.nuclear == tree.nuclear and \
               self.relation == tree.relation
:w
'''

class Result:
    def __init__(self):
        #self.subtrees = []
        self.labelled_attachments = []

class Metric:
    def __init__(self):
        self.overall_label_count = 0
        self.correct_label_count = 0
        self.predicated_label_count = 0

    def reset(self):
        self.overall_label_count = 0
        self.correct_label_count = 0
        self.predicated_label_count = 0

    def bIdentical(self):
        if self.predicated_label_count == 0:
            if self.overall_label_count == self.correct_label_count:
                return True
            return False
        else:
            if self.overall_label_count == self.correct_label_count and \
                    self.predicated_label_count == self.correct_label_count:
                return True
            return False

    def getAccuracy(self):
        if self.overall_label_count + self.predicated_label_count == 0:
            return 1.0
        if self.predicated_label_count == 0:
            return self.correct_label_count*1.0 / self.overall_label_count
        else:
            return self.correct_label_count*2.0 / (self.overall_label_count + self.predicated_label_count)

    def print(self):
        if self.predicated_label_count == 0:
            print("Accuracy:\tP=" + str(self.correct_label_count) + '/' + str(self.overall_label_count))
        else:
            print("Recall:\tP=" + str(self.correct_label_count) + "/" + str(self.overall_label_count) + "=" + str(self.correct_label_count*1.0 / self.overall_label_count), end=",\t")
            print("Accuracy:\tP=" + str(self.correct_label_count) + "/" + str(self.predicated_label_count) + "=" + str(self.correct_label_count*1.0 / self.predicated_label_count), end=",\t")
            print("Fmeasure:\t" + str(self.correct_label_count*2.0 / (self.overall_label_count + self.predicated_label_count)))


def readDisTree(file, vocab=None):
    info = []
    for line in file:
        tok = line.rstrip()
        if tok == '':
            assert len(info) % 2 == 1
            yield info
            info = []
        else:
            info.append(tok)
    if len(info) != 0:
        yield info


