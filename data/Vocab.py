from collections import Counter
from data.Discourse import *
from transition.Action import *
import numpy as np
import re

class Vocab(object):
    PAD, UNK = 0, 1
    def __init__(self, word_counter, tag_counter, rel_counter, EDUtype_counter, min_occur_count = 2):
        self._id2word = ['<pad>', '<unk>']
        self._wordid2freq = [10000, 10000]
        self._id2extword = ['<pad>', '<unk>']
        self._id2ac = []
        self._id2tag = ['<pad>', '<unk>']
        self._id2rel = []
        self._id2EDUtype = ['<pad>', '<unk>']
        for word, count in word_counter.most_common():
            if count > min_occur_count:
                self._id2word.append(word)
                self._wordid2freq.append(count)

        for tag, count in tag_counter.most_common():
            self._id2tag.append(tag)

        for type, count in EDUtype_counter.most_common():
            self._id2EDUtype.append(type)

        for rel, count in rel_counter.most_common():
            self._id2rel.append(rel)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
        if len(self._word2id) != len(self._id2word):
            print("serious bug: words dumplicated, please check!")

        self._tag2id = reverse(self._id2tag)
        if len(self._tag2id) != len(self._id2tag):
            print("serious bug: POS tags dumplicated, please check!")

        self._rel2id = reverse(self._id2rel)
        if len(self._rel2id) != len(self._id2rel):
            print("serious bug: relation labels dumplicated, please check!")

        self._EDUtype2id = reverse(self._id2EDUtype)
        if len(self._EDUtype2id) != len(self._id2EDUtype):
            print("serious bug: relation labels dumplicated, please check!")

        print("Vocab info: #words %d, #tags %d, #rels %d, #EDU type %d" % (self.vocab_size, self.tag_size, self.rel_size, self.EDUtype_size))

    def load_pretrained_embs(self, embfile):
        embedding_dim = -1
        word_count = 0
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                if word_count < 1:
                    values = line.split()
                    embedding_dim = len(values) - 1
                word_count += 1
        print('Total words: ' + str(word_count) + '\n')
        print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')

        index = len(self._id2extword)
        embeddings = np.zeros((word_count + index, embedding_dim))
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                self._id2extword.append(values[0])
                vector = np.array(values[1:], dtype='float64')
                embeddings[self.UNK] += vector
                embeddings[index] = vector
                index += 1

        embeddings[self.UNK] = embeddings[self.UNK] / word_count
        embeddings = embeddings / np.std(embeddings)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._extword2id = reverse(self._id2extword)

        if len(self._extword2id) != len(self._id2extword):
            print("serious bug: extern words dumplicated, please check!")

        return embeddings

    def create_action_table(self, all_actions):
        self._id2ac.append(Action(CODE.NO_ACTION))
        ac_counter = Counter()
        for actions in all_actions:
            for ac in actions:
                ac_counter[ac] += 1
        for ac, count in ac_counter.most_common():
            self._id2ac.append(ac)
        reverse = lambda x: dict(zip(x, range(len(x))))
        self._ac2id = reverse(self._id2ac)
        if len(self._ac2id) != len(self._id2ac):
            print("serious bug: actions dumplicated, please check!")
        print("action num: ", len(self._ac2id))
        print("action: ", end=' ')
        self.mask_shift = np.array([False] * self.ac_size)
        self.mask_reduce = np.array([False] * self.ac_size)
        self.mask_pop_root = np.array([False] * self.ac_size)
        self.mask_no_action = np.array([False] * self.ac_size)
        for (idx, ac) in enumerate(self._id2ac):
            if ac.is_shift():
                self.mask_shift[idx] = True
            if ac.is_reduce():
                self.mask_reduce[idx] = True
            if ac.is_finish():
                self.mask_pop_root[idx] = True
            if ac.is_none():
                self.mask_no_action[idx] = True
            print(ac.str(self), end= ', ')
        print()


    def create_pretrained_embs(self, embfile):
        embedding_dim = -1
        word_count = 0
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                if word_count < 1:
                    values = line.split()
                    embedding_dim = len(values) - 1
                word_count += 1
        print('Total words: ' + str(word_count) + '\n')
        print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')

        index = len(self._id2extword) - word_count
        embeddings = np.zeros((word_count + index, embedding_dim))
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                if self._extword2id.get(values[0], self.UNK) != index:
                    print("Broken vocab or error embedding file, please check!")
                vector = np.array(values[1:], dtype='float64')
                embeddings[self.UNK] += vector
                embeddings[index] = vector
                index += 1

        embeddings[self.UNK] = embeddings[self.UNK] / word_count
        embeddings = embeddings / np.std(embeddings)

        return embeddings


    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.UNK) for x in xs]
        return self._word2id.get(xs, self.UNK)

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self._id2word[x] for x in xs]
        return self._id2word[xs]

    def EDUtype2id(self, xs):
        if isinstance(xs, list):
            return [self._EDUtype2id.get(x, self.UNK) for x in xs]
        return self._EDUtype2id.get(xs, self.UNK)

    def id2EDUtype(self, xs):
        if isinstance(xs, list):
            return [self._id2EDUtype[x] for x in xs]
        return self._id2EDUtype[xs]

    def wordid2freq(self, xs):
        if isinstance(xs, list):
            return [self._wordid2freq[x] for x in xs]
        return self._wordid2freq[xs]

    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.UNK) for x in xs]
        return self._extword2id.get(xs, self.UNK)

    def id2extword(self, xs):
        if isinstance(xs, list):
            return [self._id2extword[x] for x in xs]
        return self._id2extword[xs]

    def rel2id(self, xs):
        if isinstance(xs, list):
            return [self._rel2id[x] for x in xs]
        return self._rel2id[xs]

    def id2rel(self, xs):
        if isinstance(xs, list):
            return [self._id2rel[x] for x in xs]
        return self._id2rel[xs]

    def tag2id(self, xs):
        if isinstance(xs, list):
            return [self._tag2id.get(x) for x in xs]
        return self._tag2id.get(xs)

    def ac2id(self, xs):
        if isinstance(xs, list):
            return [self._ac2id.get(x) for x in xs]
        return self._ac2id.get(xs)

    def id2tag(self, xs):
        if isinstance(xs, list):
            return [self._id2tag[x] for x in xs]
        return self._id2tag[xs]

    def id2ac(self, xs):
        if isinstance(xs, list):
            return [self._id2ac[x] for x in xs]
        return self._id2ac[xs]

    @property
    def vocab_size(self):
        return len(self._id2word)

    @property
    def extvocab_size(self):
        return len(self._id2extword)

    @property
    def tag_size(self):
        return len(self._id2tag)

    @property
    def rel_size(self):
        return len(self._id2rel)

    @property
    def ac_size(self):
        return len(self._id2ac)

    @property
    def EDUtype_size(self):
        return len(self._EDUtype2id)

def normalize_to_lowerwithdigit(str):
    str = str.lower()
    str = re.sub(r'\d', '0', str) ### replace digit 2 zero
    return str

def parseInfo(info):
    total_words = []
    total_tags = []
    sentences = []
    origin_sentences = []
    tags = []
    sent_types = []
    start = 0
    end = 0
    for sentence in info:
        words_info = sentence.split(" ")
        type = words_info[-1]
        assert type == '<P>' or type == '<S>'
        sentence = []
        origin_sentence = []
        tag = []
        end = start + len(words_info) - 1
        sent_type = start, end, type
        sent_types.append(sent_type)
        start = (end + 1)
        for info in words_info[:-1]:
            wt = info.split('_')
            assert len(wt) == 2
            origin_w, t = wt
            w = normalize_to_lowerwithdigit(origin_w)
            sentence.append(w)
            origin_sentence.append(origin_w)
            tag.append(t)
            total_words.append(w)
            total_tags.append(t)
        total_words.append(words_info[-1])
        total_tags.append('-NULL-')
        sentences.append(sentence)
        origin_sentences.append(origin_sentence)
        tags.append(tag)
    assert len(sentences) == len(tags) and len(sent_types) == len(tags)
    assert len(total_words) == len(total_tags)
    return origin_sentences, sentences, tags, sent_types, total_words, total_tags

def creatVocab(train_data, min_occur_count):
    word_counter = Counter()
    tag_counter = Counter()
    rel_counter = Counter()
    EDUtype_counter = Counter()
    for inst in train_data:
        for edu in inst.EDUs:
            EDUtype_counter[edu.type] += 1
        for word in inst.words:
            word_counter[word] += 1
        for tag in inst.tags:
            tag_counter[tag] += 1
        for ac in inst.gold_actions:
            if ac.is_reduce():
                rel_counter[ac.label_str] += 1

    return Vocab(word_counter, tag_counter, rel_counter, EDUtype_counter, min_occur_count)
