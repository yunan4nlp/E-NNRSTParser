from data.Vocab import *
import numpy as np
import torch
from torch.autograd import Variable
from data.Discourse import *
from transition.State import *

def read_corpus(file_path):
    data = []
    with open(file_path, 'r') as infile:
        for info in readDisTree(infile):
            sent_num = len(info) // 2
            origin_sentences, sentences, sentence_tags, sent_types, total_words, total_tags = parseInfo(info[:sent_num])
            doc = Discourse(origin_sentences, sentences, sentence_tags, sent_types, total_words, total_tags)
            doc.tree_str = info[-1]
            doc.other_infos = info[sent_num:-1]
            doc.parseTree(info[-1])
            data.append(doc)
    return data

def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sentences = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield sentences

def get_gold_actions(data, vocab):
    for doc in data:
        for action in doc.gold_actions:
            if action.is_reduce():
                action.label = vocab.rel2id(action.label_str)
    all_actions = []
    states = []
    for idx in range(1024):
        states.append(State())
    all_feats = []
    S = Metric()
    N = Metric()
    R = Metric()
    F = Metric()
    for doc in data:
        start = states[0]
        start.clear()
        start.ready(doc)
        step = 0
        inst_feats = []
        inst_candidate = []
        action_num = len(doc.gold_actions)
        while not states[step].is_end():
            assert step < action_num
            gold_action = doc.gold_actions[step]
            gold_feats = states[step].prepare_index()
            inst_feats.append(deepcopy(gold_feats))
            next_state = states[step + 1]
            states[step].move(next_state, gold_action, doc, vocab)
            step += 1
        all_feats.append(inst_feats)
        all_actions.append(doc.gold_actions)
        assert len(inst_feats) == len(doc.gold_actions)
        result = states[step].get_result(vocab)
        doc.evaluate_labelled_attachments(result, S, N, R, F)
        assert S.bIdentical() and N.bIdentical() and R.bIdentical() and F.bIdentical()
    return all_feats, all_actions

def get_gold_candid(data, vocab):
    states = []
    all_candid = []
    for idx in range(0, 1024):
        states.append(State())
    for doc in data:
        start = states[0]
        start.clear()
        start.ready(doc)
        step = 0
        inst_candid = []
        while not states[step].is_end():
            gold_action = doc.gold_actions[step]
            candid = states[step].get_candidate_actions(vocab)
            inst_candid.append(candid)
            next_state = states[step + 1]
            states[step].move(next_state, gold_action, doc, vocab)
            step += 1
        all_candid.append(inst_candid)
    return all_candid

def inst(data, feats=None, actions=None, candidate=None):
    inst = []
    if feats is not None and actions is not None:
        assert len(data) == len(actions) and len(data) == len(feats) and len(data) == len(candidate)
        for idx in range(len(data)):
            inst.append((data[idx], feats[idx], actions[idx], candidate[idx]))
        return inst
    else:
        for idx in range(len(data)):
            inst.append((data[idx], None, None, None))
        return inst


def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """

    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size)))

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch

def actions_variable(batch, vocab):
    batch_feats = []
    batch_actions = []
    batch_action_indexes = []
    batch_candidate = []
    for data in batch:
        feat = data[1]
        batch_feats.append(feat)
    for data in batch:
        actions = data[2]
        action_indexes = np.zeros(len(actions), dtype=np.int32)
        batch_actions.append(actions)
        for idx in range(len(actions)):
            ac = actions[idx]
            index = vocab.ac2id(ac)
            action_indexes[idx] = index
        batch_action_indexes.append(action_indexes)
    for data in batch:
        candidate = data[3]
        batch_candidate.append(candidate)
    return batch_feats, batch_actions, batch_action_indexes, batch_candidate

def batch_doc_variable(onebatch, vocab, config, token_helper):
    inst_texts = []
    for idx, instance in enumerate(onebatch):
        doc_text = " ".join(instance[0].words)
        inst_texts.append(doc_text)
    doc_input_ids_list, doc_token_type_ids_list, doc_attention_mask_list = token_helper.batch_bert_id(inst_texts, add_special_tokens=False)

    doc_tok_lengths = [len(input_ids) for input_ids in doc_input_ids_list]
    max_doc_tok_len = max(doc_tok_lengths)
    batch_size = len(onebatch)

    doc_input_ids = np.ones([batch_size, max_doc_tok_len], dtype=np.long) * token_helper.pad_token_id()
    doc_token_type_ids = np.zeros([batch_size, max_doc_tok_len], dtype=np.long)
    doc_attention_mask = np.zeros([batch_size, max_doc_tok_len], dtype=np.long)

    for idx, input_ids in enumerate(doc_input_ids_list):
        for idy, id in enumerate(input_ids):
            doc_input_ids[idx, idy] = doc_input_ids_list[idx][idy]
            doc_token_type_ids[idx, idy] = doc_token_type_ids_list[idx][idy]
            doc_attention_mask[idx, idy] = doc_attention_mask_list[idx][idy]

    doc_input_ids = torch.tensor(doc_input_ids)
    doc_token_type_ids = torch.tensor(doc_token_type_ids)
    doc_attention_mask = torch.tensor(doc_attention_mask)
    return doc_input_ids, doc_token_type_ids, doc_attention_mask


def batch_doc2edu_variable(onebatch, vocab, config, token_helper):

    batch_EDU_index_list = []
    for idx, instance in enumerate(onebatch):
        EDU_texts = []
        for idy, EDU in enumerate(instance[0].EDUs):
            text = " ".join(EDU.words)
            EDU_texts.append(text)
        EDU_tokens_list = token_helper.batch_text2tokens(EDU_texts)
        start = 0
        end = 0
        EDU_index_list = []
        for idy, EDU_tokens in enumerate(EDU_tokens_list):
            end += len(EDU_tokens)
            index_list = []
            for idz in range(start, end):
                index_list.append(idz)
            start += len(EDU_tokens)
            EDU_index_list.append(index_list)
        batch_EDU_index_list.append(EDU_index_list)

    batch_size = len(onebatch)
    edu_lengths = [len(instance[0].EDUs) for instance in onebatch]
    max_edu_num = max(edu_lengths)
    max_EDU_tok_len = max([len(EDU_tokens) for EDU_tokens_list in batch_EDU_index_list for EDU_tokens in EDU_tokens_list])

    EDU_offset_index = np.zeros([batch_size, max_edu_num, max_EDU_tok_len], dtype=np.long)
    batch_denominator = np.zeros([batch_size, max_edu_num, max_EDU_tok_len], dtype=np.float32)
    for idx, EDU_tokens_list in enumerate(batch_EDU_index_list):
        for idy, EDU_tokens in enumerate(EDU_tokens_list):
            for idz, tok in enumerate(EDU_tokens):
                EDU_offset_index[idx, idy, idz] = batch_EDU_index_list[idx][idy][idz]
                batch_denominator[idx, idy, idz] = float(1 / len(batch_EDU_index_list[idx][idy]))

    EDU_offset_index = torch.tensor(EDU_offset_index)
    batch_denominator = torch.tensor(batch_denominator)
    return EDU_offset_index, batch_denominator


def batch_bert_variable(onebatch, vocab, config, token_helper):
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []

    for idx, instance in enumerate(onebatch):
        inst_texts = []
        for idy, EDU in enumerate(instance[0].EDUs):
            text = " ".join(EDU.words[:config.max_edu_len])
            inst_texts.append(text)
        input_ids, token_type_ids, attention_mask = token_helper.batch_bert_id(inst_texts)
        input_ids_list.append(input_ids)
        token_type_ids_list.append(token_type_ids)
        attention_mask_list.append(attention_mask)

    batch_size = len(onebatch)

    edu_lengths = [len(instance[0].EDUs) for instance in onebatch]
    max_edu_num = max(edu_lengths)
    tok_lengths = [len(token_ids) for input_ids in input_ids_list for token_ids in input_ids]
    max_tok_len = max(tok_lengths)

    batch_input_ids = np.ones([batch_size, max_edu_num, max_tok_len], dtype=np.long) * token_helper.pad_token_id()
    batch_token_type_ids = np.zeros([batch_size, max_edu_num, max_tok_len], dtype=np.long)
    batch_attention_mask = np.zeros([batch_size, max_edu_num, max_tok_len], dtype=np.long)

    batch_denominator = np.zeros([batch_size, max_edu_num, max_tok_len], dtype=np.float32)
    batch_cls_index = np.zeros([batch_size, max_edu_num], dtype=np.long)

    for idx in range(batch_size):
        edu_num = len(input_ids_list[idx])
        for idy in range(edu_num):
            tok_len = len(input_ids_list[idx][idy])
            for idz in range(tok_len):
                batch_input_ids[idx, idy, idz] = input_ids_list[idx][idy][idz]
                batch_token_type_ids[idx, idy, idz] = token_type_ids_list[idx][idy][idz]
                batch_attention_mask[idx, idy, idz] = attention_mask_list[idx][idy][idz]

                batch_denominator[idx, idy, idz] = 1 / tok_len
            batch_cls_index[idx, idy] = len(input_ids_list[idx][idy]) - 1

    batch_input_ids = torch.tensor(batch_input_ids)
    batch_token_type_ids = torch.tensor(batch_token_type_ids)
    batch_attention_mask = torch.tensor(batch_attention_mask)
    batch_cls_index = torch.tensor(batch_cls_index)
    batch_denominator = torch.tensor(batch_denominator)

    return batch_input_ids, batch_token_type_ids, batch_attention_mask, edu_lengths, batch_cls_index, batch_denominator

