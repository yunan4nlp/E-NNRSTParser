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

def get_gold_actions(data, vocab, config):
    for doc in data:
        for action in doc.gold_actions:
            if action.is_reduce():
                action.label = vocab.rel2id(action.label_str)
    all_actions = []
    states = []
    for idx in range(config.max_state_len):
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

def get_gold_candid(data, vocab, config):
    states = []
    all_candid = []
    for idx in range(0, config.max_state_len):
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

def batch_data_variable(batch, vocab, config):
    batch_size = len(batch)
    max_edu_num = max([len(data[0].EDUs) for data in batch])
    max_edu_len = max([len(edu.words) for data in batch for edu in data[0].EDUs])
    if max_edu_len > config.max_edu_len: max_edu_len = config.max_edu_len

    edu_words = np.zeros((batch_size, max_edu_num, max_edu_len), dtype=int)
    edu_extwords = np.zeros((batch_size, max_edu_num, max_edu_len), dtype=int)
    edu_tags = np.zeros((batch_size, max_edu_num, max_edu_len), dtype=int)
    word_mask = np.zeros((batch_size, max_edu_num, max_edu_len), dtype=int)
    word_denominator = np.ones((batch_size, max_edu_num), dtype=int) * -1
    edu_mask = np.zeros((batch_size, max_edu_num), dtype=int)
    edu_types = np.zeros((batch_size, max_edu_num), dtype=int)

    for idx in range(batch_size):
        doc = batch[idx][0]
        EDUs = doc.EDUs
        edu_num = len(EDUs)
        for idy in range(edu_num):
            edu = EDUs[idy]
            edu_types[idx, idy] = vocab.EDUtype2id(edu.type)
            edu_len = len(edu.words)
            if edu_len > config.max_edu_len: edu_len = config.max_edu_len
            edu_mask[idx, idy] = 1
            word_denominator[idx, idy] = edu_len
            #assert edu_len == len(edu.tags)
            for idz in range(edu_len):
                word = edu.words[idz]
                tag = edu.tags[idz]
                edu_words[idx, idy, idz] = vocab.word2id(word)
                #edu_extwords[idx, idy, idz] = vocab.extword2id(word)
                tag_id = vocab.tag2id(tag)
                edu_tags[idx, idy, idz] = tag_id
                word_mask[idx, idy, idz] = 1

    edu_words = torch.tensor(edu_words, dtype=torch.long)
    edu_extwords = torch.tensor(edu_extwords, dtype=torch.long)
    edu_tags = torch.tensor(edu_tags, dtype=torch.long)
    word_mask = torch.tensor(word_mask, dtype=torch.float)
    word_denominator = torch.tensor(word_denominator, dtype=torch.float)
    edu_mask = torch.tensor(edu_mask, dtype=torch.float)
    edu_types = torch.tensor(edu_types, dtype=torch.long)
    return edu_words, edu_extwords, edu_tags, word_mask, edu_mask, word_denominator, edu_types

def batch_sent2span_offset(batch, config):
    batch_size = len(batch)
    max_sent_len = max([len(sent) for data in batch for sent in data[0].sentences])
    max_edu_num = max([len(data[0].EDUs) for data in batch])
    max_edu_len = max([len(edu.words) for data in batch for edu in data[0].EDUs])
    if config.max_edu_len < max_edu_len: max_edu_len = config.max_edu_len
    index = np.ones((batch_size, max_edu_num, max_edu_len), dtype=int) * (max_sent_len)
    for idx in range(batch_size):
        data = batch[idx]
        sentences = data[0].sentences
        sent_index = []
        for sent_idx, sentence in enumerate(sentences):
            sent_len = len(sentence)
            for sent_idy in range(sent_len):
                sent_index.append(sent_idx * (max_sent_len + 1) + sent_idy)
        edus = data[0].EDUs
        id = 0
        edu_num = len(edus)
        for idy in range(edu_num):
            edu = edus[idy]
            edu_len = len(edu.words[:config.max_edu_len])
            for idz in range(edu_len):
                index[idx, idy, idz] = sent_index[id]
                id += 1
    index = torch.from_numpy(index).view(batch_size, max_edu_num, max_edu_len)
    return index

def batch_pretrain_variable_sent_level(batch, vocab, config, tokenizer):
    batch_size = len(batch)
    max_bert_len = -1
    max_sent_num = max([len(data[0].sentences) for data in batch])
    max_sent_len = max([len(sent) for data in batch for sent in data[0].sentences])
    #if config.max_sent_len < max_sent_len:max_sent_len = config.max_sent_len
    batch_bert_indices = []
    batch_segments_ids = []
    batch_piece_ids = []
    for data in batch:
        sents = data[0].sentences
        doc_bert_indices = []
        doc_semgents_ids = []
        doc_piece_ids = []
        for sent in sents:
            sent = sent[:max_sent_len]
            bert_indice, segments_id, piece_id = tokenizer.bert_ids(' '.join(sent))
            doc_bert_indices.append(bert_indice)
            doc_semgents_ids.append(segments_id)
            doc_piece_ids.append(piece_id)
            assert len(piece_id) == len(sent)
            assert len(bert_indice) == len(segments_id)
            bert_len = len(bert_indice)
            if bert_len > max_bert_len: max_bert_len = bert_len
        batch_bert_indices.append(doc_bert_indices)
        batch_segments_ids.append(doc_semgents_ids)
        batch_piece_ids.append(doc_piece_ids)
    bert_indice_input = np.zeros((batch_size, max_sent_num, max_bert_len), dtype=int)
    bert_mask = np.zeros((batch_size, max_sent_num, max_bert_len), dtype=int)
    bert_segments_ids = np.zeros((batch_size, max_sent_num, max_bert_len), dtype=int)
    bert_piece_ids = np.zeros((batch_size, max_sent_num, max_sent_len, max_bert_len), dtype=float)

    for idx in range(batch_size):
        doc_bert_indices = batch_bert_indices[idx]
        doc_semgents_ids = batch_segments_ids[idx]
        doc_piece_ids = batch_piece_ids[idx]
        sent_num = len(doc_bert_indices)
        assert sent_num == len(doc_semgents_ids)
        for idy in range(sent_num):
            bert_indice = doc_bert_indices[idy]
            segments_id = doc_semgents_ids[idy]
            bert_len = len(bert_indice)
            piece_id = doc_piece_ids[idy]
            sent_len = len(piece_id)
            assert sent_len <= bert_len
            for idz in range(bert_len):
                bert_indice_input[idx, idy, idz] = bert_indice[idz]
                bert_segments_ids[idx, idy, idz] = segments_id[idz]
                bert_mask[idx, idy, idz] = 1
            for idz in range(sent_len):
                for sid, piece in enumerate(piece_id):
                    avg_score = 1.0 / (len(piece))
                    for tid in piece:
                        bert_piece_ids[idx, idy, sid, tid] = avg_score


    bert_indice_input = torch.from_numpy(bert_indice_input)
    bert_segments_ids = torch.from_numpy(bert_segments_ids)
    bert_piece_ids = torch.from_numpy(bert_piece_ids).type(torch.FloatTensor)
    bert_mask = torch.from_numpy(bert_mask)

    return bert_indice_input, bert_segments_ids, bert_piece_ids, bert_mask


def batch_biEDU_bert_variable(onebatch, vocab, config, token_helper):
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []

    for idx, instance in enumerate(onebatch):
        inst_biEDU_texts = []
        for idy, EDU in enumerate(instance[0].EDUs):
            text = " ".join(EDU.words[:config.max_edu_len])
            if idy - 1 >= 0:
                previous_EDU = instance[0].EDUs[idy - 1]
                previous_text = " ".join(previous_EDU.words[:config.max_edu_len])
            else:
                previous_text = token_helper.tokenizer.pad_token

            inst_biEDU_texts.append((previous_text, text))
        input_ids, token_type_ids, attention_mask = token_helper.batch_biEDU_bert_id(inst_biEDU_texts)
        input_ids_list.append(input_ids)
        token_type_ids_list.append(token_type_ids)
        attention_mask_list.append(attention_mask)

    batch_size = len(onebatch)

    edu_lengths = [len(instance[0].EDUs) for instance in onebatch]
    max_edu_num = max(edu_lengths)
    max_tok_len = max([len(token_ids) for input_ids in input_ids_list for token_ids in input_ids])

    batch_input_ids = np.ones([batch_size, max_edu_num, max_tok_len], dtype=np.long) * token_helper.pad_token_id()
    batch_token_type_ids = np.zeros([batch_size, max_edu_num, max_tok_len], dtype=np.long)
    batch_attention_mask = np.zeros([batch_size, max_edu_num, max_tok_len], dtype=np.long)

    for idx in range(batch_size):
        edu_num = len(input_ids_list[idx])
        for idy in range(edu_num):
            tok_len = len(input_ids_list[idx][idy])
            for idz in range(tok_len):
                batch_input_ids[idx, idy, idz] = input_ids_list[idx][idy][idz]
                batch_token_type_ids[idx, idy, idz] = token_type_ids_list[idx][idy][idz]
                batch_attention_mask[idx, idy, idz] = attention_mask_list[idx][idy][idz]

    batch_input_ids = torch.tensor(batch_input_ids)
    batch_token_type_ids = torch.tensor(batch_token_type_ids)
    batch_attention_mask = torch.tensor(batch_attention_mask)

    return batch_input_ids, batch_token_type_ids, batch_attention_mask, edu_lengths

