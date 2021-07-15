from transformers import AutoTokenizer


class TokenHelper(object):
    def __init__(self, bert_vocab_file):
        self.tokenizer = AutoTokenizer.from_pretrained(bert_vocab_file)
        self.key_words = ("[SEP]", "[UNK]", "[PAD]", "[CLS]", "[MASK]")

    def basic_tokenize(self, text):
        return self.tokenizer.basic_tokenizer.tokenize(text, never_split=self.tokenizer.all_special_tokens)

    def batch_text2tokens(self, inst_text):
        for idx, text in enumerate(inst_text):
            inst_text[idx] = text.replace('##', '@@')
        text_list = []
        for idx, text in enumerate(inst_text):
            text_list.append(self.tokenizer.tokenize(text))
        return text_list

    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    def batch_bert_id(self, inst_text, add_special_tokens=True):
        for idx, text in enumerate(inst_text):
            inst_text[idx] = text.replace('##', '@@')
        outputs = self.tokenizer.batch_encode_plus(inst_text, add_special_tokens=add_special_tokens)
        input_ids = outputs.data['input_ids']
        token_type_ids = outputs.data['token_type_ids']
        attention_mask = outputs.data['attention_mask']

        return input_ids, token_type_ids, attention_mask

    def bert_ids(self, text, add_special_tokens=False):
        text = text.replace('##', '@@')
        outputs = self.tokenizer.encode_plus(text, add_special_tokens=add_special_tokens, return_tensors='pt')
        bert_indice = outputs["input_ids"].squeeze(0)
        segments_id = outputs["token_type_ids"].squeeze(0)

        list_bert_indice = [idx.item() for idx in bert_indice]
        list_segments_id = [idx.item() for idx in segments_id]

        org_tokens = text.split()
        basic_tokens = self.basic_tokenize(text)
        bert_tokens = self.tokenizer.convert_ids_to_tokens(list_bert_indice)

        start_bert_id, bert_len, basic_token_len = 0, len(bert_tokens), len(basic_tokens)
        if bert_tokens[bert_len - 1] == "[SEP]": bert_len = bert_len - 1

        list_basic_id = []
        if add_special_tokens:
            start_bert_id = 1
        else:
            start_bert_id = 0


        for idx in range(basic_token_len):
            # basic token ==> UNK, one-one map
            if bert_tokens[start_bert_id] == "[UNK]":
                list_basic_id.append([start_bert_id])
                start_bert_id += 1
                continue

            cur_basic_token, cur_token_len = basic_tokens[idx], len(basic_tokens[idx])
            end_bert_id = start_bert_id
            sub_token = ""
            while end_bert_id < bert_len and len(sub_token) < cur_token_len:
                cur_sub_token = bert_tokens[end_bert_id]
                if cur_sub_token.startswith("##"): cur_sub_token = cur_sub_token[2:]
                sub_token = sub_token + cur_sub_token
                end_bert_id += 1

            if len(sub_token) == cur_token_len:
                cur_pieces = [piece_id for piece_id in range(start_bert_id, end_bert_id)]
                start_bert_id = end_bert_id
                list_basic_id.append(cur_pieces)
            else:
                print("bug here, basic token, not matched")

            if sub_token != cur_basic_token:
                print("please check, bert tokenizer changes something")

        # basic tokens to org tokens
        list_piece_id = list()
        # add root [CLS] here
        if add_special_tokens:
            list_piece_id.append([0])
        org_token_len = len(org_tokens)
        start_basic_id = 0

        for idx in range(org_token_len):
            cur_org_token, cur_token_len = org_tokens[idx], len(org_tokens[idx])

            sub_basic_tokens = self.basic_tokenize(cur_org_token)
            end_basic_id = start_basic_id + len(sub_basic_tokens)

            sub_token, cur_pieces = "", []
            for basic_id in range(start_basic_id, end_basic_id):
                dist = basic_id - start_basic_id
                cur_sub_token = basic_tokens[basic_id]
                if sub_basic_tokens[dist] != cur_sub_token:
                    print("error: tokenizer crosses org tokens, (org basic1 basic2) = (%s %s  %s)" %
                          (cur_org_token, cur_sub_token, sub_basic_tokens[dist]))
                sub_token = sub_token + cur_sub_token
                cur_pieces = cur_pieces + list_basic_id[basic_id]

            start_basic_id = end_basic_id
            list_piece_id.append(cur_pieces)

            # correct, some special cases, length not match
            # if len(sub_token) != cur_token_len:
            #     print("warning: bert tokenizer changes %s to %s, length not match, %d to %d" % \
            #           (cur_org_token, sub_token, cur_token_len, len(sub_token)))
            # elif sub_token != cur_org_token:
            #     print("warning: bert tokenizer changes %s to %s" % (cur_org_token, sub_token))

        return list_bert_indice, list_segments_id, list_piece_id
