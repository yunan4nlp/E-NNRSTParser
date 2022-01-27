from transformers import AutoTokenizer


class TokenHelper(object):
    def __init__(self, plm_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(plm_dir)
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