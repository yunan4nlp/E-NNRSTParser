from transformers import BertTokenizer


class BertTokenHelper(object):
    def __init__(self, bert_vocab_file):
        self.tokenizer = BertTokenizer.from_pretrained(bert_vocab_file)
        print("Load bert vocabulary finished")
        self.key_words = ("[SEP]", "[UNK]", "[PAD]", "[CLS]", "[MASK]")

    def basic_tokenize(self, text):
        return self.tokenizer.basic_tokenizer.tokenize(text, never_split=self.tokenizer.all_special_tokens)

    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    def batch_bert_id(self, inst_text):
        for idx, text in enumerate(inst_text):
            inst_text[idx] = text.replace('##', '@@')
        outputs = self.tokenizer.batch_encode_plus(inst_text, add_special_tokens=True)
        input_ids = outputs.data['input_ids']
        token_type_ids = outputs.data['token_type_ids']
        attention_mask = outputs.data['attention_mask']

        return input_ids, token_type_ids, attention_mask

    def batch_biEDU_bert_id(self, biEDU_texts):
        c_biEDU_texts = []
        for idx, biEDU in enumerate(biEDU_texts):
            t1 = biEDU[0].replace('##', '@@')
            t2 = biEDU[1].replace('##', '@@')
            c_biEDU_texts.append((t1, t2))
        outputs = self.tokenizer.batch_encode_plus(c_biEDU_texts, add_special_tokens=True)
        input_ids = outputs.data['input_ids']
        token_type_ids = outputs.data['token_type_ids']
        attention_mask = outputs.data['attention_mask']
        return input_ids, token_type_ids, attention_mask