from transformers.modeling_bert import *
from modules.Layer import *


class MyBertModel(BertModel):
    def __init__(self, config):
        super(MyBertModel, self).__init__(config)


class BertExtractor(nn.Module):
    def __init__(self, config, tok_helper):
        super(BertExtractor, self).__init__()
        self.config = config
        self.bert = MyBertModel.from_pretrained(config.bert_dir)
        self.bert.resize_token_embeddings(len(tok_helper.tokenizer))
        print("Load bert model finished.")

        self.bert_layers = self.bert.config.num_hidden_layers + 1
        self.start_layer = config.start_layer
        self.end_layer = config.end_layer
        if self.start_layer > self.bert_layers - 1: self.start_layer = self.bert_layers - 1
        self.layer_num = self.end_layer - self.start_layer

    def forward(self, input_ids, token_type_ids, attention_mask):
        sequence_output, pooled_output, encoder_outputs = self.bert(input_ids=input_ids,
                                                                    attention_mask=attention_mask,
                                                                    token_type_ids=token_type_ids,
                                                                    output_hidden_states=True)
        return sequence_output, pooled_output, encoder_outputs
