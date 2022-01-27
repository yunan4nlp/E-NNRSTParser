from transformers import *
from modules.Layer import *


class AutoModelExtractor(nn.Module):
    def __init__(self, plm_dir, config, tok_helper):
        super(AutoModelExtractor, self).__init__()
        self.config = config
        self.auto_model = AutoModel.from_pretrained(plm_dir)
        self.auto_model.resize_token_embeddings(len(tok_helper.tokenizer))

        self.bert_layers = self.auto_model.config.num_hidden_layers + 1
        self.start_layer = config.start_layer
        self.end_layer = config.end_layer
        self.tune_start_layer = config.tune_start_layer
        if self.start_layer > self.bert_layers - 1: self.start_layer = self.bert_layers - 1
        self.layer_num = self.end_layer - self.start_layer

        for p in self.auto_model.named_parameters():
            p[1].requires_grad = False

        for p in self.auto_model.named_parameters():
            items = p[0].split('.')
            if len(items) < 2: continue
            if items[0] == 'word_embedding' and 0 >= self.tune_start_layer:
                p[1].requires_grad = True
            if items[0] == 'layer':
                layer_id = int(items[1]) + 1
                if layer_id >= self.tune_start_layer: p[1].requires_grad = True


    def forward(self, input_ids, token_type_ids, attention_mask):
        output, new_mems, hidden_states = self.auto_model(input_ids=input_ids,
                                                          attention_mask=attention_mask,
                                                          token_type_ids=token_type_ids,
                                                          output_hidden_states=True)

        return output, new_mems, hidden_states
