import torch.nn as nn
import math
from modules.Layer import *
from modules.ScaleMix import *


class GlobalEncoder(nn.Module):

    def __init__(self, vocab, config, bert_extractor):
        super(GlobalEncoder, self).__init__()

        self.bert_extractor = bert_extractor
        self.drop_emb = nn.Dropout(config.dropout_emb)

        self.start_layer = bert_extractor.start_layer
        self.bert_layers = bert_extractor.bert_layers
        self.layer_num = bert_extractor.layer_num
        config.bert_hidden_size = bert_extractor.bert.config.hidden_size

        self.mlp_words = nn.ModuleList([NonLinear(config.bert_hidden_size, config.word_dims, activation=nn.GELU())\
                                        for i in range(self.layer_num)])

        for i in range(self.layer_num):
            nn.init.orthogonal_(self.mlp_words[i].linear.weight)
            nn.init.zeros_(self.mlp_words[i].linear.bias)

        self.rescale = ScalarMix(mixture_size=self.layer_num)

    def forward(self, input_ids, token_type_ids, attention_mask):
        batch_size, max_edu_num, max_tok_len = input_ids.size()
        input_ids = input_ids.view(-1, max_tok_len)
        token_type_ids = token_type_ids.view(-1, max_tok_len)
        attention_mask = attention_mask.view(-1, max_tok_len)

        with torch.no_grad():
            _, _, encoder_outputs = \
                self.bert_extractor(input_ids, token_type_ids, attention_mask)

        bert_inputs = []
        for idx in range(self.start_layer, self.bert_layers):
            input = encoder_outputs[idx][:, 0]
            bert_inputs.append(input)

        proj_hiddens = []
        for idx in range(self.layer_num):
            proj_hidden = self.mlp_words[idx](bert_inputs[idx])
            proj_hiddens.append(proj_hidden)
        x_embed = self.rescale(proj_hiddens)

        x_embed = x_embed.view(batch_size, max_edu_num, -1)
        x_embed = self.drop_emb(x_embed)
        return x_embed




