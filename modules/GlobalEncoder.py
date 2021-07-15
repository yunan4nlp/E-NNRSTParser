import torch.nn as nn
import math
from modules.Layer import *
from modules.ScaleMix import *


class GlobalEncoder(nn.Module):

    def __init__(self, vocab, config, auto_extractor):
        super(GlobalEncoder, self).__init__()

        self.auto_extractor = auto_extractor
        self.drop_emb = nn.Dropout(config.dropout_emb)

        self.start_layer = auto_extractor.start_layer
        self.bert_layers = auto_extractor.bert_layers
        self.layer_num = auto_extractor.layer_num
        config.bert_hidden_size = auto_extractor.auto_model.config.hidden_size

        self.mlp_words = nn.ModuleList([NonLinear(config.bert_hidden_size, config.word_dims, activation=nn.GELU())\
                                        for i in range(self.layer_num)])

        for i in range(self.layer_num):
            nn.init.orthogonal_(self.mlp_words[i].linear.weight)
            nn.init.zeros_(self.mlp_words[i].linear.bias)

        self.rescale = ScalarMix(mixture_size=self.layer_num)

    def forward(self,
                doc_input_ids, doc_token_type_ids, doc_attention_mask,
                EDU_offset_index, batch_denominator):

        _, _, doc_encoder_outputs = \
            self.auto_extractor(doc_input_ids, doc_token_type_ids, doc_attention_mask)

        bert_inputs = []
        for idx in range(self.start_layer, self.bert_layers):
            input = doc_encoder_outputs[idx]
            bert_inputs.append(input)

        proj_hiddens = []
        for idx in range(self.layer_num):
            proj_hidden = self.mlp_words[idx](bert_inputs[idx])
            proj_hiddens.append(proj_hidden)
        x_embed = self.rescale(proj_hiddens)

        batch_size, max_doc_token_len, hidden_size = x_embed.size()

        _, max_EDU_num, max_tok_len = EDU_offset_index.size()

        EDU_offset_index = \
            EDU_offset_index.unsqueeze(-1).repeat(1, 1, 1, hidden_size).view(batch_size,  max_EDU_num * max_tok_len, hidden_size)

        EDU_offset_index = EDU_offset_index.cpu()
        x_embed = x_embed.cpu()
        EDU_embed = torch.gather(x_embed, index=EDU_offset_index, dim=1)
        if next(self.parameters()).is_cuda:
            EDU_embed = EDU_embed.cuda()

        EDU_embed = EDU_embed.view(batch_size * max_EDU_num, max_tok_len, hidden_size)
        batch_denominator = batch_denominator.view(batch_size * max_EDU_num, -1).unsqueeze(1)
        pooled_EDU_embed = torch.bmm(batch_denominator, EDU_embed)

        pooled_EDU_embed = pooled_EDU_embed.view(batch_size, max_EDU_num, -1)
        pooled_EDU_embed = self.drop_emb(pooled_EDU_embed)

        return pooled_EDU_embed