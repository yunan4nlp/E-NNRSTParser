from modules.Layer import *

class TypeEmb(nn.Module):

    def __init__(self, vocab, config):
        super(TypeEmb, self).__init__()
        self.config = config

        self.type_embeddings = nn.Embedding(num_embeddings=vocab.EDUtype_size,
                                            embedding_dim=config.edu_type_dims,
                                            padding_idx=vocab.PAD
                                            )



        self.drop = nn.Dropout(config.dropout_lstm_hidden)

    def forward(self, edu_types):

        edu_embeddings = self.type_embeddings(edu_types)
        return edu_embeddings