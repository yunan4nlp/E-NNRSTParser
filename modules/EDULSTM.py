from modules.Layer import *

class EDULSTM(nn.Module):

    def __init__(self, vocab, config):
        super(EDULSTM, self).__init__()
        self.config = config

        self.lstm = nn.LSTM(input_size=config.word_dims,
                            hidden_size=config.lstm_hiddens,
                            num_layers=config.lstm_layers,
                            batch_first=True,
                            bidirectional=True)

        self.drop = nn.Dropout(config.dropout_lstm_hidden)

    def forward(self, word_represents, edu_lengths):
        lstm_input = nn.utils.rnn.pack_padded_sequence(word_represents, edu_lengths,
                                                       batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(lstm_input)
        outputs = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        lstm_outputs = self.drop(outputs[0])
        return lstm_outputs
