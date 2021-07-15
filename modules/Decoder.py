from modules.Layer import *


class Decoder(nn.Module):
    def __init__(self, vocab, config):
        super(Decoder, self).__init__()
        self.config = config
        self.nonlinear1 = NonLinear(input_size=(config.lstm_hiddens * 2  + config.word_dims + config.edu_type_dims) * 4, ## four feats
                                    hidden_size=config.hidden_size,
                                    activation=nn.Tanh())

        self.drop = nn.Dropout(config.dropout_mlp)

        self.output = nn.Linear(in_features=config.hidden_size,
                                out_features=vocab.ac_size,
                                bias=False)

        nn.init.kaiming_uniform_(self.nonlinear1.linear.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='tanh')
        nn.init.kaiming_uniform_(self.output.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='linear')


    def forward(self, hidden_state, cut=None):
        hidden = self.drop(self.nonlinear1(hidden_state))
        action_score = self.output(hidden)
        if cut is not None:
            action_score = action_score + cut
        return action_score



