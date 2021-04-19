import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.nn import functional, init
import math

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def drop_input(word_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.data.new(batch_size, seq_length).fill_(1 - dropout_emb)
    word_masks = Variable(torch.bernoulli(word_masks), requires_grad=False)
    scale = 3.0 / (2.0 * word_masks + 1e-12)
    word_masks *= scale
    word_masks = word_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks

    return word_embeddings

def drop_input_independent(word_embeddings, tag_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.data.new(batch_size, seq_length).fill_(1 - dropout_emb)
    word_masks = Variable(torch.bernoulli(word_masks), requires_grad=False)
    tag_masks = tag_embeddings.data.new(batch_size, seq_length).fill_(1 - dropout_emb)
    tag_masks = Variable(torch.bernoulli(tag_masks), requires_grad=False)
    scale = 3.0 / (2.0 * word_masks + tag_masks + 1e-12)
    word_masks *= scale
    tag_masks *= scale
    word_masks = word_masks.unsqueeze(dim=2)
    tag_masks = tag_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks
    tag_embeddings = tag_embeddings * tag_masks

    return word_embeddings, tag_embeddings

def drop_sequence_sharedmask(inputs, dropout, batch_first=True):
    if batch_first:
        inputs = inputs.transpose(0, 1)
    seq_length, batch_size, hidden_size = inputs.size()
    drop_masks = inputs.data.new(batch_size, hidden_size).fill_(1 - dropout)
    drop_masks = Variable(torch.bernoulli(drop_masks), requires_grad=False)
    drop_masks = drop_masks / (1 - dropout)
    drop_masks = torch.unsqueeze(drop_masks, dim=2).expand(-1, -1, seq_length).permute(2, 0, 1)
    inputs = inputs * drop_masks

    return inputs.transpose(1, 0)

def get_tensor_np(t):
    return t.data.cpu().numpy()

def orthonormal_initializer(output_size, input_size):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py
    """
    print(output_size, input_size)
    I = np.eye(output_size)
    lr = .1
    eps = .05 / (output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI ** 2 / 2)
            Q2 = Q ** 2
            Q -= lr * Q.dot(QTQmI) / (
                    np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        print('Orthogonal pretrainer loss: %.2e' % loss)
    else:
        print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return np.transpose(Q.astype(np.float32))


class NonLinear(nn.Module):
    def __init__(self, input_size, hidden_size, activation=None):
        super(NonLinear, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(in_features=input_size, out_features=hidden_size)
        if activation is None:
            self._activate = lambda x: x
        else:
            if not callable(activation):
                raise ValueError("activation must be callable: type={}".format(type(activation)))
            self._activate = activation

    def forward(self, x):
        y = self.linear(x)
        return self._activate(y)

    def reset_parameters(self):
        W = orthonormal_initializer(self.hidden_size, self.input_size)
        self.linear.weight.data.copy_(torch.from_numpy(W))

        b = np.zeros(self.hidden_size, dtype=np.float32)
        self.linear.bias.data.copy_(torch.from_numpy(b))


class Biaffine(nn.Module):
    def __init__(self, in1_features, in2_features, out_features,
                 bias=(True, True)):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        W = np.zeros((self.linear_output_size, self.linear_input_size), dtype=np.float32)
        self.linear.weight.data.copy_(torch.from_numpy(W))

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = input1.data.new(batch_size, len1, 1).zero_().fill_(1)
            input1 = torch.cat((input1, Variable(ones)), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = input2.data.new(batch_size, len2, 1).zero_().fill_(1)
            input2 = torch.cat((input2, Variable(ones)), dim=2)
            dim2 += 1

        affine = self.linear(input1)

        affine = affine.view(batch_size, len1*self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)

        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2)

        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)

        return biaffine

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'in1_features=' + str(self.in1_features) \
            + ', in2_features=' + str(self.in2_features) \
            + ', out_features=' + str(self.out_features) + ')'

class LSTM(nn.LSTM):
    def reset_parameters(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                for i in range(4):
                    nn.init.orthogonal(self.__getattr__(name)[self.hidden_size*i:self.hidden_size*(i+1),:])
            if "bias" in name:
                nn.init.constant(self.__getattr__(name), 0)

'''
class NewLSTM(nn.Module):
    """A module that runs multiple steps of LSTM."""

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, \
                 bidirectional=False, dropout_in=0, dropout_out=0):
        super(NewLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.num_directions = 2 if bidirectional else 1

        self.fcells = []
        self.bcells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            self.fcells.append(nn.LSTMCell(input_size=layer_input_size, hidden_size=hidden_size))
            if self.bidirectional:
                self.bcells.append(nn.LSTMCell(input_size=layer_input_size, hidden_size=hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in range(self.num_layers):
            self.fcells[layer].reset_parameters()
            if self.bidirectional:
                self.bcells[layer].reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input, masks, initial, drop_masks):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in range(max_time):
            h_next, c_next = cell(input=input[time], hx=hx)
            h_next = h_next * masks[time] + initial[0] * (1 - masks[time])
            c_next = c_next * masks[time] + initial[1] * (1 - masks[time])
            output.append(h_next)
            if drop_masks is not None: h_next = h_next * drop_masks
            hx = (h_next, c_next)
        output = torch.stack(output, 0)
        return output, hx

    @staticmethod
    def _forward_brnn(cell, input, masks, initial, drop_masks):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in reversed(range(max_time)):
            h_next, c_next = cell(input=input[time], hx=hx)
            h_next = h_next * masks[time] + initial[0] * (1 - masks[time])
            c_next = c_next * masks[time] + initial[1] * (1 - masks[time])
            output.append(h_next)
            if drop_masks is not None: h_next = h_next * drop_masks
            hx = (h_next, c_next)
        output.reverse()
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input, masks, initial=None):
        if self.batch_first:
            input = input.transpose(0, 1)
            masks = torch.unsqueeze(masks.transpose(0, 1), dim=2)
        max_time, batch_size, _ = input.size()
        masks = masks.expand(-1, -1, self.hidden_size)

        if initial is None:
            initial = Variable(input.data.new(batch_size, self.hidden_size).zero_())
            initial = (initial, initial)
        h_n = []
        c_n = []

        for layer in range(self.num_layers):
            max_time, batch_size, input_size = input.size()
            input_mask, hidden_mask = None, None
            if self.training:
                input_mask = input.data.new(batch_size, input_size).fill_(1 - self.dropout_in)
                input_mask = Variable(torch.bernoulli(input_mask), requires_grad=False)
                input_mask = input_mask / (1 - self.dropout_in)
                input_mask = torch.unsqueeze(input_mask, dim=2).expand(-1, -1, max_time).permute(2, 0, 1)
                input = input * input_mask

                hidden_mask = input.data.new(batch_size, self.hidden_size).fill_(1 - self.dropout_out)
                hidden_mask = Variable(torch.bernoulli(hidden_mask), requires_grad=False)
                hidden_mask = hidden_mask / (1 - self.dropout_out)

            layer_output, (layer_h_n, layer_c_n) = NewLSTM._forward_rnn(cell=self.fcells[layer], \
                         input=input, masks=masks, initial=initial, drop_masks=hidden_mask)
            if self.bidirectional:
                blayer_output, (blayer_h_n, blayer_c_n) = NewLSTM._forward_brnn(cell=self.bcells[layer], \
                       input=input, masks=masks, initial=initial, drop_masks=hidden_mask)

            h_n.append(torch.cat([layer_h_n, blayer_h_n], 1) if self.bidirectional else layer_h_n)
            c_n.append(torch.cat([layer_c_n, blayer_c_n], 1) if self.bidirectional else layer_c_n)
            input = torch.cat([layer_output, blayer_output], 2) if self.bidirectional else layer_output

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        return input, (h_n, c_n)

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_ih = nn.Linear(in_features=input_size,
                                   out_features=4 * hidden_size)
        self.linear_hh = nn.Linear(in_features=hidden_size,
                                   out_features=4 * hidden_size,
                                   bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


    def reset_parameters(self):
        W = orthonormal_initializer(self.hidden_size, self.hidden_size + self.input_size)
        W_h, W_x = W[:, :self.hidden_size], W[:, self.hidden_size:]
        self.linear_ih.weight.data.copy_(torch.from_numpy(np.concatenate([W_x] * 4, 0)))
        self.linear_hh.weight.data.copy_(torch.from_numpy(np.concatenate([W_h] * 4, 0)))

        b = np.zeros(4 * self.hidden_size, dtype=np.float32)
        b[self.hidden_size:2 * self.hidden_size] = -1.0
        self.linear_ih.bias.data.copy_(torch.from_numpy(b))



    def forward(self, input, hx):
        if hx is None:
            batch_size = input.size(0)
            zero_hx = Variable(
                input.data.new(batch_size, self.hidden_size).zero_())
            hx = (zero_hx, zero_hx)
        h, c = hx
        lstm_vector = self.linear_ih(input) + self.linear_hh(h)
        i, f, g, o = lstm_vector.chunk(chunks=4, dim=1)
        f = f + 1
        new_c = c*self.sigmoid(f) + self.sigmoid(i)*self.tanh(g)
        new_h = self.tanh(new_c) * self.sigmoid(o)
        return new_h, new_c
'''

class MyLSTM(nn.Module):

    """A module that runs multiple steps of LSTM."""

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, \
                 bidirectional=False, dropout_in=0, dropout_out=0):
        super(MyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.num_directions = 2 if bidirectional else 1

        self.fcells = []
        self.bcells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            self.fcells.append(nn.LSTMCell(input_size=layer_input_size, hidden_size=hidden_size))
            if self.bidirectional:
                self.bcells.append(nn.LSTMCell(input_size=layer_input_size, hidden_size=hidden_size))

        self._all_weights = []
        for layer in range(num_layers):
            layer_params = (self.fcells[layer].weight_ih, self.fcells[layer].weight_hh, \
                            self.fcells[layer].bias_ih, self.fcells[layer].bias_hh)
            suffix = ''
            param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
            param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
            param_names = [x.format(layer, suffix) for x in param_names]
            for name, param in zip(param_names, layer_params):
                setattr(self, name, param)
            self._all_weights.append(param_names)

            if self.bidirectional:
                layer_params = (self.bcells[layer].weight_ih, self.bcells[layer].weight_hh, \
                                self.bcells[layer].bias_ih, self.bcells[layer].bias_hh)
                suffix = '_reverse'
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]
                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in range(self.num_layers):
            if self.bidirectional:
                param_ih_name = 'weight_ih_l{}{}'.format(layer, '_reverse')
                param_hh_name = 'weight_hh_l{}{}'.format(layer, '_reverse')
                param_ih = self.__getattr__(param_ih_name)
                param_hh = self.__getattr__(param_hh_name)
                if layer == 0:
                    W = orthonormal_initializer(self.hidden_size, self.hidden_size + self.input_size)
                else:
                    W = orthonormal_initializer(self.hidden_size, self.hidden_size + 2 * self.hidden_size)
                W_h, W_x = W[:, :self.hidden_size], W[:, self.hidden_size:]
                param_ih.data.copy_(torch.from_numpy(np.concatenate([W_x] * 4, 0)))
                param_hh.data.copy_(torch.from_numpy(np.concatenate([W_h] * 4, 0)))
            else:
                param_ih_name = 'weight_ih_l{}{}'.format(layer, '')
                param_hh_name = 'weight_hh_l{}{}'.format(layer, '')
                param_ih = self.__getattr__(param_ih_name)
                param_hh = self.__getattr__(param_hh_name)
                if layer == 0:
                    W = orthonormal_initializer(self.hidden_size, self.hidden_size + self.input_size)
                else:
                    W = orthonormal_initializer(self.hidden_size, self.hidden_size + self.hidden_size)
                W_h, W_x = W[:, :self.hidden_size], W[:, self.hidden_size:]
                param_ih.data.copy_(torch.from_numpy(np.concatenate([W_x] * 4, 0)))
                param_hh.data.copy_(torch.from_numpy(np.concatenate([W_h] * 4, 0)))

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(self.__getattr__(name), 0)


    @staticmethod
    def _forward_rnn(cell, input, masks, initial, drop_masks):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in range(max_time):
            h_next, c_next = cell(input=input[time], hx=hx)
            h_next = h_next*masks[time] + initial[0]*(1-masks[time])
            c_next = c_next*masks[time] + initial[1]*(1-masks[time])
            output.append(h_next)
            if drop_masks is not None: h_next = h_next * drop_masks
            hx = (h_next, c_next)
        output = torch.stack(output, 0)
        return output, hx

    @staticmethod
    def _forward_brnn(cell, input, masks, initial, drop_masks):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in reversed(range(max_time)):
            h_next, c_next = cell(input=input[time], hx=hx)
            h_next = h_next*masks[time] + initial[0]*(1-masks[time])
            c_next = c_next*masks[time] + initial[1]*(1-masks[time])
            output.append(h_next)
            if drop_masks is not None: h_next = h_next * drop_masks
            hx = (h_next, c_next)
        output.reverse()
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input, masks, initial=None):
        if self.batch_first:
            input = input.transpose(0, 1)
            masks = torch.unsqueeze(masks.transpose(0, 1), dim=2)
        max_time, batch_size, _ = input.size()
        masks = masks.expand(-1, -1, self.hidden_size)

        if initial is None:
            initial = Variable(input.data.new(batch_size, self.hidden_size).zero_())
            initial = (initial, initial)
        h_n = []
        c_n = []

        for layer in range(self.num_layers):
            max_time, batch_size, input_size = input.size()
            input_mask, hidden_mask = None, None
            if self.training:
                input_mask = input.data.new(batch_size, input_size).fill_(1 - self.dropout_in)
                input_mask = Variable(torch.bernoulli(input_mask), requires_grad=False)
                input_mask = input_mask / (1 - self.dropout_in)
                input_mask = torch.unsqueeze(input_mask, dim=2).expand(-1, -1, max_time).permute(2, 0, 1)
                input = input * input_mask

                hidden_mask = input.data.new(batch_size, self.hidden_size).fill_(1 - self.dropout_out)
                hidden_mask = Variable(torch.bernoulli(hidden_mask), requires_grad=False)
                hidden_mask = hidden_mask / (1 - self.dropout_out)

            layer_output, (layer_h_n, layer_c_n) = MyLSTM._forward_rnn(cell=self.fcells[layer], \
                input=input, masks=masks, initial=initial, drop_masks=hidden_mask)
            if self.bidirectional:
                blayer_output, (blayer_h_n, blayer_c_n) = MyLSTM._forward_brnn(cell=self.bcells[layer], \
                    input=input, masks=masks, initial=initial, drop_masks=hidden_mask)

            h_n.append(torch.cat([layer_h_n, blayer_h_n], 1) if self.bidirectional else layer_h_n)
            c_n.append(torch.cat([layer_c_n, blayer_c_n], 1) if self.bidirectional else layer_c_n)
            input = torch.cat([layer_output, blayer_output], 2) if self.bidirectional else layer_output

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        return input, (h_n, c_n)


def AvgPooling(data, denominator, dim=1):
    assert len(data.size()) == 3
    assert len(denominator.size()) == 1
    sum_data = torch.sum(data, dim)
    avg_data = torch.div(sum_data.transpose(1, 0), denominator).transpose(1, 0)
    return avg_data.contiguous()
