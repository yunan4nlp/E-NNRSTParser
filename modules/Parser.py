from transition.State import *
import torch.nn.functional as F
from modules.Layer import *

class DisParser(object):
    def __init__(self, global_encoder, EDULSTM, dec, config):
        self.config = config
        self.global_encoder = global_encoder
        self.EDULSTM = EDULSTM
        self.dec = dec
        self.use_cuda = next(filter(lambda p: p.requires_grad, dec.parameters())).is_cuda
        self.batch_states = []
        self.step = []
        for idx in range(config.test_batch_size):
            self.batch_states.append([])
            self.step.append(0)
            for idy in range(config.max_state_len):
                self.batch_states[idx].append(State())

    def train(self):
        self.global_encoder.train()
        self.EDULSTM.train()
        self.dec.train()
        self.training = True

    def eval(self):
        self.global_encoder.eval()
        self.EDULSTM.eval()
        self.dec.eval()
        self.training = False

    def encode(self,
               doc_inputs,
               EDU_offset_index,
               batch_denominator,
               edu_lengths 
               ):

        doc_input_ids, doc_token_type_ids, doc_attention_mask = doc_inputs
        if self.use_cuda:
            doc_input_ids = doc_input_ids.cuda()
            doc_token_type_ids = doc_token_type_ids.cuda()
            doc_attention_mask = doc_attention_mask.cuda()

            EDU_offset_index = EDU_offset_index.cuda()
            batch_denominator = batch_denominator.cuda()

        bert_represents = self.global_encoder(
            doc_input_ids, doc_token_type_ids, doc_attention_mask,
            EDU_offset_index,
            batch_denominator)

        edu_hiddens = self.EDULSTM(bert_represents, edu_lengths)

        self.edu_represents = torch.cat([bert_represents, edu_hiddens], dim=-1)


    def max_action_len(self, batch_feats):
        max_ac_len = -1
        b = len(batch_feats)
        for idx in range(0, b):
            cur_feats = batch_feats[idx]
            tmp = len(cur_feats)
            if tmp > max_ac_len:
                max_ac_len = tmp
        return max_ac_len

    def hidden_prepare(self, batch_feats):
        batch_size, EDU_num, hidden_size = self.edu_represents.size()
        action_num = self.max_action_len(batch_feats)# training, whole step

        if self.training:
            assert action_num == EDU_num * 2
        else:
            assert action_num == 1 # 4 predicting, only one step

        bucket = torch.zeros(batch_size, 1, hidden_size).type(torch.FloatTensor)
        if self.use_cuda:
            bucket = bucket.cuda()
        edu_rep = torch.cat((self.edu_represents, bucket), 1) # batch_size, action_num + 1, hidden_size

        stack_index = torch.ones(batch_size * action_num * 3 * EDU_num).type(torch.LongTensor) * EDU_num
        stack_denominator = torch.ones(batch_size * action_num * 3).type(torch.FloatTensor) * -1
        queue_index = torch.ones(batch_size * action_num).type(torch.LongTensor) * EDU_num

        for b_iter in range(batch_size):
            r_a = len(batch_feats[b_iter])
            batch_stack_offset = b_iter * (action_num * 3 * EDU_num)
            batch_queue_offset = b_iter * action_num
            for cur_step in range(action_num):
                action_stack_offset = cur_step * (3 * EDU_num)
                if cur_step < r_a:
                    feat = batch_feats[b_iter][cur_step]
                    if feat is None:
                        break
                    feat_offest = b_iter * (EDU_num + 1)
                    if feat.q0 is not None:
                        queue_index[batch_queue_offset + cur_step] = feat_offest + feat.q0
                    if feat.s0 is not None:
                        s0_edu_start = feat.s0.edu_start
                        s0_edu_end = feat.s0.edu_end
                        l = s0_edu_end - s0_edu_start + 1
                        index_offest = batch_stack_offset + action_stack_offset + 0 * EDU_num
                        denominator_offset = index_offest // EDU_num
                        stack_denominator[denominator_offset] = l
                        for idx in range(l):
                            stack_index[index_offest + idx] = feat_offest + idx + s0_edu_start
                    if feat.s1 is not None:
                        s1_edu_start = feat.s1.edu_start
                        s1_edu_end = feat.s1.edu_end
                        l = s1_edu_end - s1_edu_start + 1
                        index_offest = batch_stack_offset + action_stack_offset + 1 * EDU_num
                        denominator_offset = index_offest // EDU_num
                        stack_denominator[denominator_offset] = l
                        for idx in range(l):
                            stack_index[index_offest + idx] = feat_offest + idx + s1_edu_start
                    if feat.s2 is not None:
                        s2_edu_start = feat.s2.edu_start
                        s2_edu_end = feat.s2.edu_end
                        l = s2_edu_end - s2_edu_start + 1
                        index_offest = batch_stack_offset + action_stack_offset + 2 * EDU_num
                        denominator_offset = index_offest // EDU_num
                        stack_denominator[denominator_offset] = l
                        for idx in range(l):
                            stack_index[index_offest + idx] = feat_offest + idx + s2_edu_start
        '''
        for b_iter in range(batch_size):
            b_offset = b_iter * action_num * 3 * EDU_num
            for cur_step in range(action_num):
                a_offset = b_offset + cur_step * 3 * EDU_num
                for idx in range(3 * EDU_num):
                    print(stack_index.data[a_offset + idx], end=' ')
                    if (idx + 1) % EDU_num == 0:
                        print(", ", end='')
                print()
            print()

        for b_iter in range(batch_size):
            b_offset = b_iter * action_num * 3
            for cur_step in range(action_num):
                a_offset = b_offset + cur_step * 3
                print(stack_denominator.data[a_offset + 0], end=",")
                print(stack_denominator.data[a_offset + 1], end=",")
                print(stack_denominator.data[a_offset + 2])
            print()
        '''

        if self.use_cuda:
            #queue_index = queue_index.cuda()
            #stack_index = stack_index.cuda()
            stack_denominator = stack_denominator.cuda()

        edu_rep = edu_rep.cpu()
        queue_state = torch.index_select(edu_rep.view(batch_size * (EDU_num + 1), hidden_size), 0, queue_index)
        stack_state = torch.index_select(edu_rep.view(batch_size * (EDU_num + 1), hidden_size), 0, stack_index)

        if self.use_cuda:
            queue_state = queue_state.cuda()
            stack_state = stack_state.cuda()

        stack_state = stack_state.view(batch_size * action_num * 3, EDU_num, hidden_size)
        stack_state = AvgPooling(stack_state, stack_denominator)
        #hidden_state = F.max_pool1d(edu_state.transpose(2, 1), kernel_size=EDU_num).squeeze(-1)

        queue_state = queue_state.view(batch_size, action_num, 1, hidden_size)
        stack_state = stack_state.view(batch_size, action_num, 3, hidden_size)
        hidden_state = torch.cat([stack_state, queue_state], -2)
        hidden_state = hidden_state.view(batch_size, action_num, -1)

        return hidden_state

    def all_states_are_finished(self, batch_states, batch_size):
        is_finish = True
        for idx in range(batch_size):
            cur_states = batch_states[idx]
            if not cur_states[self.step[idx]].is_end():
                is_finish = False
                break
        return is_finish

    def get_feats_from_state(self, batch_size):
        feats = []
        for idx in range(batch_size):
            cur_states = self.batch_states[idx]
            cur_step = self.step[idx]
            if not cur_states[cur_step].is_end():
                feat = cur_states[cur_step].prepare_index()
                feats.append([feat])
            else:
                feats.append([None])
        return feats

    def get_candidate_from_state(self, vocab, batch_size):
        candidates = []
        for idx in range(batch_size):
            cur_states = self.batch_states[idx]
            cur_step = self.step[idx]
            if not cur_states[cur_step].is_end():
                candidate = cur_states[cur_step].get_candidate_actions(vocab)
                candidates.append([candidate])
            else:
                candidates.append([None])
        return candidates

    def move(self, pred_actions, onebatch, vocab):
        batch_size = len(pred_actions)
        for idx in range(batch_size):
            cur_states = self.batch_states[idx]
            cur_step = self.step[idx]
            if not cur_states[cur_step].is_end():
                next_state = self.batch_states[idx][cur_step + 1]
                doc = onebatch[idx][0]
                cur_states[cur_step].move(next_state, pred_actions[idx][0], doc, vocab)
                self.step[idx] += 1

    def get_cut(self, feats, candidate, hidden_state, vocab):
        batch_size, action_num, _ = hidden_state.size()
        cut_data = np.array([[[0] * vocab.ac_size] * action_num] * batch_size, dtype=float)
        if self.training:
            action_num % 2 == 0
        else:
            action_num == 1
        for idx in range(batch_size):
            r_a = len(feats[idx])
            for idy in range(r_a):
                if candidate[idx][idy] is not None:
                    cut_data[idx][idy] = candidate[idx][idy] * -1e+20
        cut = Variable(torch.from_numpy(cut_data).type(torch.FloatTensor))

        if self.use_cuda:
            cut = cut.cuda()
        return cut

    def decode(self, onebatch, batch_feats, batch_candidate, vocab):
        batch_size, EDU_num, hidden_size = self.edu_represents.size()
        if self.training:
            assert batch_size == len(batch_feats)
            hidden_state = self.hidden_prepare(batch_feats)
            cut = self.get_cut(batch_feats, batch_candidate, hidden_state, vocab)
            self.decoder_outputs = self.dec(hidden_state, cut)
            batch_scores = self.decoder_outputs.data.cpu().numpy()
            predict_actions = self.get_predict_actions(batch_feats, batch_scores, vocab)
            return predict_actions
        else:
            for idx in range(batch_size):
                start_state = self.batch_states[idx][0]
                start_state.clear()
                start_state.ready(onebatch[idx][0])
                self.step[idx] = 0
            while not self.all_states_are_finished(self.batch_states, batch_size):
                feats = self.get_feats_from_state(batch_size)
                candidate = self.get_candidate_from_state(vocab, batch_size)
                hidden_state = self.hidden_prepare(feats)
                cut = self.get_cut(feats, candidate, hidden_state, vocab)
                self.decoder_outputs = self.dec(hidden_state, cut)
                batch_scores = self.decoder_outputs.data.cpu().numpy()
                predict_actions = self.get_predict_actions(feats, batch_scores, vocab)
                self.move(predict_actions, onebatch, vocab)

    def get_predict_actions(self, batch_feats, batch_scores, vocab):
        batch_size = len(batch_feats)
        assert batch_size == len(batch_scores)
        predict_actions = []
        for b_iter in range(batch_size):
            r_a = len(batch_feats[b_iter])
            actions = []
            for cur_step in range(r_a):
                cur_step_action_id = np.argmax(batch_scores[b_iter][cur_step])
                cur_step_action = vocab.id2ac(cur_step_action_id)
                actions.append(cur_step_action)
            predict_actions.append(actions)
        return predict_actions

    def compute_accuracy(self, predict_actions, gold_actions):
        total_num = 0
        correct = 0
        batch_size = len(predict_actions)
        assert batch_size == len(gold_actions)
        for b_iter in range(batch_size):
            action_num = len(predict_actions[b_iter])
            assert action_num == len(gold_actions[b_iter])
            for cur_step in range(action_num):
                if predict_actions[b_iter][cur_step] == gold_actions[b_iter][cur_step]:
                    correct += 1
                total_num += 1
        return total_num, correct

    def compute_loss(self, true_acs):
        batch_size, action_len, action_num = self.decoder_outputs.size()
        true_acs = _model_var(
            self.dec,
            pad_sequence(true_acs, length=action_len, padding=-1, dtype=np.int64))
        arc_loss = F.cross_entropy(
            self.decoder_outputs.view(batch_size * action_len, action_num), true_acs.view(batch_size * action_len),
            ignore_index=-1)
        return arc_loss

def _model_var(model, x):
    p = next(filter(lambda p: p.requires_grad, model.parameters()))
    if p.is_cuda:
        x = x.cuda(p.get_device())
    return torch.autograd.Variable(x)

def pad_sequence(xs, length=None, padding=-1, dtype=np.float64):
    lengths = [len(x) for x in xs]
    if length is None:
        length = max(lengths)
    y = np.array([np.pad(x.astype(dtype), (0, length - l),
                         mode="constant", constant_values=padding)
                  for x, l in zip(xs, lengths)])
    return torch.from_numpy(y)

