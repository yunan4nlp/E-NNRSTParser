import sys
sys.path.extend(["../../","../","./"])
import random
import argparse
from data.Dataloader import *
from data.Config import *
import time
from modules.Parser import *
from modules.EDULSTM import *
from modules.Decoder import *
from modules.XLNetTune import *
from data.TokenHelper import *
from modules.GlobalEncoder import *
from modules.Optimizer import *
import pickle

from torch.cuda.amp import autocast as autocast
from torch.cuda.amp.grad_scaler import GradScaler


def train(train_inst, dev_data, test_data, parser, vocab, config, token_helper):

    auto_param = list(parser.global_encoder.auto_extractor.parameters())

    parser_param = list(parser.global_encoder.mlp_words.parameters()) + \
                   list(parser.global_encoder.rescale.parameters()) + \
                   list(parser.EDULSTM.parameters()) + \
                   list(parser.dec.parameters())

    model_param = [{'params': auto_param, 'lr': config.plm_learning_rate},
                   {'params': parser_param, 'lr': config.learning_rate}]

    model_optimizer = Optimizer(model_param, config, config.learning_rate)
    scaler = GradScaler()

    global_step = 0
    best_FF = 0
    batch_num = int(np.ceil(len(train_inst) / float(config.train_batch_size)))

    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter))
        batch_iter = 0

        overall_action_correct,  overall_total_action = 0, 0
        for onebatch in data_iter(train_inst, config.train_batch_size, True):

            doc_inputs = \
                batch_doc_variable(onebatch, vocab, config, token_helper)

            EDU_offset_index, batch_denominator, edu_lengths = batch_doc2edu_variable(onebatch, vocab, config, token_helper)

            batch_feats, batch_actions, batch_action_indexes, batch_candidate = \
                actions_variable(onebatch, vocab)

            parser.train()
            #with torch.autograd.profiler.profile() as prof:
            with autocast():
                parser.encode(
                    doc_inputs,
                    EDU_offset_index, batch_denominator, edu_lengths
                )
                predict_actions = parser.decode(onebatch, batch_feats, batch_candidate, vocab)
                loss = parser.compute_loss(batch_action_indexes)
                loss = loss / config.update_every

            #loss.backward()
            scaler.scale(loss).backward()
            loss_value = loss.data.item()

            total_actions, correct_actions = parser.compute_accuracy(predict_actions, batch_actions)
            overall_total_action += total_actions
            overall_action_correct += correct_actions
            during_time = float(time.time() - start_time)
            acc = overall_action_correct / overall_total_action
            #acc = 0
            print("Step:%d, Iter:%d, batch:%d, time:%.2f, acc:%.2f, loss:%.8f" \
                %(global_step, iter, batch_iter,  during_time, acc, loss_value))
            batch_iter += 1

            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                scaler.unscale_(model_optimizer.optim)
                nn.utils.clip_grad_norm_(auto_param + parser_param, max_norm=config.clip)

                scaler.step(model_optimizer.optim)
                scaler.update()
                model_optimizer.schedule()

                model_optimizer.zero_grad()
                global_step += 1

            if batch_iter % config.validate_every == 0 or batch_iter == batch_num:
                print("Dev:")
                with torch.no_grad():
                    predict(dev_data, parser, vocab, config, token_helper, config.dev_file + '.' + str(global_step))
                dev_FF = evaluate(config.dev_file, config.dev_file + '.' + str(global_step))

                print("Test:")
                with torch.no_grad():
                    predict(test_data, parser, vocab, config, token_helper, config.test_file + '.' + str(global_step))
                evaluate(config.test_file, config.test_file + '.' + str(global_step))

                if dev_FF > best_FF:
                    print("Exceed best Full F-score: history = %.2f, current = %.2f" % (best_FF, dev_FF))
                    best_FF = dev_FF

                    '''
                    if config.save_after >= 0 and iter >= config.save_after:
                        discoure_parser_model = {
                            "pwordEnc": parser.pwordEnc.state_dict(),
                            "wordLSTM": parser.wordLSTM.state_dict(),
                            "sent2span": parser.sent2span.state_dict(),
                            "EDULSTM": parser.EDULSTM.state_dict(),
                            "dec": parser.dec.state_dict()
                            }
                        torch.save(discoure_parser_model, config.save_model_path + "." + str(global_step))
                        print('Saving model to ', config.save_model_path + "." + str(global_step))
                    '''

def evaluate(gold_file, predict_file):
    gold_data = read_corpus(gold_file)
    predict_data = read_corpus(predict_file)
    S = Metric()
    N = Metric()
    R = Metric()
    F = Metric()
    for gold_doc, predict_doc in zip(gold_data, predict_data):
        assert len(gold_doc.EDUs) == len(predict_doc.EDUs)
        assert len(gold_doc.sentences) == len(predict_doc.sentences)
        gold_doc.evaluate_labelled_attachments(predict_doc.result, S, N, R, F)
    print("S:", end=" ")
    S.print()
    print("N:", end=" ")
    N.print()
    print("R:", end=" ")
    R.print()
    print("F:", end=" ")
    F.print()
    return F.getAccuracy()

def predict(data, parser, vocab, config, tokenizer, outputFile):
    start = time.time()
    parser.eval()
    outf = open(outputFile, mode='w', encoding='utf8')
    for onebatch in data_iter(data, config.test_batch_size, False):
        doc_inputs = batch_doc_variable(onebatch, vocab, config, token_helper)

        EDU_offset_index, batch_denominator, edu_lengths = batch_doc2edu_variable(onebatch, vocab, config, token_helper)


        # with torch.autograd.profiler.profile() as prof:
        with autocast():
            parser.encode(
                doc_inputs,
                EDU_offset_index, batch_denominator, edu_lengths
            )
            parser.decode(onebatch, None, None, vocab)
        batch_size = len(onebatch)
        for idx in range(batch_size):
            doc = onebatch[idx][0]
            cur_states = parser.batch_states[idx]
            cur_step = parser.step[idx]
            predict_tree = cur_states[cur_step - 1]._stack[cur_states[cur_step - 1]._stack_size - 1].str
            for sent, tags, type in zip(doc.origin_sentences, doc.sentences_tags, doc.sent_types):
                assert len(sent) == len(tags)
                for w, tag in zip(sent, tags):
                    outf.write(w + '_' + tag + ' ')
                outf.write(type[-1])
                outf.write('\n')
            for info in doc.other_infos:
                outf.write(info + '\n')
            outf.write(predict_tree + '\n')
            outf.write('\n')
    outf.close()
    end = time.time()
    during_time = float(end - start)
    print("Doc num: %d,  parser time = %.2f " % (len(data), during_time))

if __name__ == '__main__':
    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))


    # torch version
    print("Torch Version: ", torch.__version__)

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='examples/default.cfg')
    argparser.add_argument('--model', default='BaseParser')
    argparser.add_argument('--thread', default=1, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.manual_seed(config.seed)

    train_data = read_corpus(config.train_file)
    dev_data = read_corpus(config.dev_file)
    test_data = read_corpus(config.test_file)
    vocab = creatVocab(train_data, config.min_occur_count)
    #vec = vocab.load_pretrained_embs(config.pretrained_embeddings_file)# load extword table and embeddings

    torch.set_num_threads(args.thread)

    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    start_a = time.time()
    train_feats, train_actions = get_gold_actions(train_data, vocab, config)
    print("Get Action Time: ", time.time() - start_a)
    vocab.create_action_table(train_actions)

    train_candidate = get_gold_candid(train_data, vocab, config)

    train_insts = inst(train_data, train_feats, train_actions, train_candidate)
    dev_insts = inst(dev_data)
    test_insts = inst(test_data)

    print("train num: ", len(train_insts))
    print("dev num: ", len(dev_insts))
    print("test num: ", len(test_insts))

    print('Load pretrained encoder: ', config.xlnet_dir)
    token_helper = TokenHelper(config.xlnet_dir)
    auto_extractor = AutoModelExtractor(config, token_helper)
    print('Load pretrained encoder ok')

    global_encoder = GlobalEncoder(vocab, config, auto_extractor)
    EDULSTM = EDULSTM(vocab, config)
    dec = Decoder(vocab, config)
    pickle.dump(vocab, open(config.save_vocab_path, 'wb'))

    print(EDULSTM)
    print(dec)
    if config.use_cuda:
        torch.backends.cudnn.enabled = True
        #torch.backends.cudnn.benchmark = True
        global_encoder.cuda()
        EDULSTM = EDULSTM.cuda()
        dec = dec.cuda()

    parser = DisParser(global_encoder, EDULSTM, dec, config)
    train(train_insts, dev_insts, test_insts, parser, vocab, config, token_helper)

