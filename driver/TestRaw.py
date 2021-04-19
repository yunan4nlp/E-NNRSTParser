import sys
sys.path.extend(["../../","../","./"])
import random
import argparse
from data.Config import *
import pickle
import time
from data.Dataloader import *
from modules.Parser import *
from modules.EDULSTM import *
from modules.Decoder import *
from modules.BertModel import *
from data.BertTokenHelper import *
from modules.GlobalEncoder import *
from modules.Optimizer import *
from data.BertTokenHelper import *
from driver.TrainTest import predict, evaluate

from torch.cuda.amp import autocast as autocast
from torch.cuda.amp.grad_scaler import GradScaler


def predict_raw(data, parser, vocab, config, token_helper, outputFile):
    start = time.time()
    parser.eval()
    outf = open(outputFile, mode='w', encoding='utf8')
    for onebatch in data_iter(data, config.test_batch_size, False):
        batch_input_ids, batch_token_type_ids, batch_attention_mask, edu_lengths = \
            batch_biEDU_bert_variable(onebatch, vocab, config, token_helper)

        # with torch.autograd.profiler.profile() as prof:
        with autocast():
            parser.encode(batch_input_ids, batch_token_type_ids, batch_attention_mask, edu_lengths)
            parser.decode(onebatch, None, None, vocab)
        batch_size = len(onebatch)
        for idx in range(batch_size):
            doc = onebatch[idx][0]
            cur_states = parser.batch_states[idx]
            cur_step = parser.step[idx]
            predict_tree = cur_states[cur_step - 1]._stack[cur_states[cur_step - 1]._stack_size - 1].str
            for words in doc.origin_sentences:
                outf.write(" ".join(words) + "\n")
            outf.write(predict_tree + "\n")
            outf.write("\n")
    outf.close()

def read(input_file):
    with open(input_file, mode='r', encoding='utf8') as inf:
        doc = [] 
        for text in inf.readlines():
            text = text.strip()
            if text == "":
                if len(doc) > 0: yield doc
                doc = []
            else:
                doc.append(text)
        if len(doc) > 0: yield doc

def read_raw_corpus(input_file):
    data = []
    for doc_info in read(input_file):
        origin_sentences = []
        sentences_tags = []
        sentences = []

        total_words = []
        for sentence in doc_info[:-1]:
            words = sentence.split(" ")
            origin_sentences.append(words)
            tags = ['NN']  * len(words)
            sentences_tags.append(tags)
            norm_words = [normalize_to_lowerwithdigit(word) for word in words]
            sentences.append(norm_words)
            total_words += norm_words

        edu_info_list = doc_info[-1].split(' ')
        edu_list = []
        for edu_info in edu_info_list:
            edu_start, edu_end = edu_info[1:-1].split(",")
            edu_start = int(edu_start)
            edu_end = int(edu_end)
            edu = EDU(edu_start, edu_end, None)
            edu.words = total_words[edu_start:edu_end + 1]
            assert len(edu.words) == edu_end + 1 - edu_start
            edu_list.append(edu)


        inst = Discourse(origin_sentences, sentences, sentences_tags, None, None, None)
        inst.EDUs = edu_list
        data.append(inst)
    return data

if __name__ == '__main__':
    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='experiment/rst_model/config.cfg')
    argparser.add_argument('--thread', default=1, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)
    argparser.add_argument('--test_file', default='text/sample.txt.out', help='without evaluation')

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    vocab = pickle.load(open(config.load_vocab_path, 'rb'))
    discoure_parser_model = torch.load(config.load_model_path)


    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    if args.test_file != "":
        raw_data = read_raw_corpus(args.test_file)
        raw_insts = inst(raw_data)

        print('Load pretrained encoder.....')
        token_helper = BertTokenHelper(config.bert_dir)
        bert_extractor = BertExtractor(config, token_helper)
        print('Load pretrained encoder ok')

        global_encoder = GlobalEncoder(vocab, config, bert_extractor)
        EDULSTM = EDULSTM(vocab, config)
        dec = Decoder(vocab, config)

        global_encoder.mlp_words.load_state_dict(discoure_parser_model["mlp_words"])
        global_encoder.rescale.load_state_dict(discoure_parser_model["rescale"])
        EDULSTM.load_state_dict(discoure_parser_model["EDULSTM"])
        dec.load_state_dict(discoure_parser_model["dec"])

        if config.use_cuda:
            torch.backends.cudnn.enabled = True
            # torch.backends.cudnn.benchmark = True
            global_encoder = global_encoder.cuda()
            EDULSTM = EDULSTM.cuda()
            dec = dec.cuda()

        parser = DisParser(global_encoder, EDULSTM, dec, config)
        predict_raw(raw_insts, parser, vocab, config, token_helper, args.test_file + '.rst')
        print("rst parsing OK!")