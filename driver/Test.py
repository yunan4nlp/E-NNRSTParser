import sys
sys.path.extend(["../../","../","./"])
import random
import argparse
from data.Config import *
import pickle
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
    argparser.add_argument('--config_file', default='experiments/rst_model/config.cfg')
    argparser.add_argument('--thread', default=1, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)
    argparser.add_argument('--test_file', default='experiments/rst/sample.txt', help='without evaluation')

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    vocab = pickle.load(open(config.load_vocab_path, 'rb'))
    discoure_parser_model = torch.load(config.load_model_path)

    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    if args.test_file != "":
        test_data = read_corpus(args.test_file)
        test_insts = inst(test_data)

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
        predict(test_insts, parser, vocab, config, token_helper, args.test_file + '.out')
        evaluate(args.test_file, args.test_file + '.out')

