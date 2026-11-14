import torch
import random
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--batch_size', type=int,default=32384)
    parser.add_argument('--recdim', type=int,default=128,)
    parser.add_argument('--lr', type=float,default=0.001,)
    parser.add_argument('--decay', type=float,default=0.,)
    parser.add_argument('--lambda1', type=float,default=0.5,)
    parser.add_argument('--data_path', type=str, default='../data',
                        help='the path to dataset')
    parser.add_argument('--cred_path', type=str, default='./assets',
                        help='the path to credential')
    parser.add_argument('--weights_path', type=str, default='./weights',
                        help='the path to credential')
    parser.add_argument('--dataset', type=str,default='micro_video',
                        help="available datasets: ['micro_video', 'kuai', 'amazon_book']")
    parser.add_argument('--topks', type=list, default=[10, 20, 50, 100])
    parser.add_argument('--epochs', type=int,default=500)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--contrast-size', type=int, default=16)
    parser.add_argument('--evaluate-interval', type=int, default=500)
    parser.add_argument('--neg-sampling', type=str, default='uniform')
    parser.add_argument('--device', type=str, default='none')
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.05)
    parser.add_argument('--c', type=int, default=3)
    parser.add_argument('--depth', type=int, default=0)
    parser.add_argument('--pair-reset-interval', type=int, default=1)
    parser.add_argument('--max-seq-len', type=int, default=50)
    parser.add_argument('--user-bucket-size', type=float, default=86400)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--n-heads', type=int, default=1)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--model-name', type=str, default="mf")
    parser.add_argument('--norm-first', type=bool, default=True)
    parser.add_argument('--dr-anchor', type=str, default="user")
    parser.add_argument('--ablation', type=str, default="none")
    parser.add_argument('--n-intents', type=int, default=128)
    parser.add_argument('--n-proto', type=int, default=128)
    parser.add_argument('--melt-alpha', type=float, default=0.5)
    parser.add_argument('--k-pop', type=int, default=1)
    parser.add_argument('--ablation2', type=str, default="none")
    parser.add_argument('--alpha1', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=1.)
    parser.add_argument('--debiased_eval', type=str, default="none")
    parser.add_argument('--lambda-cen', type=float, default=0.5, help='ANOVA-centering penalty weight (gamma)')
    parser.add_argument('--score-norm', type=str, default="normalized", choices=["normalized", "unnormalized"],
                         help='normalized: cosine similarity / tau; unnormalized: raw dot product')
    parser.add_argument('--short-half-life', type=float, default=1.0,
                         help='two-timescale Hawkes: short-component half-life, in the timestamp units used by the dataset (days)')
    parser.add_argument('--long-half-life', type=float, default=7.0,
                         help='two-timescale Hawkes: long-component half-life, in the timestamp units used by the dataset (days)')
    parser.add_argument('--half-life-grid', type=str, default='0.1,0.5,2,10,50,200',
                         help='multiscale Hawkes: comma-separated list of K fixed excitation half-lives (days), e.g. "0.02,0.05,0.15,0.4,1.0,3.0"')
    # sampled-softmax loss-control baselines (cf_ss.py / seq_rec_ss.py / seq_rec_tisasrec_ss.py)
    parser.add_argument('--ss-correction', type=int, default=1, choices=[0, 1],
                         help='1: add log((|V|-1)/M) importance correction to negative logits (same as l_choice); 0: plain sampled softmax')
    parser.add_argument('--item-init-std', type=float, default=0.02,
                         help='std of N(0, std) init for item_embedding (0.02 = same init as the Hawkes/ANOVA models; <=0 keeps nn.Embedding default)')
    parser.add_argument('--result-dir', type=str, default='./results/sampled_softmax',
                         help='where the per-run metric json is written')
    # separately-estimated Hawkes popularity + plug-in utility learning (*_hawkes_anova_plugin.py)
    parser.add_argument('--plugin-path', type=str, default='./weights_plugin',
                         help='root of the checkpoints holding the pre-fitted (frozen) Hawkes parameters: {root}/{dataset}/{plugin-model}_seed{seed}.pt')
    parser.add_argument('--plugin-model', type=str, default='tisasrec',
                         help='which backbone\'s checkpoint supplies the Hawkes parameters (same --seed is used)')
    parser.add_argument('--plugin-beta-scale', type=float, default=-1.0,
                         help='multiply the checkpoint\'s decay rate beta by this factor (time-unit conversion). <0: auto (1000 for kuairand, 1 otherwise)')
    parser.add_argument('--plugin-eta-grid', type=str, default='0.0,0.1,0.3,0.5,0.7,0.9,1.0',
                         help='comma-separated eta values evaluated after training (the run\'s --alpha1 is always added)')
    parser.add_argument('--plugin-result-dir', type=str, default='./results/plugin_hawkes',
                         help='where the per-run plug-in json (concentration + eta-grid metrics) is written')



    try:
        return parser.parse_args()
    except:
        return parser.parse_args([])


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def set_device(device="none"):
    if device == "none":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else: 
            device = "cpu"

    return device
