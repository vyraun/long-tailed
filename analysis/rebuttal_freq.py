import sys
import numpy as np
from scipy.stats import norm
import torch
import torch.nn.functional as F
from fairseq.models.transformer import *
from collections import OrderedDict
from pdb import set_trace as bp
from tqdm import tqdm
import pickle

f = open('combined_text.txt')
corpus = []
for l in f:
    s = l.strip()
    corpus.append(s)
f.close()

def get_idf_score(word):
    t = 0
    for x in corpus:
        if word in x:
            t += 1
    return t

f = open('dict.en.txt')
freq_dict = OrderedDict()
frequencies = []
#inverse_frequencies = []
total_tokens = 0

for l in tqdm(f):
    s = l.strip().split()
    word = s[0]
    freq = int(s[1])
    freq_dict[word] = freq
    total_tokens += freq
    frequencies.append(freq)
    #inverse_frequencies.append(get_idf_score(word))
f.close()
