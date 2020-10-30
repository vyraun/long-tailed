import sys
from tqdm import tqdm
import pickle

sentences = []

f = open(sys.argv[1], "r")

for l in f:
    sentences.append(l.strip())

ngram_dict = {}
n = 2

for sent in tqdm(sentences):
    s = sent.split()
    for j in range(1, n+1):
        for i in range(len(s)):

            unigram = " ".join([x for x in s[i:i+j]])
            bigram = " ".join([x for x in s[i:i+j]])

            if unigram in ngram_dict:
                ngram_dict[unigram] += 1
            else:
                ngram_dict[unigram] = 1

            if bigram in ngram_dict:
                ngram_dict[bigram] += 1
            else:
                ngram_dict[bigram] = 1


with open('ngram.pickle', 'wb') as handle:
    pickle.dump(ngram_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

#print(ngram_dict)
