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

with open('inverse_frequencies.pkl', 'rb') as f:
    inverse_frequencies = pickle.load(f)

#with open('frequencies.pkl', 'wb') as f:
#    pickle.dump(frequencies, f)

#with open('inverse_frequencies.pkl', 'wb') as f:
#    pickle.dump(inverse_frequencies, f)

freq_norm = []

for x in freq_dict:
    freq_norm.append(freq_dict[x]/total_tokens)

with open('freq_norm.pkl', 'wb') as f:
    pickle.dump(freq_norm, f)

print("Total Tokens ", total_tokens)

# Load the checkpoint
checkpoint_file = 'checkpoint_best.pt'
d = torch.load( checkpoint_file, map_location=( lambda s, _: torch.serialization.default_restore_location(s, 'cpu')),)

# Load the Model with Dictionary
iwslt = TransformerModel.from_pretrained('./', checkpoint_file=checkpoint_file)

# Sample Translations
#print(iwslt.translate("ich möchte spielen"))


#tokens = iwslt.encode("ich möchte spielen")
#iwslt.models[0].encoder(tokens.unsqueeze(0), src_lengths=[1])
#prev_output_tokens = iwslt.encode("<s>")
#iwslt.models[0].decoder(encoder_output[0], prev_output_tokens)

# Load the State Dict from the Model
#d = iwslt.models[0].state_dict()

#print("Layers in the Model")
#for key in d['model']:
#    print(key)

# Access the Layers
#print(d['model']['decoder.embed_tokens.weight'][0])
#print(d['model']['decoder.embed_tokens.weight'] == d['model']['encoder.embed_tokens.weight'])

# Edit the Weights
#saved_decoder_embed_tokens_weight = d['model']['decoder.embed_tokens.weight']

print("The Embedding Layer Shape")
print(d['model']['decoder.embed_tokens.weight'].shape)

# Get the embeddings in numpy
numpy_weights = d['model']['decoder.embed_tokens.weight'].numpy()
print(numpy_weights.shape)

# Now get the Norms
norms = [1]*numpy_weights.shape[0]

for i in range(0, numpy_weights.shape[0]):
    norms[i] = np.linalg.norm(numpy_weights[i], 2)

for i in range(4, numpy_weights.shape[0]):
    print(frequencies[i-4], norms[i])

bp()

print(len(norms))
inverse_frequencies = [1, 1, 1, 1] + inverse_frequencies
frequencies = [1, 1, 1, 1] + frequencies

# Normalize the Embeddings
for i in range(0, numpy_weights.shape[0]): # 4 numpy_weights.shape[0] - 5000
    #print(np.log2(1 + frequencies[i]/inverse_frequencies[i]))
    #if freq_norm[i-4] != 0: # (norms[i]) ** 0.22 (norms[i]) ** 0.22
    #if i < 100:
        #print((inverse_frequencies[i-4]/frequencies[i-4])** 0.05)
    #pass
    #if i < 4:
    #print(frequencies[i])
    #if i > 2000: [0.17]
    ## Exponent
    #exp = 0.1 * (i/numpy_weights.shape[0])
    numpy_weights[i] = (numpy_weights[i]/(norms[i]) ** 0.24)  # * np.log2(1 + frequencies[i] )    # * freq_norm[i-4] # (100 * freq_norm[i-4]) # *(freq_norm[i-4]) # norms[i]

# Normalization method 1
#d['model']['decoder.embed_tokens.weight'] = F.normalize(d['model']['decoder.embed_tokens.weight'], p=2, dim=1) 31.59
# Normalization method 2
d['model']['decoder.embed_tokens.weight'] = torch.from_numpy(numpy_weights) # 31.59

#bp()

#d['model']['encoder.embed_tokens.weight'] = F.normalize(d['model']['encoder.embed_tokens.weight'], p=3, dim=1)

# d['decoder.embed_tokens.weight'] = torch.zeros((10152, 512))
# Can update the model
iwslt.models[0].state_dict().update(d['model'])
iwslt.models[0].load_state_dict(d['model'])

# Check the Weights
#print(d['model']['decoder.embed_tokens.weight'][0])

# Sample Translation
#print(iwslt.translate("ich möchte spielen"))

# Save the Model
torch.save(d, "checkpoint_normalized.pt")
print("Normalized Model is Saved")
