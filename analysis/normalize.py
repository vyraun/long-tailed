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

# Load the checkpoint
checkpoint_file = 'checkpoint_best.pt'
d = torch.load( checkpoint_file, map_location=( lambda s, _: torch.serialization.default_restore_location(s, 'cpu')),)

# Load the Model with Dictionary
iwslt = TransformerModel.from_pretrained('./', checkpoint_file=checkpoint_file)

jj = d
bp()

print("The Embedding Layer Shape")
print(d['model']['decoder.embed_tokens.weight'].shape)

# Get the embeddings in numpy
numpy_weights = d['model']['decoder.embed_tokens.weight'].numpy()
print(numpy_weights.shape)

# Now get the Norms
norms = [1]*numpy_weights.shape[0]

for i in range(0, numpy_weights.shape[0]):
    norms[i] = np.linalg.norm(numpy_weights[i], 2)
    print(norms[i])

# Normalize the Embeddings
for i in range(0, numpy_weights.shape[0]):
    numpy_weights[i] = (numpy_weights[i]/(norms[i]) ** 0.2)

d['model']['decoder.embed_tokens.weight'] = torch.from_numpy(numpy_weights)

iwslt.models[0].state_dict().update(d['model'])
iwslt.models[0].load_state_dict(d['model'])

# Save the Model
torch.save(d, "checkpoint_normalized.pt")
print("Normalized Model is Saved")
