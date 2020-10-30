import sys
from bert_score import score

with open(sys.argv[1]) as f:
    hyps = f.read().splitlines()

with open(sys.argv[2]) as f:
    refs = f.read().splitlines()

print("Hypotheses = {}, References = {}".format(len(hyps), len(refs)))

P, R, F1 = score(hyps, refs, lang="en", verbose=True)

print(f"System level F1 score: {100*F1.mean():.3f}")
