import sys

# Read the Dictionary
f = open(sys.argv[1])  # The Joint Dictionary
freq_dict ={}

# Read the Source and Target Sentences
sent = open(sys.argv[2]) # Source Sentences
refer = open(sys.argv[3]) # Target Sentences

sent_scores = []
sentences = []
references = []

for l in sent:
    s = l.strip()
    sentences.append(s)
sent.close()

for l in refer:
    s = l.strip()
    references.append(s)
refer.close()

print("Number of Sentences = {}".format(len(sentences)))

for l in f:
    s = l.strip().split()
    word = s[0]
    freq = int(s[1])
    freq_dict[word] = freq
    #print(word, freq)

f.close()

# Compute the Scores

for s in sentences:
    words = s.split()
    score = 0
    for w in words:
        score += freq_dict[w]
    sent_scores.append(score/len(words))  # len(words)**0.5 length normalized

sorted_sentences = [x for _, x in sorted(zip(sent_scores,sentences), reverse=True)]
sorted_references = [x for _, x in sorted(zip(sent_scores,references), reverse=True)]

# Split the Sentences into 3

most_rare_src = open("most.de", "w")
most_rare_tgt = open("most.en", "w")
for s, r in zip(sorted_sentences[0:2400], sorted_references[0:2400]):
    most_rare_src.write(s + "\n")
    most_rare_tgt.write(r + "\n")

medium_rare_src = open("medium.de", "w")
medium_rare_tgt = open("medium.en", "w")
for s, r in zip(sorted_sentences[2400:4800], sorted_references[2400:4800]):
    medium_rare_src.write(s + "\n")
    medium_rare_tgt.write(r + "\n")

least_rare_src = open("least.de", "w")
least_rare_tgt = open("least.en", "w")
for s, r in zip(sorted_sentences[4800:7200], sorted_references[4800:7200]):
    least_rare_src.write(s + "\n")
    least_rare_tgt.write(r + "\n")

print("Most Rare", sorted_sentences[0])
print("Most Frequent", sorted_sentences[-1])
