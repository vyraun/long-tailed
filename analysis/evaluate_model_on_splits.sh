python analyze_dictionary.py dict.de.txt ../examples/translation/iwslt14.tokenized.de-en/valid.de ../examples/translation/iwslt14.tokenized.de-en/valid.en
bash binarize_splits.sh
#bash evaluate.sh checkpoint_best.pt train # Most Rare
#bash evaluate.sh checkpoint_best.pt valid # Medium Rare
#bash evaluate.sh checkpoint_best.pt test  # Least Rare

