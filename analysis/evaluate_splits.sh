MODEL=${1?Error: No Model Directory Provided}
python analyze_dictionary_log.py dict.de.txt ../examples/translation/iwslt14.tokenized.de-en/valid.de ../examples/translation/iwslt14.tokenized.de-en/valid.en
bash binarize_splits.sh
bash evaluate.sh $MODEL/checkpoint_best.pt train # Most Rare
mv target.txt most_freq_target_test.txt
mv hyp.txt most_freq_hyp_test.txt
#bash evaluate.sh $MODEL/checkpoint_best.pt valid # Medium Rare
mv target.txt medium_freq_target_test.txt
mv hyp.txt medium_freq_hyp_test.txt
#bash evaluate.sh $MODEL/checkpoint_best.pt test  # Least Rare
mv target.txt least_freq_target_test.txt
mv hyp.txt least_freq_hyp_test.txt
