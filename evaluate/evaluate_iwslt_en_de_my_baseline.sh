SAVE_DIR=${1?Error: No Model Directory Provided}

# Separate script for evaluation
fairseq-generate data-bin/iwslt14.tokenized.en-de \
    --gen-subset valid \
    --path $SAVE_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe
