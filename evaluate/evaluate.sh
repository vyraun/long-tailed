SAVE_DIR=${1?Error: No Model Directory Provided}
MODEL=${2?Error: No Model Directory Provided}

# Separate script for evaluation
CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --gen-subset train \
    --path $SAVE_DIR/$MODEL \
    --batch-size 32 --beam 5 --remove-bpe
