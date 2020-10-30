SAVE_DIR=${1?Error: No Model Directory Provided}

DATA=iwslt14.tokenized.en-de

# Separate script for evaluation
CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/$DATA \
    --gen-subset valid \
    --path $SAVE_DIR \
    --batch-size 64 --beam 5 --remove-bpe --quiet
