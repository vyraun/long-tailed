MODEL=${1?Error: No Model Directory Provided}
SPLIT=${2?Error: Either train valid test}

# Separate script for evaluation
CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/my_splits \
    --gen-subset $SPLIT \
    --path $MODEL \
    --batch-size 32 --beam 5 --remove-bpe
