SRC=${1?Error}
TGT=${2?Error}
SAVE_DIR=${3?Error: No Model Directory Provided}

if [ "$TGT" = "en" ]; then
    DATA=iwslt14.tokenized.${SRC}-${TGT}
else
    DATA=iwslt14.tokenized.${TGT}-${SRC}
fi

# Separate script for evaluation
CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/$DATA \
    --gen-subset train \
    --path $SAVE_DIR \
    --batch-size 64 --beam 5 #--remove-bpe

#--remove-bpe
