SRC=${1?Error}
TGT=${2?Error}
SAVE_DIR=${3?Error: No Model Directory Provided}
DIR=tedtalks.tokenized.$SRC-$TGT

# Evaluate the best model
fairseq-generate data-bin/$DIR \
    --gen-subset valid \
    --path $SAVE_DIR \
    --batch-size 64 --beam 5 --remove-bpe
