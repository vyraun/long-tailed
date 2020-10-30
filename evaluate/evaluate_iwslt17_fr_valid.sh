SAVE_DIR=${1?Error: No Model Directory Provided}
DATA=iwslt17.fr.en.bpe10k

# Separate script for evaluation
CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/$DATA \
    --source-lang fr --target-lang en \
    --gen-subset valid \
    --path $SAVE_DIR \
    --batch-size 64 --beam 5 --remove-bpe --quiet
