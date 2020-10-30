# Temp
# Download and prepare the data
#cd examples/translation/
#bash prepare-iwslt14.sh
#cd ../..

#mkdir data-bin/iwslt14.tokenized.en-de
#rm -rf data-bin/iwslt14.tokenized.de-en

# Preprocess/binarize the data
#TEXT=examples/translation/iwslt14.tokenized.de-en
#fairseq-preprocess --source-lang en --target-lang de \
#    --joined-dictionary \
#    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
#    --destdir data-bin/iwslt14.tokenized.en-de \
#    --workers 20

# Train on the data
SAVE_DIR=iwslt_en_de_60k
mkdir -p $SAVE_DIR

tempdir=$(mktemp -d /tmp/XXXXX)
cp -r data-bin/iwslt14.tokenized.en-de $tempdir

# Train on the data
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train \
    --distributed-world-size 4 \
    $tempdir/iwslt14.tokenized.en-de \
    --arch transformer_iwslt_de_en \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 5.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 16000 \
    --update-freq 1 \
    --save-dir $SAVE_DIR \
    --max-update 60000 \
    --num-workers 12 \
    --log-format json \
    --log-interval 1000 \
    --keep-last-epochs 20 \
    --seed 42 \
    --fp16 \

#    --eval-bleu \
#    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#    --eval-bleu-detok moses \
#    --eval-bleu-remove-bpe \
#    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \

# Evaluate the best model
# Separate script for evaluation
#fairseq-generate data-bin/iwslt14.tokenized.de-en \
#    --path $SAVE_DIR/checkpoint_best.pt \
#    --batch-size 128 --beam 5 --remove-bpe
