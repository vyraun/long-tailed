# TED-Talks Dataset Setting
SRC=${1?Error: no source language given}
TGT=${2?Error: no target langauge given}
AFFIX=${3?Error: Affix}
gpu=${4}

# Download and prepare the data
#cd examples/translation/
#bash prepare-tedtalks.sh $SRC $TGT
#cd ../..

# Preprocess/binarize the data --> later, will make it language specific
DIR=tedtalks.tokenized.$SRC-$TGT
#TEXT=examples/translation/$DIR
#fairseq-preprocess --source-lang $SRC --target-lang $TGT \
#    --joined-dictionary \
#    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
#    --destdir data-bin/$DIR \
#    --workers 20

# Train on the data
SAVE_DIR=checkpoints/$SRC.$TGT.$AFFIX
mkdir -p $SAVE_DIR

tempdir=$(mktemp -d /tmp/XXXXX)
cp -r data-bin/$DIR $tempdir

CUDA_VISIBLE_DEVICES=$gpu fairseq-train $tempdir/$DIR \
  -a transformer_iwslt_de_en --optimizer adam --lr 0.0005 -s $SRC -t $TGT \
  --dropout 0.3 --max-tokens 4000 \
  --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --criterion focal_loss --gamma 1 --alpha 1 --max-update 40000 \
  --warmup-updates 4000 --warmup-init-lr '1e-07' \
  --adam-betas '(0.9, 0.98)' --save-dir $SAVE_DIR \
  --fp16 \
  --save-interval-updates 2000 --keep-interval-updates 10 --keep-last-epochs 10

# 40K is now reduced to 10K

#CUDA_VISIBLE_DEVICES=0 fairseq-train \
#    data-bin/$DIR \
#    --arch transformer_tedtalks --share-decoder-input-output-embed \
#    --reset-optimizer \
#    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#    --dropout 0.4 --weight-decay 0.0001 \
#    --criterion focal_loss \
#    --max-tokens 4096 \
#    --eval-bleu \
#    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#    --eval-bleu-detok moses \
#    --eval-bleu-remove-bpe \
#    --eval-bleu-print-samples \
#    --save-dir $SAVE_DIR \
#    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
#    --fp16

# Evaluate the best model
#fairseq-generate data-bin/$DIR \
#    --path $SAVE_DIR/checkpoint_best.pt \
#    --batch-size 128 --beam 5 --remove-bpe
