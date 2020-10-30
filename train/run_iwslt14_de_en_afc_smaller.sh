# Temp
# Download and prepare the data

#cd examples/translation/
#bash prepare-iwslt14-en-es.sh
#cd ../..

# prepare-iwslt14-en-es.sh
#rm -rf data-bin/iwslt14.tokenized.es-en

# Preprocess/binarize the data

#TEXT=examples/translation/iwslt14.tokenized.es-en
#fairseq-preprocess --source-lang es --target-lang en \
#    --joined-dictionary \
#    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
#    --destdir data-bin/iwslt14.tokenized.es-en \
#    --workers 20

#location=${1?Error: No Model Directory Provided}

# Train on the data
SAVE_DIR=checkpoints/transformer_de_en_final_afc_ls_smaller_0.03
mkdir -p $SAVE_DIR

tempdir=$(mktemp -d /tmp/XXXXX)
cp -r data-bin/iwslt14.tokenized.de-en $tempdir

CUDA_VISIBLE_DEVICES=2 fairseq-train $tempdir/iwslt14.tokenized.de-en \
  -a transformer_iwslt_de_en --optimizer adam --lr 0.0005 -s de -t en \
  --dropout 0.3 --max-tokens 4000 \
  --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --criterion label_smoothed_focal_loss --gamma 1 --label-smoothing 0.03 --max-update 50000 \
  --warmup-updates 4000 --warmup-init-lr '1e-07' \
  --adam-betas '(0.9, 0.98)' --save-dir $SAVE_DIR \
  --fp16

# Train on the data
#SAVE_DIR=dir_ce+ls_iwslt_de_en_baseline
#mkdir -p $SAVE_DIR

#tempdir=$(mktemp -d /tmp/XXXXX)
#cp -r data-bin/iwslt14.tokenized.de-en $tempdir

# Train on the data --> get bleu score
#CUDA_VISIBLE_DEVICES=0,1 fairseq-train \
#    --distributed-world-size 2 \
#    $tempdir/iwslt14.tokenized.de-en \
#    --arch transformer_iwslt_de_en \
#    --share-decoder-input-output-embed \
#    --optimizer adam --adam-betas '(0.9, 0.98)' \
#    --clip-norm 0.0 \
#    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#    --dropout 0.3 --weight-decay 0.0001 \
#    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#    --max-tokens 4096 \
#    --save-dir $SAVE_DIR \
#    --max-update 40000 \
#    --num-workers 12 \
#    --log-format json \
#    --log-interval 100 \
#    --keep-last-epochs 20 \
#    --seed 42 \
#    --fp16

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
