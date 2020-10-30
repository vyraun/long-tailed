# Then download and preprocess the data
#cd examples/translation/
#bash prepare-iwslt17-koen.sh
#cd ../..

#rm -rf data-bin/iwslt17.ko.en.bpe10k

#TEXT=examples/translation/iwslt17.ko.en.bpe10k
# Binarize the ko-en dataset
#fairseq-preprocess --source-lang ko --target-lang en \
#    --joined-dictionary \
#    --trainpref $TEXT/train.bpe.ko-en \
#    --validpref $TEXT/valid0.bpe.ko-en,$TEXT/valid1.bpe.ko-en,$TEXT/valid2.bpe.ko-en,$TEXT/valid3.bpe.ko-en,$TEXT/valid4.bpe.ko-en,$TEXT/valid5.bpe.ko-en \
#    --testpref $TEXT/valid6.bpe.ko-en \
#    --destdir data-bin/iwslt17.ko.en.bpe10k \
#    --workers 10

# Train on the data
SAVE_DIR=checkpoints/transformer_koen
mkdir -p $SAVE_DIR

tempdir=$(mktemp -d /tmp/XXXXX)
cp -r data-bin/iwslt17.ko.en.bpe10k $tempdir

CUDA_VISIBLE_DEVICES=0 fairseq-train $tempdir/iwslt17.ko.en.bpe10k \
  -a transformer_iwslt_de_en --optimizer adam --lr 0.0005 -s ko -t en \
  --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 \
  --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --criterion label_smoothed_cross_entropy --max-update 50000 \
  --warmup-updates 4000 --warmup-init-lr '1e-07' \
  --adam-betas '(0.9, 0.98)' --save-dir $SAVE_DIR

# Train on the data --> get bleu score
#CUDA_VISIBLE_DEVICES=0,1 fairseq-train \
#    --distributed-world-size 2 \
#    $tempdir/iwslt14.tokenized.de-en \
#    --arch transformer_iwslt_de_en \
#    --share-decoder-input-output-embed \
#    --optimizer adam --adam-betas '(0.9, 0.98)' \
#    --clip-norm 0.0 \
#    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \#
#    --dropout 0.3 --weight-decay 0.0001 \
#    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#    --max-tokens 4096 \
#    --save-dir $SAVE_DIR \#
#    --max-update 40000 \
#    --num-workers 12 \
#    --log-format json \
#    --log-interval 100 \
#    --keep-last-epochs 20 \
#    --seed 42 \
#    --fp16

# Start Training
#mkdir -p checkpoints/transformer_iwslt17#
#CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt17.ko.en.bpe10k/ \
#    --max-epoch 50 \
#    --task translation \
#    --share-decoder-input-output-embed \#
#    --arch transformer_iwslt_de_en \
#    --optimizer adam --adam-betas '(0.9, 0.98)' \
#    --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
#    --warmup-updates 4000 --warmup-init-lr '1e-07' \
#    --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
#    --dropout 0.3 --weight-decay 0.0001 \
#    --save-dir checkpoints/transformer_iwslt17 \
#    --max-tokens 4000 \
#    --update-koeq 8 \
#    --fp16
