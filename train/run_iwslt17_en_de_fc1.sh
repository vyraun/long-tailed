# Then download and preprocess the data
#cd examples/translation/
#bash prepare-iwslt17-deen.sh
#cd ../..

#rm -rf data-bin/iwslt17.de.en.bpe10k

#TEXT=examples/translation/iwslt17.de.en.bpe10k
# Binarize the de-en dataset
#fairseq-preprocess --source-lang de --target-lang en \
#    --joined-dictionary \
#    --trainpref $TEXT/train.bpe.de-en \
#    --validpref $TEXT/valid0.bpe.de-en,$TEXT/valid1.bpe.de-en,$TEXT/valid2.bpe.de-en,$TEXT/valid3.bpe.de-en,$TEXT/valid4.bpe.de-en,$TEXT/valid5.bpe.de-en \
#    --testpref $TEXT/valid6.bpe.de-en \
#    --destdir data-bin/iwslt17.de.en.bpe10k \
#    --workers 10

# Train on the data
SAVE_DIR=checkpoints/transformer_en_de_fc1_final17
mkdir -p $SAVE_DIR

tempdir=$(mktemp -d /tmp/XXXXX)
cp -r data-bin/iwslt17.de.en.bpe10k $tempdir

CUDA_VISIBLE_DEVICES=0 fairseq-train $tempdir/iwslt17.de.en.bpe10k \
  -a transformer_iwslt_de_en --optimizer adam --lr 0.0005 -s en -t de \
  --dropout 0.3 --max-tokens 4000 \
  --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --criterion original_focal_loss --gamma 1 --max-update 50000 \
  --warmup-updates 4000 --warmup-init-lr '1e-07' \
  --adam-betas '(0.9, 0.98)' --save-dir $SAVE_DIR \
  --seed 42 \
  --fp16


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
#CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt17.de.en.bpe10k/ \
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
#    --update-freq 8 \
#    --fp16
