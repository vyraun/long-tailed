# Then download and preprocess the data
cd examples/translation/
bash prepare-iwslt17-multilingual.sh
cd ../..

# Binarize the de-en dataset
TEXT=examples/translation/iwslt17.de_fr.en.bpe16k
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train.bpe.de-en \
    --validpref $TEXT/valid0.bpe.de-en,$TEXT/valid1.bpe.de-en,$TEXT/valid2.bpe.de-en,$TEXT/valid3.bpe.de-en,$TEXT/valid4.bpe.de-en,$TEXT/valid5.bpe.de-en \
    --destdir data-bin/iwslt17.de_fr.en.bpe16k \
    --workers 10

# Binarize the fr-en dataset
# NOTE: it's important to reuse the en dictionary from the previous step
fairseq-preprocess --source-lang fr --target-lang en \
    --trainpref $TEXT/train.bpe.fr-en \
    --validpref $TEXT/valid0.bpe.fr-en,$TEXT/valid1.bpe.fr-en,$TEXT/valid2.bpe.fr-en,$TEXT/valid3.bpe.fr-en,$TEXT/valid4.bpe.fr-en,$TEXT/valid5.bpe.fr-en \
    --tgtdict data-bin/iwslt17.de_fr.en.bpe16k/dict.en.txt \
    --destdir data-bin/iwslt17.de_fr.en.bpe16k \
    --workers 10

# Train a multilingual transformer model
# NOTE: the command below assumes 1 GPU, but accumulates gradients from
#       8 fwd/bwd passes to simulate training on 8 GPUs
mkdir -p checkpoints/multilingual_transformer
CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt17.de_fr.en.bpe16k/ \
    --max-epoch 50 \
    --ddp-backend=no_c10d \
    --task multilingual_translation --lang-pairs de-en,fr-en \
    --arch multilingual_transformer_iwslt_de_en \
    --share-decoders --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
    --dropout 0.3 --weight-decay 0.0001 \
    --save-dir checkpoints/multilingual_transformer \
    --max-tokens 4000 \
    --update-freq 8

# Generate and score the test set with sacrebleu
SRC=de
sacrebleu --test-set iwslt17 --language-pair ${SRC}-en --echo src \
    | python scripts/spm_encode.py --model examples/translation/iwslt17.de_fr.en.bpe16k/sentencepiece.bpe.model \
    > iwslt17.test.${SRC}-en.${SRC}.bpe
cat iwslt17.test.${SRC}-en.${SRC}.bpe \
    | fairseq-interactive data-bin/iwslt17.de_fr.en.bpe16k/ \
      --task multilingual_translation --lang-pairs de-en,fr-en \
      --source-lang ${SRC} --target-lang en \
      --path checkpoints/multilingual_transformer/checkpoint_best.pt \
      --buffer-size 2000 --batch-size 128 \
      --beam 5 --remove-bpe=sentencepiece \
    > iwslt17.test.${SRC}-en.en.sys
grep ^H iwslt17.test.${SRC}-en.en.sys | cut -f3 \
    | sacrebleu --test-set iwslt17 --language-pair ${SRC}-en
