SAVE_DIR=${1?Error: No Model Directory Provided}

# sacrebleu doesn't have iwslt 2014
# bash sacrebleu_pregen.sh iwslt2014 de en gen_file

# Preprocess/binarize the data
TEXT=examples/translation/iwslt14.tokenized.de-en
cd $TEXT
mkdir iwslt14.tokenized.detokenized.de-en
mkdir iwslt14.tokenized.detruecased.de-en

sacremoses detruecase -j 4 < valid.en > iwslt14.tokenized.detruecased.de-en/valid.en
sacremoses detruecase -j 4 < valid.de > iwslt14.tokenized.detruecased.de-en/valid.de
sacremoses detruecase -j 4 < test.en > iwslt14.tokenized.detruecased.de-en/test.en
sacremoses detruecase -j 4 < test.de > iwslt14.tokenized.detruecased.de-en/test.de
sacremoses detruecase -j 4 < train.en > iwslt14.tokenized.detruecased.de-en/train.en
sacremoses detruecase -j 4 < train.de > iwslt14.tokenized.detruecased.de-en/train.de

sacremoses detokenize -j 4 < iwslt14.tokenized.detruecased.de-en/valid.en > iwslt14.tokenized.detokenized.de-en/valid.en
sacremoses detokenize -j 4 < iwslt14.tokenized.detruecased.de-en/valid.de > iwslt14.tokenized.detokenized.de-en/valid.de
sacremoses detokenize -j 4 < iwslt14.tokenized.detruecased.de-en/test.en > iwslt14.tokenized.detokenized.de-en/test.en
sacremoses detokenize -j 4 < iwslt14.tokenized.detruecased.de-en/test.de > iwslt14.tokenized.detokenized.de-en/test.de
sacremoses detokenize -j 4 < iwslt14.tokenized.detruecased.de-en/train.en > iwslt14.tokenized.detokenized.de-en/train.en
sacremoses detokenize -j 4 < iwslt14.tokenized.detruecased.de-en/train.de > iwslt14.tokenized.detokenized.de-en/train.de

cd ../../../

TEXT=examples/translation/iwslt14.tokenized.de-en/iwslt14.tokenized.detokenized.de-en

fairseq-preprocess --source-lang de --target-lang en \
    --srcdict data-bin/iwslt14.tokenized.de-en/dict.de.txt \
    --tgtdict data-bin/iwslt14.tokenized.de-en/dict.en.txt \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.detokenized.de-en \
    --workers 20

# Separate script for evaluation
fairseq-generate data-bin/iwslt14.tokenized.detokenized.de-en \
    --gen-subset test \
    --path $SAVE_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe
