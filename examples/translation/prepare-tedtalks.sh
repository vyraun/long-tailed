#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

#echo 'Cloning Moses github repository (for tokenization scripts)...'
#git clone https://github.com/moses-smt/mosesdecoder.git

#echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
#git clone https://github.com/rsennrich/subword-nmt.git

src=${1?Error: no source language given}
tgt=${2?Error: no target langauge given}
bpe=${3?Error: BPE Size}

BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=$bpe

# This url has to be fixed
if [ "$src" == "tr" ]; then
    URL="https://www.dropbox.com/s/fywjyn6mr21rgak/tr_to_en.zip"
fi

if [ "$src" == "az" ]; then
    URL="https://www.dropbox.com/s/a5fbjgxxep63qfj/az_to_en.zip"
fi

if [ "$src" == "be" ]; then
    URL="https://www.dropbox.com/s/yys4gyvd8xep8k4/be_to_en.zip"
fi

if [ "$src" == "gl" ]; then
    URL="https://www.dropbox.com/s/i5b36eppi6xz8fz/gl_to_en.zip"
fi

if [ "$src" == "ru" ]; then
    URL="https://www.dropbox.com/s/so04d8omb1b5m74/ru_to_en.zip"
fi

if [ "$src" == "pt" ]; then
    URL="https://www.dropbox.com/s/v7g0ais58propv6/pt_to_en.zip"
fi

GZ=${src}_to_${tgt}.zip

lang=${src}-${tgt}
prep=tedtalks.tokenized.$bpe.$lang
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

echo "Downloading data from ${URL}..."
cd $orig
wget "$URL"

if [ -f $GZ ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

unzip -j $GZ
cd ..

echo "creating train, valid, test files"
for l in $src $tgt; do
    cp $orig/$l.train $prep/$l.train
    cp $prep/$l.train $tmp/train.$l

    cp $orig/$l.dev $prep/$l.valid
    cp $prep/$l.valid $tmp/valid.$l

    cp $orig/$l.test $prep/$l.test
    cp $prep/$l.test $tmp/test.$l
done

echo "creating train file for learning a joint vocabulary"
TRAIN=$tmp/train.$src-$tgt
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
    done
done

echo "Okay, this is done, training can proceed now :)"
