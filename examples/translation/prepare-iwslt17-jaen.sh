#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# "en"

SRCS=(
    "ja"
)
TGT=en

ROOT=$(dirname "$0")
SCRIPTS=$ROOT/../../scripts
SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode.py

BPESIZE=10000
ORIG=$ROOT/iwslt17_orig
DATA=$ROOT/iwslt17.ja.en.bpe10k
mkdir -p "$ORIG" "$DATA"

TRAIN_MINLEN=1  # remove sentences with <1 BPE token
TRAIN_MAXLEN=250  # remove sentences with >250 BPE tokens

URLS=(
    "https://www.dropbox.com/s/9993iff6l0d8wii/ja-en.tgz"
)
ARCHIVES=(
    "ja-en.tgz"
)
VALID_SETS=(
    "IWSLT17.TED.dev2010.ja-en IWSLT17.TED.tst2010.ja-en IWSLT17.TED.tst2011.ja-en IWSLT17.TED.tst2012.ja-en IWSLT17.TED.tst2013.ja-en IWSLT17.TED.tst2014.ja-en IWSLT17.TED.tst2015.ja-en"
)

# download and extract data
for ((i=0;i<${#URLS[@]};++i)); do
    ARCHIVE=$ORIG/${ARCHIVES[i]}
    if [ -f "$ARCHIVE" ]; then
        echo "$ARCHIVE already exists, skipping download"
    else
        URL=${URLS[i]}
        wget -P "$ORIG" "$URL"
        if [ -f "$ARCHIVE" ]; then
            echo "$URL successfully downloaded."
        else
            echo "$URL not successfully downloaded."
            exit 1
        fi
    fi
    FILE=${ARCHIVE: -4}
    if [ -e "$FILE" ]; then
        echo "$FILE already exists, skipping extraction"
    else
        tar -C "$ORIG" -xzvf "$ARCHIVE"
    fi
done

echo "pre-processing train data..."
for SRC in "${SRCS[@]}"; do
    for LANG in "${SRC}" "${TGT}"; do
        cat "$ORIG/${SRC}-${TGT}/train.tags.${SRC}-${TGT}.${LANG}" \
            | grep -v '<url>' \
            | grep -v '<talkid>' \
            | grep -v '<keywords>' \
            | grep -v '<speaker>' \
            | grep -v '<reviewer' \
            | grep -v '<translator' \
            | grep -v '<doc' \
            | grep -v '</doc>' \
            | sed -e 's/<title>//g' \
            | sed -e 's/<\/title>//g' \
            | sed -e 's/<description>//g' \
            | sed -e 's/<\/description>//g' \
            | sed 's/^\s*//g' \
            | sed 's/\s*$//g' \
            > "$DATA/train.${SRC}-${TGT}.${LANG}"
    done
done

echo "pre-processing valid data..."
for ((i=0;i<${#SRCS[@]};++i)); do
    SRC=${SRCS[i]}
    VALID_SET=(${VALID_SETS[i]})
    for ((j=0;j<${#VALID_SET[@]};++j)); do
        FILE=${VALID_SET[j]}
        for LANG in "$SRC" "$TGT"; do
            grep '<seg id' "$ORIG/${SRC}-${TGT}/${FILE}.${LANG}.xml" \
                | sed -e 's/<seg id="[0-9]*">\s*//g' \
                | sed -e 's/\s*<\/seg>\s*//g' \
                | sed -e "s/\’/\'/g" \
                > "$DATA/valid${j}.${SRC}-${TGT}.${LANG}"
        done
    done
done

# learn BPE with sentencepiece
TRAIN_FILES=$(for SRC in "${SRCS[@]}"; do echo $DATA/train.${SRC}-${TGT}.${SRC}; echo $DATA/train.${SRC}-${TGT}.${TGT}; done | tr "\n" ",")
echo "learning joint BPE over ${TRAIN_FILES}..."
python "$SPM_TRAIN" \
    --input=$TRAIN_FILES \
    --model_prefix=$DATA/sentencepiece.bpe \
    --vocab_size=$BPESIZE \
    --character_coverage=1.0 \
    --model_type=bpe

# encode train/valid
echo "encoding train with learned BPE..."
for SRC in "${SRCS[@]}"; do
    python "$SPM_ENCODE" \
        --model "$DATA/sentencepiece.bpe.model" \
        --output_format=piece \
        --inputs $DATA/train.${SRC}-${TGT}.${SRC} $DATA/train.${SRC}-${TGT}.${TGT} \
        --outputs $DATA/train.bpe.${SRC}-${TGT}.${SRC} $DATA/train.bpe.${SRC}-${TGT}.${TGT} \
        --min-len $TRAIN_MINLEN --max-len $TRAIN_MAXLEN
done

echo "encoding valid with learned BPE..."
for ((i=0;i<${#SRCS[@]};++i)); do
    SRC=${SRCS[i]}
    VALID_SET=(${VALID_SETS[i]})
    for ((j=0;j<${#VALID_SET[@]};++j)); do
        python "$SPM_ENCODE" \
            --model "$DATA/sentencepiece.bpe.model" \
            --output_format=piece \
            --inputs $DATA/valid${j}.${SRC}-${TGT}.${SRC} $DATA/valid${j}.${SRC}-${TGT}.${TGT} \
            --outputs $DATA/valid${j}.bpe.${SRC}-${TGT}.${SRC} $DATA/valid${j}.bpe.${SRC}-${TGT}.${TGT}
    done
done
