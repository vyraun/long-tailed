# Generate and score the test/valid set with sacrebleu
# Modify as appropriate
SRC=${1?Error: No Model Directory Provided}
TGT=${2?Error: No Model Directory Provided}
MODEL=${3?Error: No Model Directory Provided}

if [ "$TGT" = "en" ]; then
    DATA=iwslt17.${SRC}.${TGT}.bpe10k
else
    DATA=iwslt17.${TGT}.${SRC}.bpe10k
fi

sacrebleu --test-set iwslt17 --language-pair ${SRC}-${TGT} --echo src \
    | python scripts/spm_encode.py --model examples/translation/${DATA}/sentencepiece.bpe.model \
    > iwslt17.test.${SRC}-${TGT}.${SRC}.bpe
cat iwslt17.test.${SRC}-${TGT}.${SRC}.bpe \
    | fairseq-interactive data-bin/$DATA/ \
      --task translation \
      --source-lang ${SRC} --target-lang ${TGT} \
      --path ${MODEL} \
      --buffer-size 2000 --batch-size 64 \
      --beam 5 --remove-bpe=sentencepiece \
    > iwslt17.test.${SRC}-${TGT}.${TGT}.sys
grep ^H iwslt17.test.${SRC}-${TGT}.${TGT}.sys | cut -f3 \
    | sacrebleu --test-set iwslt17 --language-pair ${SRC}-${TGT}
