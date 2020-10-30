# Preprocess/binarize the data
mkdir data-bin/my_splits_test

fairseq-preprocess --source-lang de --target-lang en \
    --srcdict dict.de.txt \
    --tgtdict dict.en.txt \
    --trainpref ./most --validpref ./medium --testpref ./least \
    --destdir data-bin/my_splits_test \
    --workers 2
