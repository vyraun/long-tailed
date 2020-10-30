# Normalization Script

#bash prepare_pretrained.sh checkpoints/transformer data-bin/iwslt14.tokenized.de-en/ de-en-34
#cd de-en-34
python normalize_last_layer.py
cd ..
bash evaluate.sh de-en-34 checkpoint_normalized.pt

