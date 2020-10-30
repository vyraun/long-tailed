model=${1?Error}
fairseq-generate data-bin/writingPrompts --path $model/checkpoint_best.pt --batch-size 32 --beam 1 --sampling --sampling-topk 10 --temperature 0.8 --nbest 1
