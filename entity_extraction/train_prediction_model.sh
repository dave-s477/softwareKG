#!/bin/bash

# Here it is assumed that SSC_pre_training.sh was already run!

# SSC + GSC train+devel+test -> predict new data
python perform_training.py --learning-rate 0.0015 --learning-decay 0.00007 --dropout-rate 0.4 --epochs 10 --train-set gold_all --test-set gold_test --save-name final_model_0 --train train --log-name final_model --sample-weights 0.1 --log-training 1 --custom-eval 1 --checkpoint pre_training_1 --lstm-size 100
python perform_training.py --learning-rate 0.0008 --learning-decay 0.00006 --dropout-rate 0.4 --epochs 10 --train-set gold_all --test-set gold_test --save-name final_model_1 --train train --log-name final_model --sample-weights 0.1 --log-training 1 --custom-eval 1 --global_epoch 10 --checkpoint final_model_0 --lstm-size 100
python perform_training.py --learning-rate 0.0002 --learning-decay 0.000015 --dropout-rate 0.4 --epochs 7 --train-set gold_all --test-set gold_test --save-name final_model_2 --train train --log-name final_model --sample-weights 0.1 --log-training 1 --custom-eval 1 --global_epoch 20 --checkpoint final_model_1 --lstm-size 100