#!/bin/bash

# Here it is assumed that SSC_pre_training.sh was already run!

# SSC + GSC train -> predict on GSC devel
python perform_training.py --learning-rate 0.0015 --learning-decay 0.00007 --dropout-rate 0.4 --epochs 10 --train-set gold_train_with_pos --test-set gold_devel --save-name GSC_pre_train_trained_0 --train train --log-name GSC_pre_train_trained --sample-weights 0.1 --log-training 1 --custom-eval 1 --checkpoint pre_training_1 --lstm-size 100
python perform_training.py --learning-rate 0.0008 --learning-decay 0.00006 --dropout-rate 0.4 --epochs 10 --train-set gold_train_with_pos --test-set gold_devel --save-name GSC_pre_train_trained_1 --train train --log-name GSC_pre_train_trained --sample-weights 0.1 --log-training 1 --custom-eval 1 --global_epoch 10 --checkpoint GSC_pre_train_trained_0 --lstm-size 100
python perform_training.py --learning-rate 0.0002 --learning-decay 0.000015 --dropout-rate 0.4 --epochs 7 --train-set gold_train_with_pos --test-set gold_devel --save-name GSC_pre_train_trained_2 --train train --log-name GSC_pre_train_trained --sample-weights 0.1 --log-training 1 --custom-eval 1 --global_epoch 20 --checkpoint GSC_pre_train_trained_1 --lstm-size 100

# SSC + GSC train+devel -> predict on GSC test
python perform_training.py --learning-rate 0.0015 --learning-decay 0.00007 --dropout-rate 0.4 --epochs 10 --train-set gold_devel --test-set gold_test --save-name GSC_pre_train_devel_trained_0 --train train --log-name GSC_pre_train_devel_trained --sample-weights 0.1 --log-training 1 --custom-eval 1 --checkpoint pre_training_1 --lstm-size 100
python perform_training.py --learning-rate 0.0008 --learning-decay 0.00006 --dropout-rate 0.4 --epochs 10 --train-set gold_devel --test-set gold_test --save-name GSC_pre_train_devel_trained_1 --train train --log-name GSC_pre_train_devel_trained --sample-weights 0.1 --log-training 1 --custom-eval 1 --global_epoch 10 --checkpoint GSC_pre_train_devel_trained_0 --lstm-size 100
python perform_training.py --learning-rate 0.0002 --learning-decay 0.000015 --dropout-rate 0.4 --epochs 7 --train-set gold_devel --test-set gold_test --save-name GSC_pre_train_devel_trained_2 --train train --log-name GSC_pre_train_devel_trained --sample-weights 0.1 --log-training 1 --custom-eval 1 --global_epoch 20 --checkpoint GSC_pre_train_devel_trained_1 --lstm-size 100