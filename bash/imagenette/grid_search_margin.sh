#!/usr/bin/env bash
MODEL=$1
DEVICE=$2

for memory in 200 500 1000 2000
do
  for past_margin_w in 1 0.5 0.25 0.1 0.05 0.025 0.01
  do
      for gamma in 1
        do
        for alpha in 0
        do
          python main.py +scenario=cil_cifar10_5 model=$MODEL +training=cifar10_5 +method=margin device=$DEVICE method.mem_size=$memory method.past_task_reg=$past_margin_w method.gamma=$gamma  hydra=search +wadnb_tags=[grid_search_margin] method.margin_type=adaptive +method.rehearsal_metric='mse' experiment=dev head=margin_head
          for m in 1 0.5 0.25:
          do
            python main.py +scenario=cil_cifar10_5 model=$MODEL +training=cifar10_5 +method=margin device=$DEVICE method.mem_size=$memory method.past_task_reg=$past_margin_w method.past_margin=$m method.gamma=$gamma  hydra=search +wadnb_tags=[grid_search_margin] method.margin_type=mean +method.rehearsal_metric='mse' experiment=dev head=margin_head
          done
#          python main.py +scenario=cil_cifar10_5 model=$MODEL +training=cifar10_5 +method=margin device=$DEVICE method.mem_size=$memory method.past_task_reg=$past_margin_w method.gamma=$gamma  hydra=search +wadnb_tags=[grid_search_margin] method.margin_type=adaptive  experiment=dev head=margin_head &
#          wait
        done
        done
    done
  done
