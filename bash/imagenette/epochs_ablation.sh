#!/usr/bin/env bash

METHOD=$1
MODEL=$2
DEVICE=$3

case $METHOD in
gem)
  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=gem_200   device="$DEVICE"
  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=gem_500   device="$DEVICE"
  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=gem_1000   device="$DEVICE"
  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=gem_2000   device="$DEVICE"
;;
naive)
  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5  +method=naive   device="$DEVICE"
;;
cumulative)
  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=cumulative   device="$DEVICE"
;;
replay)
  for epochs in 20 30 50 100
    do
      python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=replay_500 device="$DEVICE" experiment=base1 training.epochs=$epochs +wadnb_tags=[ablation]
    done
;;
#margin)
##  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=margin_cifar10 method.mem_size=200 head=margin_head method.gamma=1 +method.alpha=0 method.past_task_reg=0.05  device="$DEVICE" optimizer.momentum=0.9
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=margin_cifar10 method.mem_size=200 head=margin_head method.gamma=1 +method.alpha=0 method.past_task_reg=0.1 +method.rehearsal_metric='mse' method.margin_type=adaptive device="$DEVICE" +wadnb_tags=[final_results_margin]
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=margin_cifar10 method.mem_size=500 head=margin_head method.gamma=1 +method.alpha=0 method.past_task_reg=0.25 +method.rehearsal_metric='mse' method.margin_type=adaptive  device="$DEVICE" +wadnb_tags=[final_results_margin]
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=margin_cifar10 method.mem_size=1000 head=margin_head method.gamma=1 +method.alpha=0 method.past_task_reg=0.25 +method.rehearsal_metric='mse'  device="$DEVICE" +wadnb_tags=[final_results]
##  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=er_ace_default method.mem_size=500   device="$DEVICE"
##  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=er_ace_default method.mem_size=1000   device="$DEVICE"
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=er_ace_default method.mem_size=2000   device="$DEVICE"
;;
#ssil)
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=ssil_200   device="$DEVICE" training.epochs=100
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=ssil_500   device="$DEVICE" training.epochs=100
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=ssil_1000   device="$DEVICE" training.epochs=100
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=ssil_2000   device="$DEVICE" training.epochs=100
#;;
replay)
  for epochs in 20 30 50 100
    do
      python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=er_ace_default method.mem_size=500 device="$DEVICE" experiment=base1 training.epochs=$epochs +wadnb_tags=[ablation]
    done
;;
#er_ace)
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=er_ace_default method.mem_size=200   device="$DEVICE"
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=er_ace_default method.mem_size=500   device="$DEVICE"
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=er_ace_default method.mem_size=1000   device="$DEVICE"
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=er_ace_default method.mem_size=2000   device="$DEVICE"
#;;
#gdumb)
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=gdumb method.mem_size=200   device="$DEVICE"
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=gdumb method.mem_size=500   device="$DEVICE"
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=gdumb method.mem_size=1000   device="$DEVICE"
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=gdumb method.mem_size=2000   device="$DEVICE"
#;;
der)
    for epochs in 20 30 50 100
    do
      python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=der_500  device=$DEVICE +model.head_classes=100 method.alpha=0.5 method.beta=0.8 head=linear experiment=base1 training.epochs=$epochs +wadnb_tags=[ablation]
    done
;;
*)
  echo -n "Unrecognized method"
esac
