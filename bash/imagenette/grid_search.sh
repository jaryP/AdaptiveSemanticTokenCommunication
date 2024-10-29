#!/usr/bin/env bash
METHOD=$1
MODEL=$2
DEVICE=$3

#max_jobs=$4
#num_jobs="\j"

case $METHOD in
der)
  for memory in 200 500 1000 2000
    do
    for alpha in 0.1 0.2 0.5 0.8 1.0
    do
      for beta in 0.1 0.2 0.5 0.8 1.0
      do
        python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=der device=$DEVICE head=linear method.alpha=$alpha method.beta=$beta experiment=dev method.mem_size=$memory hydra=search +wadnb_tags=[grid_search]
      done
    done
  done
;;
lode)
  for memory in 200 500 1000 2000
    do
    for rho in 0.1 0.2 0.5 0.8 1.0
    do
      python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=lode device=$DEVICE head=incremental method.rho=$rho experiment=dev method.mem_size=$memory hydra=search +wadnb_tags=[grid_search]
    done
  done
;;
margin)
for memory in 200 500 1000 2000
do
  for past_margin_w in 1 0.5 0.25 0.1 0.05 0.025 0.01
  do
    while (( ${num_jobs@P} >= $((max_jobs + 0)) )); do
      wait -n
    done
    python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=margin head=margin_head device=$DEVICE method.mem_size=$memory method.past_task_reg=$past_margin_w method.gamma=1 +wadnb_tags=[grid_search_margin] method.margin_type=adaptive experiment=dev hydra=search &
  done
done
;;
margin1)
echo
for memory in 200 500 1000 2000
do
  for past_margin_w in 0.5 0.25 0.1 0.01
  do
    python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=margin head=margin_head device=$DEVICE method.mem_size=$memory method.past_task_reg=$past_margin_w method.gamma=1 +wadnb_tags=[grid_search_margin] method.margin_type=adaptive head.scale_each_class=False experiment=dev hydra=search
  done
done
;;
logit_d)
  for memory in 200 500 1000 2000
  do
    for alpha in 0.1 0.25 0.5 0.75 1
    do
      while (( ${num_jobs@P} >= ${max_jobs:-1} )); do
        wait -n
      done
      python main.py +scenario=cil_cifar10_5 model=$MODEL +training=cifar10_5 +method=logit_d device=$DEVICE method.mem_size=$memory method.alpha=$alpha hydra=search experiment=dev head=incremental +wadnb_tags=[grid_search] &
    done
  done
;;
*)
  echo -n "Unrecognized method"
esac
