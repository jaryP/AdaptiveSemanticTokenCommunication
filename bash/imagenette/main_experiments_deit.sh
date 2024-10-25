#!/usr/bin/env bash

METHOD=$1
DEVICE=$2

case $METHOD in
adaptive)
  python main.py training_pipeline=imagenette224_vit16 pretraining_pipeline=imagenette224 +model=imagenette224_vit16 method=proposal device="$DEVICE"
  python main.py training_pipeline=imagenette224_vit16 pretraining_pipeline=imagenette224 model=deit_tiny_patch16_224 method=proposal device="$DEVICE"
  ;;
*)
  echo -n "Unrecognized method"
esac
