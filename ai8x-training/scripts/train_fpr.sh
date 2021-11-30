#!/bin/sh
./train.py --epochs 200 --optimizer Adam --lr 0.001 --batch-size 64 --gpus 3 --deterministic --compress schedule-fpr.yaml --model ai85net_fpr --dataset fpr --param-hist --pr-curves --embedding --device MAX78000 "$@"
