#!/bin/bash
./train.sh base_train.yaml && ./train.sh accel_train.yaml && ./train.sh hf_train.yaml
