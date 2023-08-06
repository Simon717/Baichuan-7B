#!/bin/bash
deepspeed --num_gpus 1 train.py \
--deepspeed \
--deepspeed_config config/deepspeed.json
