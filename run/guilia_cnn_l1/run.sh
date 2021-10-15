#!/bin/bash

for i in {0.3,0.2,0.1,0.05,0.025,0.0125,0.005}
do
	nice -n 19 ../../train.py config.yaml -l $i -n train_CNNSkip_cpu_lr_$i > /dev/null 2> err$i &
done
