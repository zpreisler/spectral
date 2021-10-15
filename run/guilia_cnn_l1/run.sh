#!/bin/bash

for i in {1..48}
do
	nice -n 19 ../../train.py config.yaml > /dev/null 2> err$i &
done
