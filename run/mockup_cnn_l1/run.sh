#!/bin/bash

for i in {1..12}
do
	nice -n 19 ../work.py config_init.yaml -n l3$i > /dev/null &
done
