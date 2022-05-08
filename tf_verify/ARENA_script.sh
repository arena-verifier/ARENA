#!/bin/bash
gap=1
imgid=0
while (( $imgid <= 99))
do
    timeout -k 10 12600 python3 run_ARENA.py --imageid $imgid --netname mnist_relu_6_200.onnx --dataset mnist --epsilon 0.016 --is_refinement True
    imgid=$((imgid+gap))
done


