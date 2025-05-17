#!/bin/bash

make

python3 performance_test.py data/updated_flower.csv data/test_flower.csv
python3 performance_test.py data/updated_mouse.csv data/test_mouse.csv