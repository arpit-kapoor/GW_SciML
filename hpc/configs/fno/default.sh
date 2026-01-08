#!/bin/bash
# Default FNOInterpolate configuration (bilinear interpolation, 2 variables)

TRAIN_ARGS="
--epochs 10
--batch-size 256
--learning-rate 5e-4
--scheduler-type exponential
--target-cols mass_concentration head
--input-window-size 5
--output-window-size 1
--lambda-conc-focus 0.0
--padding-mode border
"
