#!/bin/sh

# $1 -- name
# $2 -- epoch to extract weights from

# Extract weights from iteration $ITER as a pickled svector object
NAME=$1
ITER=$2
WEIGHTS_FILE=weights.`head -1 $NAME.out`
./extract-weights.py $WEIGHTS_FILE $ITER $NAME

