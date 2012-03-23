#!/bin/bash
#PBS -l walltime=00:30:00,nodes=10:ppn=4
#PBS -N nile-train

cd $PBS_O_WORKDIR  # Connect to working directory
###################################################################
# Initialize MPI
###################################################################
export PATH=/home/nlg-03/riesa/mpich2-install/bin:$PATH
export PYTHONPATH=/home/nlg-03/riesa/boost_1_48_0/stage/lib:$PYTHONPATH
export LD_LIBRARY_PATH=/home/nlg-03/riesa/boost_1_48_0/stage/lib:$LD_LIBRARY_PATH
NUMCPUS=`wc -l $PBS_NODEFILE | awk '{print $1}'`
###################################################################

K=128
DATE=`date +%m%d%y`

BASEDIR=/home/nlg-03/riesa/projects/alignment
DATA=$BASEDIR/data
TRAIN=$DATA/train
LANGPAIR=zh-en

NAME=d$DATE.k${K}.n$NUMCPUS.$LANGPAIR

mpiexec -n $NUMCPUS $PYTHON nile.py \
  --f $TRAIN/train.f \
  --e $TRAIN/train.e \
  --etrees $TRAIN/train.e-parse \
  --ftrees $TRAIN/train.f-parse \
  --evcb $DATA/e.vcb \
  --fvcb $DATA/f.vcb \
  --pef $DATA/GIZA++.m4.pef  \
  --pfe $DATA/GIZA++.m4.pfe \
  --langpair $LANGPAIR \
  --train \
  --k $K 1> $NAME.out 2> $NAME.err
