#!/bin/bash
#PBS -l walltime=00:30:00,nodes=10:ppn=4
#PBS -N test.zh

echo "Started" `date`
echo "Initiated on `hostname`"
echo ""

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
LANGPAIR=ce
DESCRIPTION=gold-1best.target-tree.0

BASEDIR=/home/nlg-03/riesa/projects/alignment
DATA=$BASEDIR/data
TEST=$DATA/test

WEIGHTS=my-training-run.weights-22
NAME=$WEIGHTS.test-output.a

mpiexec -n $NUMCPUS $PYTHON nile.py \
  --f $TEST/test.f \
  --e $TEST/test.e \
  --etrees $TEST/test.e-parse \
  --ftrees $TEST/test.f-parse \
  --evcb $TEST/test.e.vcb \
  --fvcb $GOLD/test.f.vcb \
  --langpair zh-en \
  --pef $DATA/GIZA++.m4.pef  \
  --pfe $DATA/GIZA++.m4.pfe \
  --align \
  --weights $WEIGHTS \
  --out $NAME \
  --k $K
