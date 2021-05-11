#! /bin/zsh

clear
#virtualenvdir=../stratification_ILRM/myvenv/bin/python

gpu=0

indir=$1
test_set=$2

if [ ! -z "$test_set" ]
then
    test_set="--test_set $test_set"
fi

#CUDA_VISIBLE_DEVICES=$gpu $virtualenvdir -u ./patient_representations.py $indir $test_set
CUDA_VISIBLE_DEVICES=$gpu python3 -u ./patient_representations.py $indir $test_set

# train
# sh learn-patient-representations.sh ./data

# test
# sh learn-patient-representations.sh ./data test

