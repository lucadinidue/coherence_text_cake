for D in fanfic news ted wiki
do
    python finetuning.py -m microsoft/deberta-v3-xsmall -d $D -l en -r 2e-5 -e 3
    python finetuning.py -m microsoft/deberta-v3-small -d $D -l en -r 2e-5 -e 3
    python finetuning.py -m microsoft/deberta-v3-base -d $D -l en -r 2e-5 -e 3
    python finetuning.py -m microsoft/deberta-v3-large -d $D -l en -r 8e-6 -e 3
done
