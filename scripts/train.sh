# EN in_domain

python finetuning.py -m microsoft/deberta-v3-xsmall -d wiki -l en
python finetuning.py -m microsoft/deberta-v3-xsmall -d ted -l en
python finetuning.py -m microsoft/deberta-v3-xsmall -d fanfic -l en
python finetuning.py -m microsoft/deberta-v3-xsmall -d news -l en

python finetuning.py -m microsoft/deberta-v3-small -d wiki -l en
python finetuning.py -m microsoft/deberta-v3-small -d ted -l en
python finetuning.py -m microsoft/deberta-v3-small -d fanfic -l en
python finetuning.py -m microsoft/deberta-v3-small -d news -l en

python finetuning.py -m microsoft/deberta-v3-base -d wiki -l en
python finetuning.py -m microsoft/deberta-v3-base -d ted -l en
python finetuning.py -m microsoft/deberta-v3-base -d fanfic -l en
python finetuning.py -m microsoft/deberta-v3-base -d news -l en