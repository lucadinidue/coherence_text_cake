# EN in_domain

python finetuning.py -m microsoft/deberta-v3-xsmall -d wiki -l en -r 4.5e-5
python finetuning.py -m microsoft/deberta-v3-xsmall -d ted -l en -r 4.5e-5
python finetuning.py -m microsoft/deberta-v3-xsmall -d fanfic -l en -r 4.5e-5
python finetuning.py -m microsoft/deberta-v3-xsmall -d news -l en -r 4.5e-5

python finetuning.py -m microsoft/deberta-v3-small -d wiki -l en -r 4.5e-5
python finetuning.py -m microsoft/deberta-v3-small -d ted -l en -r 4.5e-5
python finetuning.py -m microsoft/deberta-v3-small -d fanfic -l en -r 4.5e-5
python finetuning.py -m microsoft/deberta-v3-small -d news -l en -r 4.5e-5

python finetuning.py -m microsoft/deberta-v3-base -d wiki -l en -r 2e-5
python finetuning.py -m microsoft/deberta-v3-base -d ted -l en -r 2e-5
python finetuning.py -m microsoft/deberta-v3-base -d fanfic -l en -r 2e-5
python finetuning.py -m microsoft/deberta-v3-base -d news -l en -r 2e-5

python finetuning.py -m microsoft/deberta-v3-large -d wiki -l en -r 6e-6
python finetuning.py -m microsoft/deberta-v3-large -d ted -l en -r 6e-6
python finetuning.py -m microsoft/deberta-v3-large -d fanfic -l en -r 6e-6
python finetuning.py -m microsoft/deberta-v3-large -d news -l en -r 6e-6