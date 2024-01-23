from transformers import AutoModelForSequenceClassification, AutoTokenizer, default_data_collator, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from datasets import load_dataset
import pandas as pd
import numpy as np
import evaluate
import logging
import torch
import ast
import sys

def init_logging():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    return logger



def main():
    logger = init_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device = {device}')

    model_name = 'microsoft/deberta-v3-xsmall'
    data_files = {'train': '../data/datasets/wiki/en_train.tsv',
            'validation': '../data/datasets/wiki/en_eval.tsv'}

    dataset = load_dataset('csv', data_files=data_files, sep='\t') 
    logger.info(f"Dataset loaded!")

    label_list = list(set(dataset['train']['label']))
    label_list.sort()
    num_labels = len(label_list)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, return_tensors='pt')

    label_to_id = {v: i for i, v in enumerate(label_list)}
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in model.config.label2id.items()}

    def preprocess_dataset(dataset, tokenizer):
    
        def preprocessing_function(examples):
            result = tokenizer(examples['text'], padding='max_length', max_length=256, truncation=True)
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
            # result['label'] = [1 if perturb_type is None else 0 for perturb_type in examples['label']]
            return result
    
        tokenized_dataset = dataset.map(
                preprocessing_function,
                batched=True,
                remove_columns=['passage_id', 'text'],
                desc="Running tokenizer on dataset",
            )
    
        return tokenized_dataset

    train_dataset = preprocess_dataset(dataset['train'], tokenizer)
    eval_dataset = preprocess_dataset(dataset['validation'], tokenizer)

    logger.info(f"Dataset is ready!")

    out_dir = 'prova'
    
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=1,
        do_train=True,
        do_eval=True,
        evaluation_strategy='steps',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_strategy="no",
    )

    metric = evaluate.load('accuracy')#, cache_dir=training_args.cache_dir)

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    data_collator = default_data_collator

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model('prova')
    model.eval()


if __name__ == '__main__':
    main()
