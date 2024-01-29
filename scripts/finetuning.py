from transformers import AutoModelForSequenceClassification, AutoTokenizer, default_data_collator, TrainingArguments, \
    Trainer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from datasets import load_dataset
import numpy as np
import evaluate
import argparse
import logging
import torch
import sys
import os


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
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language', choices=['en', 'es', 'id', 'it', 'nl'])
    parser.add_argument('-d', '--dataset', choices=['ted', 'wiki', 'news', 'fanfic'])
    parser.add_argument('-m', '--model')
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-r', '--learning_rate', type=float, default=5e-5)
    args = parser.parse_args()

    logger = init_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device = {device}')

    model_name = args.model
    data_files = {'train': f'../data/datasets/{args.dataset}/{args.language}_train.tsv',
                  'validation': f'../data/datasets/{args.dataset}/{args.language}_eval.tsv'}
    model_str = model_name.split('/')[-1]
    out_dir = f'../models/{model_str}/{args.learning_rate}/{args.language}_{args.dataset}'

    # 'microsoft/deberta-v3-xsmall'
    # data_files = {'train': '../data/datasets/wiki/en_train.tsv',
    #         'validation': '../data/datasets/wiki/en_eval.tsv'}

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

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        do_train=True,
        do_eval=False,
        evaluation_strategy='epoch',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=128,
        save_strategy='no',
        warmup_steps=500
    )

    metric = evaluate.load('accuracy')  # , cache_dir=training_args.cache_dir)

    # def compute_metrics_training(p):
    #     preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    #     preds = np.argmax(preds, axis=1)
    #     result = metric.compute(predictions=preds, references=p.label_ids)
    #     if len(result) > 1:
    #         result["combined_score"] = np.mean(list(result.values())).item()
    #     return result

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        y_true = [model.config.id2label[y_true] for y_true in p.label_ids]
        y_pred = [model.config.id2label[y_pred] for y_pred in preds]
        labels = list(model.config.label2id.keys())

        c_r = classification_report(y_true=y_true, y_pred=y_pred, target_names=labels)
        c_m = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)

        predictions_dir = os.path.join(out_dir, 'predictions')
        if not os.path.exists(predictions_dir):
            os.mkdir(predictions_dir)

        with open(os.path.join(predictions_dir, 'y_true.txt'), 'w+') as out_file:
            for el in p.label_ids:
                out_file.write(f'{el}\n')

        with open(os.path.join(predictions_dir, 'y_pred.txt'), 'w+') as out_file:
            for el in preds:
                out_file.write(f'{el}\n')

        with open(os.path.join(predictions_dir, 'classification_report.txt'), 'w+') as out_file:
            out_file.write(c_r)

        cm_disp = ConfusionMatrixDisplay(confusion_matrix=c_m, display_labels=labels)
        cm_disp.plot(cmap='Blues', xticks_rotation='vertical').figure_.savefig(
            os.path.join(predictions_dir, 'confusion_matrix.png'), bbox_inches='tight')
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
    trainer.save_model(out_dir)


if __name__ == '__main__':
    main()
