from transformers import AutoModelForSequenceClassification, AutoTokenizer, default_data_collator, TrainingArguments, \
    Trainer
from eval_utils import save_predictions
import numpy as np
import evaluate
import logging
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


def preprocess_dataset(dataset, tokenizer, label_to_id):
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


def train_model(model_name, dataset, logger, out_dir, args):
    label_list = list(set(dataset['train']['label']))
    label_list.sort()
    num_labels = len(label_list)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, return_tensors='pt')

    label_to_id = {v: i for i, v in enumerate(label_list)}
    model.config.label2id = label_to_id
    model.config.id2label = {label_id: label for label, label_id in model.config.label2id.items()}

    train_dataset = preprocess_dataset(dataset['train'], tokenizer, label_to_id)
    eval_dataset = preprocess_dataset(dataset['validation'], tokenizer, label_to_id)

    per_device_train_batch_size = 8 if 'large' not in model_name else 4
    per_device_eval_batch_size = 128 if 'large' not in model_name else 64
    
    logger.info(f"Dataset is ready!")

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        do_train=True,
        do_eval=False,
        evaluation_strategy='epoch',
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        save_strategy='no',
        warmup_steps=500
    )

    metric = evaluate.load('accuracy')  # , cache_dir=training_args.cache_dir)

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()

        save_predictions(p.label_ids, preds, model.config.id2label, out_dir)

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
