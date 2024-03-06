from transformers import AutoModel, AutoTokenizer, default_data_collator
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import argparse
import shutil
import torch
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess_dataset(dataset, tokenizer):
    def preprocessing_function(examples):
        result = tokenizer(examples['text'], padding='max_length', max_length=256, truncation=True)
        return result

    tokenized_dataset = dataset.map(
        preprocessing_function,
        batched=True,
        remove_columns=['passage_id', 'text', 'label'],
        desc="Running tokenizer on dataset",
    )

    return tokenized_dataset


def prepare_dataloaders(dataset, language, batch_size, tokenizer):
    data_files = {'train': f'../data/datasets/{dataset}/{language}_train.tsv',
                  'validation': f'../data/datasets/{dataset}/{language}_eval.tsv'}

    dataset = load_dataset('csv', data_files=data_files, sep='\t')
    train_dataset = preprocess_dataset(dataset['train'], tokenizer)
    eval_dataset = preprocess_dataset(dataset['validation'], tokenizer)

    train_dataloader = DataLoader(train_dataset, shuffle=False, collate_fn=default_data_collator,
                                  batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=default_data_collator,
                                 batch_size=batch_size)

    return train_dataloader, eval_dataloader


def save_tensor_chunk(tensor, out_dir, chunk_idx):
    for layer_idx in range(tensor.shape[0]):
        layer_dir = os.path.join(out_dir, str(layer_idx))
        if not os.path.exists(layer_dir):
            os.makedirs(layer_dir)
        out_path = os.path.join(layer_dir, f'{chunk_idx}.pt')
        torch.save(tensor[layer_idx], out_path)


def extract_representations(model, dataloader, out_dir, slice):
    chunk_idx = 0
    all_hidden_states = None
    model.eval()
    with torch.no_grad():
        for batch_cpu in tqdm(dataloader):
            batch = {key: value.to(device) for key, value in batch_cpu.items()}
            hidden_states = model(**batch)['hidden_states']
            hidden_states = torch.stack(hidden_states, dim=1)

            non_pad_tokens = batch['attention_mask'].sum(axis=1)
            mask = batch['attention_mask'].unsqueeze(1).unsqueeze(3)  # batch_size, 1, max_sequence_len, 1
            mask = mask.expand(-1, hidden_states.shape[1], -1,
                               hidden_states.shape[-1])  # batch_size, num_layers + 1, max_sequence_len, hidden_size

            masked_hidden_states = hidden_states * mask
            masked_hidden_states = masked_hidden_states[:, 1:, :, :]
            sum_hidden_states = torch.sum(masked_hidden_states, dim=2)
            average_hidden_states = torch.div(sum_hidden_states,
                                              non_pad_tokens.view(-1, 1, 1))  # batch_size, num_layers + 1, hidden_size
            average_hidden_states = torch.reshape(average_hidden_states,
                                                  (average_hidden_states.shape[1], average_hidden_states.shape[0], -1))
            all_hidden_states = average_hidden_states if all_hidden_states is None else torch.cat(
                (all_hidden_states, average_hidden_states), dim=1)

            if slice and all_hidden_states.shape[1] > 10000:
                save_tensor_chunk(all_hidden_states, out_dir, chunk_idx)
                chunk_idx += 1
                all_hidden_states = None
        if all_hidden_states is not None:
            save_tensor_chunk(all_hidden_states, out_dir, chunk_idx)


def get_pretrained_model_name(model_name):
    if 'xsmall' in model_name:
        return 'microsoft/deberta-v3-xsmall'
    if 'small' in model_name:
        return 'microsoft/deberta-v3-small'
    if 'base' in model_name:
        return 'microsoft/deberta-v3-base'
    if 'large' in model_name:
        return 'microsoft/deberta-v3-large'
    else:
        raise Exception(f'Tokenizer not found for model {model_name}')


def extract_sentence_representations(model_name, language, dataset, out_dir, batch_size=8, slice=False):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    tokenizer_name = get_pretrained_model_name(model_name)
    print(f'Model = {model_name}')
    print(f'Tokenizer = {tokenizer_name}')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False, return_tensors='pt')

    print(f'Device = {device}')

    model.to(device)

    train_dataloader, eval_dataloader = prepare_dataloaders(dataset, language, batch_size, tokenizer)

    print('Extracting training set representations')
    train_out_dir = os.path.join(out_dir, 'train')
    extract_representations(model, train_dataloader, train_out_dir, slice)

    print('Extracting validation set representations')
    eval_out_dir = os.path.join(out_dir, 'eval')
    extract_representations(model, eval_dataloader, eval_out_dir, slice)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str)
    parser.add_argument('-l', '--language', type=str)
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-s', '--slice_tensor', action='store_true')
    args = parser.parse_args()

    extract_sentence_representations(args.model_name, args.language, args.dataset, args.batch_size, args.slice)


if __name__ == '__main__':
    main()
