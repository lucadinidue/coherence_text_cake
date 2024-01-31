from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
from spacy.lang.en import English
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os


def get_tokenizer(language):
    if language == 'en':
        nlp = English()
    else:
        raise Exception(f'Language {language} not implemented yet!')
    return nlp.tokenizer


def extract_passage_ngrams(passage_ngrams, sent_idx, tokens, n):
    for i in range(0, len(tokens) - n + 1):
        ngram_words = tokens[i: i + n]
        ngram = f'{sent_idx}_{n}_' + '_'.join(ngram_words)
        if ngram not in passage_ngrams:
            passage_ngrams[ngram] = 0
        passage_ngrams[ngram] += 1
    return passage_ngrams


def extract_dataset_ngrams(dataset, ns):
    result = []
    for passage in tqdm(dataset):
        passage_ngrams = dict()
        for sent_idx, sentence_id in enumerate(['sentence_0', 'sentence_1', 'sentence_2', 'sentence_3']):
            for n in ns:
                passage_ngrams = extract_passage_ngrams(passage_ngrams, sent_idx, passage[sentence_id], n)
        result.append(passage_ngrams)
    return result


def filter_features(train_dataset, min_occurrences):
    features_counter = dict()
    for passage in train_dataset:
        for feature in passage:
            if feature not in features_counter:
                features_counter[feature] = 0
            features_counter[feature] += 1

    for passage in train_dataset:
        passage_features = list(passage.keys())
        for feature in passage_features:
            if features_counter[feature] < min_occurrences:
                passage.pop(feature)

    return train_dataset


def prepare_datasets(dataset, language, ns):
    tokenizer = get_tokenizer(language)
    def tokenize_dataset(dataset):
        result = {}
        for sentence_id in ['sentence_0', 'sentence_1', 'sentence_2', 'sentence_3']:
            result[sentence_id] = [[token.text for token in doc] for doc in tokenizer.pipe(dataset[sentence_id])]
            # sentence_id_tokens = []
            # for doc in tokenizer.pipe(dataset[sentence_id]):
            #     tokens = [token.text for token in doc]
            #     sentence_id_tokens.append(tokens)
            # result[sentence_id] = sentence_id_tokens
        result['label'] = dataset['label']
        return result

    tokenized_train = dataset['train'].map(
        tokenize_dataset,
        batched=True,
        remove_columns=['passage_id'],
        desc="Running tokenizer train on dataset",
    )

    tokenized_eval = dataset['validation'].map(
        tokenize_dataset,
        batched=True,
        remove_columns=['passage_id'],
        desc="Running tokenizer eval on dataset",
    )

    y_train = tokenized_train['label']
    y_eval = tokenized_eval['label']

    train_ngrams = extract_dataset_ngrams(tokenized_train, ns)
    eval_ngrams = extract_dataset_ngrams(tokenized_eval, ns)

    filtered_train_ngrams = filter_features(train_ngrams, 5)

    vectorizer = DictVectorizer()
    scaler = MaxAbsScaler()

    X_train = vectorizer.fit_transform(filtered_train_ngrams)
    X_train = scaler.fit_transform(X_train)

    X_eval = vectorizer.transform(eval_ngrams)
    X_eval = scaler.transform(X_eval)

    return X_train, y_train, X_eval, y_eval


def save_metrics(y_true, y_pred, out_dir):
    c_r = classification_report(y_true=y_true, y_pred=y_pred)
    c_m = confusion_matrix(y_true=y_true, y_pred=y_pred)

    predictions_dir = os.path.join(out_dir, 'predictions')
    if not os.path.exists(predictions_dir):
        os.mkdir(predictions_dir)

    with open(os.path.join(predictions_dir, 'y_true.txt'), 'w+') as out_file:
        for el in y_true:
            out_file.write(f'{el}\n')

    with open(os.path.join(predictions_dir, 'y_pred.txt'), 'w+') as out_file:
        for el in y_pred:
            out_file.write(f'{el}\n')

    with open(os.path.join(predictions_dir, 'classification_report.txt'), 'w+') as out_file:
        out_file.write(c_r)

    cm_disp = ConfusionMatrixDisplay(confusion_matrix=c_m)
    cm_disp.plot(cmap='Blues', xticks_rotation='vertical', colorbar=False).figure_.savefig(
        os.path.join(predictions_dir, 'confusion_matrix.png'), bbox_inches='tight')

    cm_disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(10, 10))
    cm_disp.plot(cmap='Blues', xticks_rotation='vertical', ax=ax, colorbar=False).figure_.savefig(
        os.path.join(predictions_dir, 'confusion_matrix_perc.png'), bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language', choices=['en', 'es', 'id', 'it', 'nl'])
    parser.add_argument('-d', '--dataset', choices=['ted', 'wiki', 'news', 'fanfic'])
    parser.add_argument('-n', nargs="+", type=int)
    args = parser.parse_args()

    data_files = {'train': f'../data/baseline_datasets/{args.dataset}/{args.language}_train.tsv',
                  'validation': f'../data/baseline_datasets/{args.dataset}/{args.language}_eval.tsv'}

    n_string = '_'.join([str(n) for n in args.n])
    out_dir = f'../models/baseline/ngrams_{n_string}/{args.language}_{args.dataset}'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dataset = load_dataset('csv', data_files=data_files, sep='\t')

    X_train, y_train, X_eval, y_eval = prepare_datasets(dataset, args.language, args.n)

    svc = LinearSVC(dual=True, max_iter=10000)
    svc.fit(X_train, y_train)
    eval_predictions = svc.predict(X_eval)

    save_metrics(y_eval, eval_predictions, out_dir)


if __name__ == '__main__':
    main()
