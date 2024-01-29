from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

ID2LABEL = {
    '0': 'Orig',
    '1': 'sub_0',
    '2': 'sub_1',
    '3': 'sub_2',
    '4': 'sub_3',
    '5': 'swap_0_1',
    '6': 'swap_0_2',
    '7': 'swap_0_3',
    '8': 'swap_1_2',
    '9': 'swap_1_3',
    '10': 'swap_2_3'
}


def load_labels_from_file(src_path):
    labels_list = []
    for line in open(src_path, 'r'):
        if line != '\n':
            labels_list.append(ID2LABEL[line.strip()])
    return labels_list


def compute_confusion_matrix(results_dir):
    y_true = load_labels_from_file(os.path.join(results_dir, 'y_true.txt'))
    y_pred = load_labels_from_file(os.path.join(results_dir, 'y_pred.txt'))
    labels = list(ID2LABEL.values())
    cm_disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, normalize='true')

    fig, ax = plt.subplots(figsize=(10, 10))
    cm_disp.plot(cmap='Blues', xticks_rotation='vertical', ax=ax, values_format='.2g', colorbar=False).figure_.savefig(
        os.path.join(results_dir, 'confusion_matrix_perc.png'), bbox_inches='tight')


def main():
    results_dir = '../../models/deberta-v3-xsmall/4.5e-05/en_wiki/predictions'
    compute_confusion_matrix(results_dir)


if __name__ == '__main__':
    main()
