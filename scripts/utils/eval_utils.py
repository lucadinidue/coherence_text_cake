from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os


def load_labels_from_file(src_path):
    labels_list = []
    for line in open(src_path, 'r'):
        if line != '\n':
            labels_list.append(line.strip())
    return labels_list


def save_labels_to_file(labels_list, out_path):
    with open(out_path, 'w+') as out_file:
        for label in labels_list:
            out_file.write(f'{label}\n')


def save_predictions(p_label_ids, preds, id2label, out_dir):
    y_true = [id2label[y_true] for y_true in p_label_ids]
    y_pred = [id2label[y_pred] for y_pred in preds]

    predictions_dir = os.path.join(out_dir, 'predictions')
    if not os.path.exists(predictions_dir):
        os.mkdir(predictions_dir)
    save_labels_to_file(y_true, os.path.join(predictions_dir, 'y_true.txt'))
    save_labels_to_file(y_pred, os.path.join(predictions_dir, 'y_pred.txt'))


def save_evaluation_metrics(out_dir):
    predictions_dir = os.path.join(out_dir, 'predictions')
    y_true = load_labels_from_file(os.path.join(predictions_dir, 'y_true.txt'))
    y_pred = load_labels_from_file(os.path.join(predictions_dir, 'y_pred.txt'))

    c_r = classification_report(y_true=y_true, y_pred=y_pred)

    with open(os.path.join(predictions_dir, 'classification_report.txt'), 'w+') as out_file:
        out_file.write(c_r)

    cm_disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    cm_disp.plot(cmap='Blues', xticks_rotation='vertical', colorbar=False).figure_.savefig(
        os.path.join(predictions_dir, 'confusion_matrix.png'), bbox_inches='tight')

    cm_disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(10, 10))
    cm_disp.plot(cmap='Blues', xticks_rotation='vertical', ax=ax, colorbar=False).figure_.savefig(
        os.path.join(predictions_dir, 'confusion_matrix_perc.png'), bbox_inches='tight')
