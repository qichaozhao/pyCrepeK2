from sklearn.metrics import confusion_matrix, classification_report
# import seaborn as sns
# import matplotlib.pyplot as plt
# % matplotlib inline
import importlib
import pandas as pd
import numpy as np


def evaluate_predictions(module_name):
    modelClass = importlib.import_module(module_name, package=None)

    tr_df = pd.read_csv('data/testing_metadata_v2.csv')
    y_true = np.asarray(tr_df['case_intensity'].tolist())

    trainer = modelClass.CaseTrainer()
    y_pred = trainer.run([], 'predict')

    # convert y_pred back into ordinal
    y_pred_ord = np.zeros(len(y_pred))
    y_pred_ord[np.arange(len(y_pred))] = np.argmax(y_pred, axis=1)

    sns.heatmap(confusion_matrix(y_true, y_pred_ord))

    print(module_name)
    print(classification_report(y_true, y_pred_ord))

if __name__ == '__main__':

    model_list = ['char_cnn_lstm',
                  'char_cnn_v1',
                  'char_cnn_v2',
                  'char_cnn_v3',
                  'lstm_metadata',
                  'lstm_v1']

    for model in model_list:
        evaluate_predictions(model)