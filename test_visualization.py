import json
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch

import data
import models
import pandas 
import numpy as np 
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt


def mae_accuracy(dataset) -> None:
    print("xxx")
    encoding, sentiment = dataset.yelp_reviews[0], dataset.yelp_reviews[1]
    # add 1 to turn into proper 5 star ratings
    preds = (model(encoding, sentiment).argmax(1) + 1).float().tolist()
    full_table = dataset.df
    print("hi")
    full_table['pred'] = preds
    full_table['dist'] = np.abs(full_table['pred'] - full_table['stars']) 
        
        
    print("MAE")
    print(np.sum(full_table['dist'])/500) 
        
    accurate = len(full_table[full_table['dist'] == 0])
    print("Accuracy")
    print(accurate/500)
            
def matrix(dataset1, dataset2) -> None:
    table_1 = dataset1.df
    table_2 = dataset2.df
    encoding1, sentiment1 = dataset1.yelp_reviews[0], dataset.yelp_reviews[1]
    # add 1 to turn into proper 5 star ratings
    preds1 = (model(encoding1, sentiment1).argmax(1) + 1).float().tolist()

    encoding2, sentiment2 = dataset2.yelp_reviews[0], dataset.yelp_reviews[1]
    # add 1 to turn into proper 5 star ratings
    preds2 = (model(encoding1, sentiment1).argmax(1) + 1).float().tolist()

    table_1['pred'] = preds1
    table_2['pred'] = preds2



    full_table = pandas.concat([table_1.df, table_2.df]) 
  
    confusion_mtx = confusion_matrix(full_table['stars'], full_table['pred']) 
    # plot the confusion matrix
    plot_confusion_matrix(confusion_mtx, classes = range(1,5))
    plt.show()
    confusion_matrix.savefig("matrix.png")

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
        

    #with Path("out_tmp.json").open("w") as f:
        #for line in d:
            #print(json.dumps(line), file=f)

if __name__ == "__main__":
    model = models.LanguageModel.load_from_checkpoint(
            "lightning_logs/version_110/checkpoints/epoch=1-step=87.ckpt",
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
    model.eval()
    dataset_5 = data.YelpDataset(model.args, sys.argv[1])
    dataset_8 = data.YelpDataset(model.args, sys.argv[2])

    mae_accuracy(dataset_5)
    # mae_accuracy(dataset_8)
    # matrix(dataset_5, dataset_8)
    
