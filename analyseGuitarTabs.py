# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 19:41:28 2016

@author: matt-666
"""

import re
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split


with open("scraped_guitar_tabs_df.p", "rb") as input_file:
    guitar_tabs_df = pickle.load(input_file)

maxFret = 28
fretDivider = 1
frets = range(fretDivider, maxFret,fretDivider)

guitar_feat_df = pd.DataFrame(0, index = guitar_tabs_df.index, columns = frets)

 
for index, row in guitar_tabs_df.iterrows():
    tt1 = re.findall('[\s+]?\n?([A-Za-z0-9])?((\||:|-).+-.+)[^(\r|\n|\s)]',row.guitar_tabs)
    
    fretDistr = [sum(sublist[1].count("%s%s"%(num, '-')) for num in map(str, range(fret-fretDivider,fret)) for sublist in tt1) for fret in frets]
    guitar_feat_df.loc[index,frets] = fretDistr/sum(fretDistr)
    
    
tot_guitar_df = guitar_tabs_df.join(guitar_feat_df, how = 'inner').dropna()

with open("scraping_df.p", "rb") as input_file:
    labelled_df = pickle.load(input_file)

joined_guitar_df = tot_guitar_df.merge(labelled_df, on = 'song_links', how = 'inner',)\
                                .drop_duplicates()\
                                .dropna()   

# Train
X = joined_guitar_df.loc[:,frets]
y = joined_guitar_df.difficulty

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

crossVal = 5
C = np.logspace(-5,5,11)

lr_l2 = LogisticRegressionCV(cv = crossVal, Cs = C, penalty = 'l2', solver = 'liblinear')
lr_l2.fit(X_train, y_train)

y_pred = lr_l2.predict(X_test)
classif_rate_l2 = np.mean(y_pred.ravel() == y_test.ravel()) * 100

classes = lr_l2.classes_
y_probs = lr_l2.predict_proba(X_test)

plt.figure(666);plt.clf()
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

for i, class_ in enumerate(classes):
    y_class_probs = [x[i] for x in y_probs]
    fpr, tpr, thresholds = roc_curve(y_test, y_class_probs, pos_label = classes[i])
    roc_auc = auc(fpr, tpr)    
    plt.plot(fpr, tpr, lw=2, label='ROC class %s (area = %0.2f)' % (classes[i], roc_auc))

plt.xlim([-0.05, 1.05]); plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC - Predicting Guitar Difficulty')
plt.legend(loc="lower right"); plt.axis('equal')
