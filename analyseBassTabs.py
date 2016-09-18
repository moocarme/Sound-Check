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
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sqlalchemy import create_engine
import psycopg2


with open("more_scraped_bass_tabs_df.p", "rb") as input_file:
    bass_tabs_df = pickle.load(input_file)

maxFret = 24
fretDivider = 1
frets = range(fretDivider, maxFret,fretDivider)
feats = range(fretDivider, maxFret,fretDivider)
total_frets = len(frets)
feats.append('quick_notes')
feats.append('h_count')
feats.append('p_count')
feats.append('slides')
feats.append('hp_count')
feats.append('ringers')
feats.append('mean_fret')
feats.append('fret_std')
feats.append('pc_filled')

bass_feat_df = pd.DataFrame(0, index = bass_tabs_df.index, columns = feats)
 
for index, row in bass_tabs_df.iterrows():
    tt1 = re.findall('[\s+]?\n?([A-Za-z0-9])?((\||:|-)[^\sA-Za-z].+-.+)[^(\r|\n|\s)]',row.bass_tabs)
    
    fretDistr = [sum(sublist[1].count("%s%s"%(num, '-')) for num in map(str, range(fret-fretDivider,fret)) for sublist in tt1) for fret in frets]
    normFretDistr = fretDistr/sum(fretDistr)
    mean_fret = sum(normFretDistr*np.arange(total_frets))    
    fret_std = np.sqrt(sum((np.arange(total_frets) - mean_fret)**2*normFretDistr))
    
    bass_feat_df.loc[index, 'pc_filled'] = np.mean([1-str_info[1].count('-')/len(str_info[1]) for str_info in tt1])
    bass_feat_df.loc[index, frets] = normFretDistr
    bass_feat_df.loc[index, 'mean_fret'] = mean_fret
    bass_feat_df.loc[index, 'fret_std'] = fret_std
    bass_feat_df.loc[index, 'quick_notes'] =  np.mean([len(x) for x in re.findall('[0-9][0-9]+',row.bass_tabs)])
    bass_feat_df.loc[index, 'h_count'] = len(re.findall('[0-9]h', row.bass_tabs))
    bass_feat_df.loc[index, 'p_count'] = len(re.findall('[0-9]p', row.bass_tabs))
    bass_feat_df.loc[index, 'slides'] = len(re.findall('/', row.bass_tabs))
    bass_feat_df.loc[index, 'hp_count'] = len(re.findall('[0-9]h[0-9]+p[0-9]', row.bass_tabs))
    bass_feat_df.loc[index, 'ringers'] = len(re.findall('~', row.bass_tabs))
bass_feat_df.dropna()
    
tot_bass_df = bass_tabs_df.join(bass_feat_df, how = 'inner').dropna()

with open("scraping_df.p", "rb") as input_file:
    labelled_df = pickle.load(input_file)

joined_bass_df = tot_bass_df.merge(labelled_df, on = 'song_links', how = 'inner',)\
                                .drop_duplicates()\
                                .dropna()   

# Train
X = joined_bass_df.loc[:,feats]
y = joined_bass_df.difficulty

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

crossVal = 5
C = np.logspace(-5,5,11)

lr_l2 = LogisticRegressionCV(cv = crossVal, Cs = C, penalty = 'l1', solver = 'liblinear')
lr_l2.fit(X_train, y_train)

y_pred = lr_l2.predict(X_test)
classif_rate_l2 = np.mean(y_pred.ravel() == y_test.ravel()) * 100

classes = lr_l2.classes_
y_probs = lr_l2.predict_proba(X_test)

plt.figure(667);plt.clf()
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

for i, class_ in enumerate(classes):
    y_class_probs = [x[i] for x in y_probs]
    fpr, tpr, thresholds = roc_curve(y_test, y_class_probs, pos_label = classes[i])
    roc_auc = auc(fpr, tpr)    
    plt.plot(fpr, tpr, lw=2, label='ROC class %s (area = %0.2f)' % (classes[i], roc_auc))

plt.xlim([-0.05, 1.05]); plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC - Predicting Bass Difficulty')
plt.legend(loc="lower right"); plt.axis('equal')

# =================================================
# insert into db

dbname = 'reco_db'
user = 'matt-666'
db = create_engine('postgresql://matt-666:matt-666@localhost:5432/%s'%(dbname))

con = None
con = psycopg2.connect(dbname = dbname, user = user, host='localhost', password='everything')
cur = con.cursor()

cur.execute("ALTER TABLE scraped_info ADD %s TEXT" % ('predicted_difficulty'))
con.commit()
# get song_links
song_links = []
for i, song_link in enumerate(song_links):
    cur.execute("UPDATE scraped_info SET predicted_diffculty = %s WHERE song_link = %s")