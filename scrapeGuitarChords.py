# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 18:30:08 2016

@author: matt-666
"""

import requests
import re
from bs4 import BeautifulSoup
import pandas as pd
import pickle
import numpy as np

def removeTags(string):
    '''
    Function to remove html tags
    '''
    return re.sub('<[^<]+?>', '', string)


with open("scraping_df.p", "rb") as input_file:
    tot_df = pickle.load(input_file)

song_links = tot_df[(tot_df.tab_type == 'Chords') & (tot_df.difficulty.isin(['novice', 'intermediate', 'advanced']))].song_links

guitar_chords = []
for i, song_link in enumerate(song_links):
    try:    
        webpage = requests.get(song_link)
    except requests.exceptions.ConnectionError:
        print('Connection Lost')
        guitar_chords.append(None)
        continue
    webpageBS = BeautifulSoup(webpage.content)
    content = webpageBS.find('pre', {'class':'js-tab-content'})
    if content:
        song_chords = list(content.find_all('span')) 
        song_chords = [removeTags(str(chord)) for chord in song_chords]
        guitar_chords.append(song_chords)
    else:
        print('Tab not found')
        guitar_chords.append(None)
    if i%1000 == 0:
        pickleFile = open('all_labelled_scraped_guitar_chords.p', 'wb')
        pickle.dump((song_links.values[:i], guitar_chords), pickleFile)
        pickleFile.close()

        
pickleFile = open('all_labelled_scraped_guitar_chords.p', 'wb')
pickle.dump((song_links, guitar_chords), pickleFile)
pickleFile.close()
   