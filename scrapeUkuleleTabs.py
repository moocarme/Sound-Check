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

song_links = tot_df[tot_df.tab_type == 'Ukulele'].song_links

Ukulele_chords = []
for i, song_link in enumerate(song_links[5198:]):
    i += 5198
    webpage = requests.get(song_link)
    webpageBS = BeautifulSoup(webpage.content)
    content = webpageBS.find('pre', {'class':'js-tab-content'})
    if content:
        song_chords = list(content.find_all('span')) 
        song_chords = [removeTags(str(chord)) for chord in song_chords]
        Ukulele_chords.append(song_chords)
    else:
        print('Tab not found')
        Ukulele_chords.append(None)
    if i%1000 == 0:
        pickleFile = open('scraped_Ukulele_chords.p', 'wb')
        pickle.dump((song_links, Ukulele_chords), pickleFile)
        pickleFile.close()

        
pickleFile = open('scraped_Ukulele_chords.p', 'wb')
pickle.dump((song_links, Ukulele_chords), pickleFile)
pickleFile.close()
   