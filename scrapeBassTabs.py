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

Num2Scrape = 10000
classes = ['novice', 'intermediate', 'advanced']
song_links = tot_df[(tot_df.tab_type == 'Bass Tabs') & (tot_df.difficulty.isin(classes))].song_links[Num2Scrape:]

bass_tabs = []
i = 0
for song_link in song_links:
    webpage = requests.get(song_link)
    webpageBS = BeautifulSoup(webpage.content)
    content = removeTags(str(webpageBS.find('pre', {'class':'js-tab-content'})))
    if content:
        #print(str(content)[:20])
        bass_tabs.append(content)
    else:
        print('Tab not found')
        bass_tabs.append(None)
    if i%1000 == 0:
        pickleFile = open('more_scraped_bass_tabs.p', 'wb')
        pickle.dump((song_links, bass_tabs), pickleFile)
        pickleFile.close()
        print(i)
    i += 1
        
# Put into pandas df and pickle
d = {'song_links': song_links, 'bass_tabs':bass_tabs}

bass_tabs_df = pd.DataFrame(d).dropna()

pickleFile = open('more_scraped_bass_tabs_df.p', 'wb')
pickle.dump((bass_tabs_df), pickleFile)
pickleFile.close()
