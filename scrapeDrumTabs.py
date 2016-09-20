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

song_links = tot_df[tot_df.tab_type == 'Drum Tabs'].song_links

drum_tabs = []
for song_link in song_links:
    webpage = requests.get(song_link)
    webpageBS = BeautifulSoup(webpage.content)
    content = removeTags(str(webpageBS.find('pre', {'class':'js-tab-content'})))
    if content:
        print(str(content)[:20])
        drum_tabs.append(content)
    else:
        print('Tab not found')
        drum_tabs.append(None)
        
pickleFile = open('scraped_drum_tabs.p', 'wb')
pickle.dump((song_links, drum_tabs), pickleFile)
pickleFile.close()
