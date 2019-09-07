# -*- coding: utf-8 -*-
"""
Created on Thu May 10 10:11:41 2018

@author: Vika Mello
"""
import pandas as pd
import xml.etree.ElementTree as ET
tree = ET.parse('YaMarket-yml_27_06.xml')
root = tree.getroot()

#print (root[0][10][0][1].text)
print ('Число офферов ' , len(root[0][10]))

ids = []
names = []
descriptions = []
urls = []
prices = []
urls_of_images = []
gtins = []
brands = []

i=50000
while i<57384:
    id = 'undefined'
    name = 'undefined'
    description = 'undefined'
    url = 'undefined'
    price = 'undefined'
    url_of_image = 'undefined'
    gtin = 'undefined'
    brand = 'undefined'
    
    typePrefix = 'undefined'
    model = 'undefined'
    
    #print (root[0][10][i][0].text)
    
    id = root[0][10][i].attrib['id']
    for node in root[0][10][i].getiterator():
        
        if node.tag=='vendor':
            brand = node.text
        
        if node.tag=='typePrefix':
            typePrefix = node.text
            
        if node.tag=='model':
            model = node.text
            
        name = typePrefix + ' ' + model
        
        if node.tag=='description':
            description = node.text
            
        if node.tag=='url':
            url = node.text
            
        if node.tag=='price':
            price = node.text + ' RUB'
            
        if node.tag=='picture':
            url_of_image = node.text
            
        if node.tag=='barcode':
            gtin = node.text
            
    ids.append(id)
    brands.append(brand)
    names.append(name)
    descriptions.append(description)
    urls.append(url)
    prices.append(price)
    urls_of_images.append(url_of_image)
    gtins.append(gtin)
    
    i=i+1

df_brands = pd.DataFrame(brands, columns=['марка'])
df_ids = pd.DataFrame(ids, columns=['id'])
df_names = pd.DataFrame(names, columns=['название'])
df_descriptions = pd.DataFrame(descriptions, columns=['описание'])
df_urls = pd.DataFrame(urls, columns=['ссылка'])
df_prices = pd.DataFrame(prices, columns=['цена'])
df_urls_of_images = pd.DataFrame(urls_of_images, columns=['ссылка на изображение'])
df_gtins = pd.DataFrame(gtins, columns=['gtin'])

frames = [df_ids, df_names, df_descriptions, df_brands, df_urls, df_prices, df_urls_of_images, df_gtins]
result = pd.concat(frames, axis=1)
result.to_excel('output_YaMarket-yml_27_06_5.xlsx','Sheet1')