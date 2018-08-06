# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 12:26:52 2018

@author: alish
"""
#Convert JSON catery file to CSV format

#importing libraries
import pandas as pd
import json

#Loading data
data=r'US_category_id.json'

# Reads and converts json to dict.
def input_data(data):
   with open(data, encoding='utf-8') as file_in:
       return(json.load(file_in))

if __name__ == "__main__":
    my_dic_data = input_data(data)

#Extracting required attributes from dictionary    
keys= my_dic_data.keys()
dict_extract={'my_items':my_dic_data['items']for key in keys}

#Creating data frame with extracted data
df=pd.DataFrame(dict_extract)
df2=df['my_items'].apply(pd.Series)
df3=pd.concat([df2.drop(['snippet'],axis=1),df2['snippet'].apply(pd.Series)],axis=1)

print ("df3",df3)
df3.to_csv('csv_category_USA.csv')

                      
