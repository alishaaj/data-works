# -*- coding: utf-8 -*-
"""
Created on Wed May 23 11:03:36 2018

@author: alish
"""
#----Train-----------
import pandas as pd

# Importing the dataset
rating_data = pd.read_csv('train5.csv')
traindata=rating_data[['User','Item','Rating']]
#selecting positive irems ; rating>3
posdata_train=traindata.loc[traindata['Rating']>3]
posdata_train=posdata_train.sort_values(['User', 'Rating'], ascending=[True, False])
#posdata=posdata.sort_values(by='User') for single column
userdata_train=posdata_train[['User','Item']]

groupuserdata_train = userdata_train.groupby('User')
print(list(groupuserdata_train))

users_train=[]
items_train=[]
i=0
from collections import OrderedDict
for name,group in groupuserdata_train:
#    print (name)
#    print (list(group['Item']))
    users_train.append(name)
    items_train.append(list(OrderedDict.fromkeys(group['Item'])))
    
useritem_dict_train=dict(zip(users_train, items_train))

uservec=[]
for key,val in useritem_dict_train.items():
    uservec.append(val)
    

#import item vectors as array
itemdata = pd.read_csv('itemvector.csv')
itemvec = itemdata.as_matrix()


for idx in range(len(uservec)):
    for jx in range(len(uservec[idx])):
        uservec[idx][jx] = itemvec[uservec[idx][jx]-1]      
    
#calculating mean vector for each user in it
for i in range(len(uservec)):
    sum=0;
    for j in range(len(uservec[i])):  
        sum+=uservec[i][j]
    uservec[i]=sum/len(uservec[i])
    

# Similarity between each user vector and item vector
from sklearn.metrics.pairwise import cosine_similarity
usersim = cosine_similarity(uservec,Y=itemvec) 

usersim_dict=dict(zip(users_train, usersim))

#fold 3 special case for missing user 1185
s=0
for t in range(len(usersim)):
        s+=usersim[t]
mean=s/len(usersim)
usersim_dict[1185]=mean



