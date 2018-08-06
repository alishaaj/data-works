# -*- coding: utf-8 -*-
"""
Created on Wed May 23 16:52:33 2018

@author: alish
"""
#----Train-----------
import pandas as pd

# Importing the dataset
rating_data = pd.read_csv('train5.csv')
traindata=rating_data[['User','Item','Rating']]
#selecting positive irems ; rating>3
#posdata_train=traindata.loc[traindata['Rating']>3]
posdata_train=traindata.sort_values(['User', 'Rating'], ascending=[True, False])
#posdata=posdata.sort_values(by='User')
userdata_train=posdata_train[['User','Item']]

groupuserdata_train = userdata_train.groupby('User')
print(list(groupuserdata_train))

users_train=[]
items_train=[]
i=0
from collections import OrderedDict
for name,group in groupuserdata_train:
    users_train.append(name)
    items_train.append(list(OrderedDict.fromkeys(group['Item'])))
    
useritem_dict_train=dict(zip(users_train, items_train))

#----Test-----------
import pandas as pd

# Importing the dataset
rating_data_t = pd.read_csv('test5.csv')
testdata=rating_data_t[['User','Item','Rating']]
#selecting positive irems ; rating>3
#posdata_test=testdata.loc[testdata['Rating']>3]
posdata_test=testdata.sort_values(['User', 'Rating'], ascending=[True, False])
#posdata=posdata.sort_values(by='User')
userdata_test=posdata_test[['User','Item']]

groupuserdata_test = userdata_test.groupby('User')
print(list(groupuserdata_test))

#dictionary of users and actual selections
users_test=[]
items_test=[]
i=0
from collections import OrderedDict
for name,group in groupuserdata_test:
    users_test.append(name)
    items_test.append(list(OrderedDict.fromkeys(group['Item'])))
    
useritem_dict_test=dict(zip(users_test, items_test))

#get recommended list by decreasing order of similarity
n=0
i=0  
toplist=[]
for key,val in useritem_dict_test.items():
    print(key)
    b=usersim_dict.get(key)
    inlist1=sorted(range(len(b)), key=lambda i: b[i], reverse=True)
    toplist.append([x+1 for x in inlist1])
    
toplist_dict=dict(zip(users_test, toplist))

#get top 10 recommendation from recommended list not in test list
top10rec_dict={}


for key,val in toplist_dict.items():
    listrc=[]
    c=0
    for i in range(len(val)):
        if val[i] not in useritem_dict_train.get(key):
            listrc.append(val[i])
            c+=1
        if c==10:
            top10rec_dict[key]=listrc
            break


    