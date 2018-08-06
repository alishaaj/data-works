# -*- coding: utf-8 -*-
# Calculate Precision and Recall with test list and recommended list
"""
Created on Thu May 24 12:18:26 2018

@author: alish
"""


testlist=[]
reclist=[]

#adding actual user list to testlist
for key,val in useritem_dict_test.items():
    testlist.append(val)

#adding recommended list to reclist   
for key,val in top10rec_dict.items():
    reclist.append(val)    
    
#----True Positives----
tp1=[]
tp5=[]
tp10=[]        
for i in range(len(testlist)):
    #i=7
    val=testlist[i]
    #print(val,"-", reclist[i])
    tp1count=0
    tp5count=0
    tp10count=0
    for val in testlist[i]:
        if val in reclist[i][:1]:
                tp1count+=1
        if val in reclist[i][:5]:
                tp5count+=1
        if val in reclist[i]:
                tp10count+=1
    #print(tp1count)
    tp1.append(tp1count)
    #print(tp5count)
    tp5.append(tp5count)
    #print(tp10count)
    tp10.append(tp10count)
    
#----False Positives-----
fp1=[]
fp5=[]
fp10=[]        
for i in range(len(reclist)):
    #i=7
    val=reclist[i]
    #print(val,"-", testlist[i])
    fp1count=0
    fp5count=0
    fp10count=0
    for val in reclist[i][:1]:
        if val not in testlist[i]:
                fp1count+=1
    for val in reclist[i][:5]:
        if val not in testlist[i]:
                fp5count+=1
    for val in reclist[i]:
        if val not in testlist[i]:
                fp10count+=1
    #print(fp1count)
    fp1.append(fp1count)
    #print(fp5count)
    fp5.append(fp5count)
    #print(fp10count)
    fp10.append(fp10count)
    
#----False Negatives-----
#values in test, but not in rec
fn1=[]
fn5=[]
fn10=[]        
for i in range(len(testlist)):
    #i=7
    val=testlist[i]
    #print(val,"-", reclist[i])
    fn1count=0
    fn5count=0
    fn10count=0
    for val in testlist[i]:
        if val not in reclist[i][:1]:
            fn1count+=1
        if val not in reclist[i][:5]:
            fn5count+=1
        if val not in reclist[i]:
            fn10count+=1
    #print(fn1count)
    fn1.append(fn1count)
    #print(fn5count)
    fn5.append(fn5count)
    #print(fn10count)
    fn10.append(fn10count)
    
#---Prec@10---
prec10=[]
for i in range(len(tp10)):
    prec10.append(tp10[i]/(tp10[i]+fp10[i]))  
    
#---Recall@10---
recall10=[]
for i in range(len(tp10)):
    recall10.append(tp10[i]/(tp10[i]+fn10[i])) 
    
#---Prec@5---
prec5=[]
for i in range(len(tp5)):
    prec5.append(tp5[i]/(tp5[i]+fp5[i]))  
    
#---Recall@10---
recall5=[]
for i in range(len(tp5)):
    recall5.append(tp5[i]/(tp5[i]+fn5[i]))