# -*- coding: utf-8 -*-
"""
Created on Tue May 29 13:17:27 2018
Credits for dcg and ndcg fns: https://gist.github.com/bwhite/3726239
@author: alish
"""
import numpy as np
def dcg_at_k(r, k, method=0):
    """
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


testlist=[]
reclist=[]

#ranked list of relevant items from test set
for key,val in useritem_dict_test.items():
    testlist.append(val)
#ranked list of recommendations   
for key,val in top10rec_dict.items():
    reclist.append(val)    
 
revlist=[]
for i in range(len(reclist)):
    #i=18
    val=reclist[i]
    #print(val,"-", testlist[i])
    
    ls=[]
    for val in reclist[i]:
        index=reclist[i].index(val)
        if val not in testlist[i]:
            r=0
        else:
            if val in testlist[i]:
                if testlist[i].index(val)==index:
                    r=3
                else:
                    if testlist[i].index(val)<index+3:
                        r=2
                    else:
                        r=1
        ls.append(r)
    revlist.append(ls)        
    
ndcg10=[]
for i in range(len(revlist)):
    ndcg10.append(ndcg_at_k(revlist[i], 10))
    
ndcg5=[]
for i in range(len(revlist)):
    ndcg5.append(ndcg_at_k(revlist[i], 5))
    