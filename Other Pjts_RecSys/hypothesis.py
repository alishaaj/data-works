import pandas as pd

df=pd.read_csv("dataset.csv")

dmdata=df.loc[df['criteria1']==2008]
ndmdata=df.loc[df['criteria1']!=2008]

dm=dmdata['context1']
ndm=ndmdata['context1']

#To remove NaN records
#dm=dmdata.dropna()['Ease']
#ndm=ndmdata.dropna()['Ease']

#Calculating Means
dm_mean=dm.mean()
ndm_mean=ndm.mean()

#Calculating Standard deviations
std1 = dm.std()
std2 = ndm.std()

#Calculating record count
N1 = dm.count()
N2 = ndm.count()
df = N1+N2-2

#z-test
#statsmodels.stats.weightstats.CompareMeans.ztest_ind
#CompareMeans.ztest_ind(alternative='two-sided', usevar='pooled', value=0)

#t-test
from scipy.stats import ttest_ind
ind_t_test=ttest_ind(dm,ndm) #equal_var=False

# crtical t-value at 95% CI using percent point function(ppf)
from scipy.stats import t
t_val = t.ppf([0.975], df)

# Calculate the mean difference, margin of error to calculate 95% confidence interval
# CI = mean difference (+ or -) Margin of Error (MoE)
from math import sqrt
diff_mean=dm_mean-ndm_mean
std_N1N2 = sqrt(((N1 - 1)*(std1)**2 + (N2 - 1)*(std2)**2) / df) 
MoE = t.ppf(0.975, df) * std_N1N2 * sqrt(1/N1 + 1/N2)

#test statistic-value and p-value
print('The results of the independent t-test are: \n\tt-value = ',ind_t_test[0],'\n\tp-value = ',ind_t_test[1])
#Confidence Interval
print ('\nThe difference between groups is {:3.3f} [{:3.3f} to {:3.3f}] (mean [95% CI])'.format(diff_mean, diff_mean - MoE, diff_mean + MoE))
#Margin of Error
print("Margin of Error: ",MoE)