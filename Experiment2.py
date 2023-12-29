import pandas as pd
import numpy as nm
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.svm import SVC
import math
from scipy.stats import norm
import scipy.stats
from scipy.stats import ks_2samp
from scipy import stats
import random
import matplotlib.pyplot as plt
from statistics import variance

random.seed(1)


df = pd.read_csv("C:\\Users\\Thanasis Moustakas\\Desktop\\Paper 5\\australia\\aus.csv", sep=",")
#df = df.drop(['id','date','day','period','nswprice','vicprice','transfer'],axis = 1)
df = df.drop(['id','date','day','period'],axis = 1)
DS = list()
for h in range(3):
    DS.append(df.iloc[15000*h:15000*(h+1),:])
    
 
DN = list()
targets = list()
for h in range(3):  
    for i in range(3):
        help1 = DS[h].iloc[5000*i:5000*(i+1),:]
        y = help1.pop('class')
        X = help1
        DN.append(X)
        targets.append(y)
'''    
for i in range(9):
    sum1 = 0
    for j in range(6):
        if (ks_2samp(DN[i].iloc[:,j][0:500],DN[4].iloc[:,j][500:1000])[1]>0.8):
            sum1 = sum1 + 1
    print(sum1/6) 
'''

k = 10
d = 5

'Train the models in the datasets for the first three periods'

pm = nm.empty((9,k))
pmN =  nm.empty((9,k)) 
md = list()
lm = nm.empty(9)
md2 = list()
lm2 = nm.empty(9)
EMWA = nm.empty((9,k))
Var = nm.empty((9,k))
ucl = nm.empty((9,k))
lcl = nm.empty((9,k))
for i in range(9):
    X = DN[i][0:1500]
    y = targets[i][0:1500]
                
        
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X, y)
    #print( model.score(X, y))
    md.append(model.coef_)
    lm[i] = model.intercept_
    md2.append(model.coef_)
    lm2[i] = model.intercept_
    pm[i][0] = model.score(X, y)
    pm[i][1] = model.score(X, y)
    pm[i][2] = model.score(X, y)
    pmN[i][0] = model.score(X, y)
    pmN[i][1] = model.score(X, y)
    pmN[i][2] = model.score(X, y)
    
    
for i in range(9):
    EMWA[i][0] = pm[i][0]
    EMWA[i][1] = pm[i][1]
    EMWA[i][2] = pm[i][2]
    Var[i][0] = 0
    Var[i][1] = 0
    Var[i][2] = 0.01

m = nm.empty((9,k,d))
v = nm.empty((9,k,d))   
for j in range(k):
    for i in range(9):
        for l in range(d):
            m[i][j][l] = DN[i].iloc[:,l][j*500:(j+1)*500].mean()
            v[i][j][l] = variance(DN[i].iloc[:,l][j*500:(j+1)*500])
        
W = 3

mbase = nm.empty((9,d))
vbase = nm.empty((9,d))
for i in range(9):
    for l in range(d):
        sum1  = 0
        sum2  = 0
        for j in range(W):
            sum1 = sum1 + m[i][j][l]
            sum2 = sum2 + v[i][j][l]
        mbase[i][l] = sum1/W
        vbase[i][l] = sum2/W

load = nm.empty((9,k))
queries = nm.empty((9,k))
offload = nm.empty((9,k))

for i in range(9):
    for j in range(k):
        load[i][j] = random.uniform(0.3,0.7)
        offload[i][j] = load[i][j]
        
#THRESHOLDS!!!
load1 = 0.9        
Dm = 0.4
Dv = 0.2
thr1 = 0.9 #α
la = 0.6 #λ
de = 1.5 #δ
beta = 0.8
gamma = 0.8

str2 = "Case8"

"PERFORMANCE DRIFT"
for j in range(W,k):   
    "DATA DRIFT"           
    mnew = nm.empty((9,d))
    vnew = nm.empty((9,d))
    for i in range(9):
        sum3 = 0
        for l in range(d):
            sum1  = 0
            sum2  = 0
            for g in range(j-W+1,j):
                sum1 = sum1 + m[i][g][l]
                sum2 = sum2 + v[i][g][l]
            mnew[i][l] = sum1/W
            vnew[i][l] = sum2/W
            if ( abs(mbase[i][l] - mnew[i][l]) >= Dm or abs(vbase[i][l] - vnew[i][l]) >= Dv ):
                sum3 = sum3 +1
        dd = sum3/d
        if (dd >= thr1):
            #print(" Data Drift detected for node:", i ,"in period:", j )
            mbase[i] = mnew[i] 
            vbase[i]  = vnew[i] 
    for i in range(9):    
        X = DN[i][500*j:500*(j+1)]
        y = targets[i][500*j:500*(j+1)]
        model.intercept_ = lm[i]
        model.coef_ = md[i]
        model.predict_proba(X)
        ypred = model.predict(X)
        pm[i][j] = model.score(X,y)
        #print('period:',j,'node:',i,'score:',pm[i][j])
        #pmN[i][j] = pm[i][j] 
        EMWA[i][j] = la*pm[i][j] + (1-la)*EMWA[i][j-1]
        Var[i][j] =  la*(pm[i][j] - EMWA[i][j])**2 + (1-la)*Var[i][j-1]
        ucl[i][j] = EMWA[i][0] + de*math.sqrt(Var[i][j])
        lcl[i][j] = EMWA[i][0] - de*math.sqrt(Var[i][j])
        if (EMWA[i][j] > ucl[i][j] ) or (EMWA[i][j] < lcl[i][j] ) :
            pd = 1
            EMWA[i][0] = EMWA[i][j]
        else:
            pd = 0
        if (pd or dd ):
            #print("Performance Drift detected for node:",i,"in period:",j+1)  
            load[i][j] = load[i][j] + 0.6
            model2 = LogisticRegression(solver='liblinear', random_state=0)
            model2.fit(X, y)
            md2[i] = model2.coef_
            lm2[i] = model2.intercept_
            pmN[i][j] = model2.score(X, y)
            #print(pm[i][j],pmN[i][j])
        else:
            pmN[i][j] = pm[i][j]
        #if (dd):
            #print("DD")
        if (offload[i][j]>=0.9):
            sim = nm.empty(9)
            
            for p in range(9):
                if (p != i):
                    sum2 = 0
                    for l in range(d):
                        #print(ks_2samp(DS[i][l], DS[j][l])[1])
                        if(ks_2samp(DN[i].iloc[:,l][(j-1)*500:j*500], DN[p].iloc[:,l][(j-1)*500:j*500])[1] < beta):
                            sum2 = sum2 + 0
                        else:
                            sum2 = sum2 + 1

                    if(sum2/d < beta):
                        sim[p] = 0
                    else:
                        sim[p] = sum2/d
                else:
                    sim[p] = 0
            #print(i,j,sim)
            print(sim)
            diff = offload[i][j] - load1
            for p in range(10):
                if (sim[p] > gamma):
                    if (offload[p][j] + diff <= load1):
                        #print('Send queries from node: ',i,'to node:', p)
                        offload[i][j] = offload[i][j] - diff
                        offload[p][j] = offload[p][j] + diff
                        break
                    else:
                        ndiff = load1 - offload[p][j] 
                        offload[p][j] = offload[p][j] + ndiff
                        offload[i][j] = offload[i][j] - ndiff
                        diff = diff -ndiff
                        if (diff <= 0):
                            break
'''            
X = DN[5][0:1000]
y = targets[0][0:1000]                                                
model3 = LogisticRegression(solver='liblinear', random_state=0)
model3.fit(X, y) 
print(model3.score(X, y))    
'''  
acc1 = nm.empty(9)
acc2 = nm.empty(9)
for i in range(9):
    min1 = 2
    min2 = 2
    for j in range(k):
        if pm[i][j] < min1:
            min1 = pm[i][j] 
        if pmN[i][j] < min2:
            min2 = pmN[i][j]
    acc1[i] = min1
    acc2[i] = min2
    
maxload = nm.empty(9)
maxoffload = nm.empty(9)
X = [1,2,3,4,5,6,7,8,9]
X_axis = nm.arange(len(X))
for i in range(9):
    maxl = 0
    maxoffl = 0
    for j in range(k):
        if load[i][j] > maxl:
            maxl = load[i][j]
        if offload[i][j] > maxoffl:
            maxoffl = offload[i][j]
    maxload[i] = maxl 
    maxoffload[i] = maxoffl  
    
barWidth = 0.1
fig = plt.subplots(figsize =(12, 8)) 

 
# Set position of bar on X axis 
br1 = nm.arange(len(maxload)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 
 
# Make the plot
plt.bar(br1, maxload, color ='r', width = barWidth, 
        edgecolor ='grey', label ='FCFS')
plt.bar(br2, maxoffload, color ='g', width = barWidth, 
        edgecolor ='grey', label ='DBTM')  
plt.bar(br3, maxload, color ='b', width = barWidth, 
        edgecolor ='grey', label ='LJF') 
plt.axhline(y = 0.9, color = 'y', linestyle = '-',label = 'load threshold (ζ)')
plt.title(str2, fontweight ='bold', fontsize = 15) 
# Adding Xticks 
plt.xlabel('Nodes', fontweight ='bold', fontsize = 15) 
plt.ylabel('Load', fontweight ='bold', fontsize = 15) 
plt.xticks([r + barWidth for r in range(len(maxload))], 
        X) 
plt.legend(loc='best')
            
# set width of bar 
barWidth = 0.1
fig = plt.subplots(figsize =(12, 8)) 

 
# Set position of bar on X axis 
br1 = nm.arange(len(acc1)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 
 
# Make the plot
plt.bar(br1, acc1, color ='r', width = barWidth, 
        edgecolor ='grey', label ='FCFS') 
plt.bar(br2, acc2, color ='g', width = barWidth, 
        edgecolor ='grey', label ='DBTM') 
plt.bar(br3, acc2, color ='b', width = barWidth, 
        edgecolor ='grey', label ='LJF') 

plt.title(str2, fontweight ='bold', fontsize = 15) 
# Adding Xticks 
plt.xlabel('Nodes', fontweight ='bold', fontsize = 15) 
plt.ylabel('Accuracy', fontweight ='bold', fontsize = 15) 
plt.xticks([r + barWidth for r in range(len(acc1))], 
        X)
 
plt.legend(loc='best')
plt.ylim([0,1])
plt.show()   