
import time
import math
import numpy as nm
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from random import seed
from random import randint
from statistics import NormalDist
import statistics
from statsmodels.stats.weightstats import ztest as ztest
import scipy.stats
from numpy import sqrt, abs, round
from scipy.stats import norm
from scipy.stats import ks_2samp
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import random
import skfuzzy as fuzz
import skfuzzy.membership as mf
import matplotlib.pyplot as plt
from statistics import variance
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

random.seed(1)
start = time.time()

def twoSampZ(X1, X2, mudiff, sd1, sd2, n1, n2):
    x = (sd1**2/n1) + (sd2**2/n2)
    pooledSE = sqrt(x)
    z = ((X1 - X2) - mudiff)/pooledSE
    pval = 2*(norm.sf(abs(z)))
    return round(pval, 4)

nm.random.seed(0)
'parameters'


n = 6000 #arithmos data vectors
d = 4 #diastaseis mazi me y
k = 12 #periods
'datasets'


def initialize(n, d):  
    nm.random.seed(0)
    DS = list()
    load = nm.empty((10,k))
    queries = nm.empty((10,k))
    offload = nm.empty((10,k))
    #for i in range(10):
        #load[i] = random.uniform(0,1)

    load[0][0] =  0.65 
    load[1][0]  =  0.15
    load[2][0]  =  0.70
    load[3][0]  =  0.54
    load[4][0]  =  0.50
    load[5][0]  =  0.27
    load[6][0]  =  0.24
    load[7][0]  =  0.43
    load[8][0]  =  0.66
    load[9][0]  =  0.20
    
    offload[0][0] =  0.65 
    offload[1][0]  =  0.15
    offload[2][0]  =  0.70
    offload[3][0]  =  0.54
    offload[4][0]  =  0.50
    offload[5][0]  =  0.27
    offload[6][0]  =  0.24
    offload[7][0]  =  0.43
    offload[8][0]  =  0.66
    offload[9][0]  =  0.20
    
    queries[0][0]  =  0.45 
    queries[1][0]  =  0.05
    queries[2][0]  =  0.70
    queries[3][0]  =  0.54
    queries[4][0]  =  0.60
    queries[5][0]  =  0.17
    queries[6][0]  =  0.04
    queries[7][0]  =  0.53
    queries[8][0]  =  0.56
    queries[9][0]  =  0.10
    
    for i in range(10):
        y = nm.empty(n)
        dr = list()
        a = 1
        b = 2
        c = 3
        for j in range(d):
            if(i % 3 == 0):
                nm.random.seed(0)
                x = nm.random.normal(a,  0.1, n)
                a = a + 1
            elif(i % 3 == 1):
                nm.random.seed(0)
                x = nm.random.normal(b,  0.1, n)
                b = b + 1
            else:
                nm.random.seed(0)
                x = nm.random.normal(c,  0.1, n)
                c = c + 1

            dr.append(x)
        if(i % 3 == 0):
            for i in range(n):
                if dr[0][i] > 1 :
                    y[i] = 1
                else:
                    y[i] = 0
                
        elif(i % 3 == 1):
            for i in range(n):
                if dr[0][i] > 2:
                    y[i] = 1
                else:
                    y[i] = 0
        else:
            for i in range(n):
                if dr[0][i] > 3:
                    y[i] = 1
                else:
                    y[i] = 0
        dr.append(y)
        DS.append(dr)
        
       
    
    
    return DS,load,queries,offload

    
    

DS,load,queries,offload = initialize(n, d)
for i in range(10):
    for j in range(1,k):
        load[i][j] = load[i][j-1] 
        queries[i][j] = queries[i][j-1]
        offload[i][j] = offload[i][j-1] 


'drift'

for l in range(d):
    #temp = DS[3][l][4000:5999] 
    #temp = nm.empty(2000)
    for a in range(4000,5999):
        temp = DS[3][l][a]
        DS[3][l][a] = DS[4][l][a]
        DS[4][l][a] = temp

for l in range(d):
    #temp = DS[5][l][4500:5999] 
    #temp = nm.empty(1500)
    for a in range(4500,5999):
        temp = DS[5][l][a]
        DS[5][l][a] = DS[7][l][a]
        DS[7][l][a] = temp
        
for l in range(d):
    #temp = DS[0][l][5000:5999] 
    #temp = nm.empty(1000)
    for a in range(5000,5999):
        temp = DS[2][l][a]
        DS[0][l][a] = DS[2][l][a]
        DS[2][l][a] = temp
    


'Train the models in the datasets for the first three periods'

pm = nm.empty((10,k))
pmN =  nm.empty((10,k)) 
md = list()
lm = nm.empty(10)
md2 = list()
lm2 = nm.empty(10)
EMWA = nm.empty((10,k))
Var = nm.empty((10,k))
ucl = nm.empty((10,k))
lcl = nm.empty((10,k))
for i in range(10):
    X = nm.empty((1500,d))
    y = nm.empty(1500)
    for a in range(1500):
        for h in range(d):
            X[a][h] = DS[i][h][a]
        y[a] = DS[i][d][a] 
                
        
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
    '''regr = linear_model.LinearRegression()
    regr.fit(X, y) 
    md[i] = regr.coef_
    l[i] = regr.intercept_'''
    
for i in range(10):
    EMWA[i][0] = pm[i][0]
    EMWA[i][1] = pm[i][1]
    EMWA[i][2] = pm[i][2]
    Var[i][0] = 0
    Var[i][1] = 0
    Var[i][2] = 0.01
    
m = nm.empty((10,k,d))
v = nm.empty((10,k,d))   
for j in range(k):
    for i in range(10):
        for l in range(d):
            m[i][j][l] = DS[i][l][j*500:(j+1)*500].mean()
            v[i][j][l] = variance(DS[i][l][j*500:(j+1)*500])
        
W = 3

mbase = nm.empty((10,d))
vbase = nm.empty((10,d))
for i in range(10):
    for l in range(d):
        sum1  = 0
        sum2  = 0
        for j in range(W):
            sum1 = sum1 + m[i][j][l]
            sum2 = sum2 + v[i][j][l]
        mbase[i][l] = sum1/W
        vbase[i][l] = sum2/W
        

#THRESHOLDS!!!
load1 = 0.9        
Dm = 3
Dv = 0.5
thr1 = 0.9 #α
la = 0.7 #λ
de = 2 #δ
beta = 0.5
gamma = 0.5

deik=0

for j in range(W,k):
    "DATA DRIFT"           
    mnew = nm.empty((10,d))
    vnew = nm.empty((10,d))
    for i in range(10):
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
            print(" Data Drift detected for node:", i ,"in period:", j )
            mbase[i] = mnew[i] 
            vbase[i]  = vnew[i] 
        "PERFORMANCE DRIFT"
        
        X = nm.empty((500,d))
        y = nm.empty(500)
        for a in range(500):
            for h in range(d):
                X[a][h] = DS[i][h][j*500 + a]
            y[a] = DS[i][d][j*500+ a] 
        #else:
            #for a in range(250):
                #for h in range(d):
                   #X[a][h] = DN[i][h][(j-k)*250 + a]
                #y[a] = DN[i][d][(j-k)*250 + a] 
            
        
        
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
        if (pd):
            print("Performance Drift detected for node:",i,"in period:",j+1)
        if ( pd or dd): #or dd
            deik = deik +1
            load[i][j] = load[i][j] + 0.50
            offload[i][j] = offload[i][j] + 0.50
        if (offload[i][j]>=0.9):
            sim = nm.empty(10)
            
            for p in range(10):
                if (p != i):
                    sum2 = 0
                    for l in range(d):
                        #print(ks_2samp(DS[i][l], DS[j][l])[1])
                        if(ks_2samp(DS[i][l][(j-1)*500:j*500], DS[p][l][(j-1)*500:j*500])[1] < beta):
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
        if (pd or dd):
            X = nm.empty((500,d))
            y = nm.empty(500)
            for a in range(500):
                for h in range(d):
                    X[a][h] = DS[i][h][j*500+a]
                y[a] = DS[i][d][j*500+a] 
                        
                
            model2 = LogisticRegression(solver='liblinear', random_state=0)
            model2.fit(X, y)
            print( "SCORE ", i,j,model2.score(X, y))
            md2[i] = model2.coef_
            lm2[i] = model2.intercept_
            pmN[i][j] = model2.score(X, y)
            #print(i,j,pmN[i][j])
        else:
            X = nm.empty((500,d))
            y = nm.empty(500)
            for a in range(500):
                for h in range(d):
                    X[a][h] = DS[i][h][j*500+a]
                y[a] = DS[i][d][j*500+a] 
            indicator = 0
            if (md2[i][0][0] != md[i][0][0]):
                    indicator = 1
            if (indicator):
                model.intercept_ = lm2[i]
                model.coef_ = md2[i]
                model.predict_proba(X)
                ypred = model.predict(X)
                pmN[i][j] = model.score(X,y)
            else:
                pmN[i][j] = pm[i][j] 
                
            
import matplotlib.pyplot as plt
maxload = nm.empty(10)
maxoffload = nm.empty(10)
#offload[7][4] = 0
X = [1,2,3,4,5,6,7,8,9,10]
X_axis = nm.arange(len(X))
for i in range(10):
    maxl = 0
    maxoffl = 0
    for j in range(k):
        if load[i][j] > maxl:
            maxl = load[i][j]
        if offload[i][j] > maxoffl:
            maxoffl = offload[i][j]
    maxload[i] = maxl 
    maxoffload[i] = maxoffl      
#plt.bar(X_axis - 0.1, nm.arange(len(maxload)), maxload)
#plt.bar(X_axis + 0.1,nm.arange(len(maxoffload)), maxoffload)
'''
plt.bar(X_axis-0.3,maxload,0.2, label = 'FCFS')
plt.bar(X_axis+0.3,maxoffload,0.2, label = 'TM')
plt.axhline(y = 0.9, color = 'r', linestyle = '-')
plt.xlabel("Nodes")
plt.ylabel("Load")
plt.xticks(X)
plt.legend()
plt.show() 
'''
# set width of bar 
barWidth = 0.15
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
plt.axhline(y = 0.9, color = 'y', linestyle = '-',label = 'load threshold (ζ)')
plt.bar(br3, maxload, color ='b', width = barWidth, 
        edgecolor ='grey', label ='LJF') 

plt.title("Case 1", fontweight ='bold', fontsize = 15)  
# Adding Xticks 
plt.xlabel('Nodes', fontweight ='bold', fontsize = 15) 
plt.ylabel('Load', fontweight ='bold', fontsize = 15) 
plt.xticks([r + barWidth for r in range(len(maxload))], 
        X)
 
plt.legend()
plt.show()               
#print('Retraining', deik)

acc1 = nm.empty(10)
acc2 = nm.empty(10)
for i in range(10):
    min1 = 2
    min2 = 2
    for j in range(k):
        if pm[i][j] < min1:
            min1 = pm[i][j] 
        if pmN[i][j] < min2:
            min2 = pmN[i][j]
    acc1[i] = min1
    acc2[i] = min2

# set width of bar 
barWidth = 0.15
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

plt.title("Case 1", fontweight ='bold', fontsize = 15) 
# Adding Xticks 
plt.xlabel('Nodes', fontweight ='bold', fontsize = 15) 
plt.ylabel('Accuracy', fontweight ='bold', fontsize = 15) 
plt.xticks([r + barWidth for r in range(len(acc1))], 
        X)
 
plt.legend()
plt.show()                                
