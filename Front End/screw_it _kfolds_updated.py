import pandas
import numpy
#import matplotlib.pyplot as plt


# In[284]:


df=pandas.read_csv('data_combined.csv')
#print df[:5]


df['THRESTBPS'] = df['THRESTBPS'].replace(['?'],'120')
df['THRESTBPS'] = df['THRESTBPS'].astype('int64')



#randomly filling values with 80% with 0 and 20% with 1s
v=df.FBS.values=='?'
df.loc[v, 'FBS'] = numpy.random.choice(('0','1'), v.sum(), p=(0.8,0.2))
#print df['FBS'].value_counts()
df['FBS']=df['FBS'].astype('int64')


# # Replacing missing values in CHOL

# In[291]:


df['CHOL'].value_counts().head()
#evenly distributed...
#so will replace with mean of the class


# In[292]:


df['CHOL']=df['CHOL'].replace('?','-69')#temporarily replacing ? with -69
df['CHOL']=df['CHOL'].astype('int64')
k=int(df[df['CHOL']!=-69]['CHOL'].mean())
df['CHOL']=df['CHOL'].replace(-69,k)


#print df['CHOL'].unique() #completed !--!


# ## Replacing missing values in RESTECG

# In[293]:


#print df['RESTECG'].value_counts()

#replacing with max occuring value for attribute
df['RESTECG']=df['RESTECG'].replace('?','0')
#print df['RESTECG'].unique()
#print df['RESTECG'].value_counts()
df['RESTECG'] = df['RESTECG'].astype('int64')



#print "after replacing\n",df['RESTECG'].value_counts()


# ## Replacing missing values in THALACH

# In[294]:


df['THALACH'].value_counts().head()


# In[295]:


df['THALACH']=df['THALACH'].replace('?','-69')#temporarily replacing ? with -69
df['THALACH']=df['THALACH'].astype('int64')
k=int(df[df['THALACH']!=-69]['THALACH'].mean())
#print k
df['THALACH']=df['THALACH'].replace(-69,k)


# In[296]:


df['THALACH'].value_counts().head()


# ## Replacing missing values in EXANG

# In[297]:


#exang:exercise induced angina (1 = yes; 0 = no)
#print df['EXANG'].value_counts()


# In[298]:


k=528.0/(337.0+528.0)

v=df.EXANG.values=='?'
df.loc[v,'EXANG'] = numpy.random.choice(('0','1'), v.sum(), p=(0.61,0.39))
#print df['EXANG'].value_counts()
df['EXANG']=df["EXANG"].astype('int64')


df['OLDPEAK']=df['OLDPEAK'].replace('?','-69')#temporarily replacing ? with -69
df['OLDPEAK']=df['OLDPEAK'].astype('float64')
k=df[df['OLDPEAK']!=-69]['OLDPEAK'].mean()
#print k
df['OLDPEAK']=df['OLDPEAK'].replace(-69,numpy.round(k,1))



k=203.0/(345.0+203.0+63.0)

v=df.SLOPE.values=='?'
df.loc[v,'SLOPE'] = numpy.random.choice(('2','1','3'), v.sum(), p=(0.6,0.30,0.10))
df['SLOPE']=df['SLOPE'].astype('int64')


k=(41.0)/(181+67+41+20)


v=df.CA.values=='?'
df.loc[v,'CA'] = numpy.random.choice(('0','1','2','3'), v.sum(), p=(0.60,0.20,0.13,0.07))
df['CA']=df['CA'].astype('int64')

df['THAL']=df['THAL'].replace('?',-1)
'''
df['THAL']=df['THAL'].replace('?',-1)
for row in df.iterrows():
    if row['THAL']==-1 and row['CATEGORY']>=1:
        df.loc[row.Index, 'ifor'] = 7
        
    elif row['THAL']==-1 and row['CATEGORY']==0:
        df.loc[row.Index, 'ifor'] = 3
'''
df.loc[(df['THAL']==-1)&(df['CATEGORY']!=0),'THAL']='7'
#print df['THAL'].value_counts()
df.loc[(df['THAL']==-1)&(df['CATEGORY']==0),'THAL']='3'
#print df['THAL'].value_counts()
df['THAL']=df['THAL'].astype('int64')


# In[312]:


#print df.dtypes


# In[313]:


dummies = pandas.get_dummies(df["CP"],prefix="CP")
df = df.join(dummies)

dummies = pandas.get_dummies(df["RESTECG"],prefix="RESTECG")
df      = df.join(dummies)

dummies = pandas.get_dummies(df["SLOPE"],prefix="SLOPE")
df      = df.join(dummies)

dummies = pandas.get_dummies(df["THAL"],prefix="THAL")
df      = df.join(dummies)

#dummies = pandas.get_dummies(df["EXANG"],prefix="EXANG")
#df = df.join(dummies)

#del df['SEX']
del df['CP']
del df['RESTECG']
del df['SLOPE']
del df['THAL']
#del df['EXANG']


# In[314]:


#print df.dtypes


# In[315]:


for g in df.columns:
    if df[g].dtype=='uint8':
        df[g]=df[g].astype('int64')


# In[316]:


df.dtypes
df.loc[df['CATEGORY']>0,'CATEGORY']=1


# In[317]:


stdcols = ["AGE","THRESTBPS","CHOL","THALACH","OLDPEAK"]
nrmcols = ["CA"]


import pickle
stddf   = df.copy()


f=open("stdcols", 'wb')
pickle.dump(stdcols,f)
f.close()

f=open("nrmcols", 'wb')
pickle.dump(nrmcols,f)
f.close()




normal={}
for q in nrmcols:
    normal[q+"_mean"]=stddf[q].mean()
    normal[q+"_min"]=min(stddf[q])
    normal[q+"_max"]=max(stddf[q])
print(normal)
f=open("normal", 'wb')
pickle.dump(normal,f)
f.close()

standardize={}
for q in stdcols:
    standardize[q+"_mean"]=stddf[q].mean()
    standardize[q+"_std"]=stddf[q].std()


print(standardize)
f=open("standardize", 'wb')
pickle.dump(standardize,f)
f.close()


stddf[stdcols] = stddf[stdcols].apply(lambda x: (x-x.mean())/x.std())
stddf[nrmcols] = stddf[nrmcols].apply(lambda x: (x-x.mean())/(x.max()-x.min()))




from sklearn.model_selection import train_test_split



df_copy=stddf.copy()
df_copy=df_copy.drop(['CATEGORY'],axis=1)

dat=df_copy.values


labels=df['CATEGORY'].values

x_train,x_test,y_train,y_test=train_test_split(dat,labels, test_size=0.25, random_state=42)



from sklearn import svm
clf = svm.SVC(gamma=0.001, C=5)
clf.fit(x_train,y_train)

f=open("svm_main", 'wb')
pickle.dump(clf,f)
f.close()





clfprob = svm.SVC(gamma=0.001,kernel='linear',C=100,probability=True)
clfprob.fit(x_train,y_train)
print("actual output:",labels[2])
print("predicted",clf.predict([dat[2]]))

f=open("svm_prob", 'wb')
pickle.dump(clfprob,f)
f.close()

print(clf.score(x_test,y_test))
print(clfprob.predict_proba([dat[2]]))

 