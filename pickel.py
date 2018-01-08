
import pandas
import numpy
import matplotlib.pyplot as plt
import pickle


df=pandas.read_csv('Preprocessed/data_combined.csv')
print df[:15]

print df.dtypes

print df['CATEGORY'].value_counts()

print df['THRESTBPS'].value_counts().head()

#average rest blood pressure is  generally in range 120-140
df['THRESTBPS'] = df['THRESTBPS'].replace(['?'],'120')
df['THRESTBPS'] = df['THRESTBPS'].astype('int64')

#print df.columns
print df['FBS'].value_counts()
print "male:\n",df[df['SEX']==1]['FBS'].value_counts()
print "Female:\n",df[df['SEX']==0]['FBS'].value_counts()#directly replace with 0

#randomly filling values with 80% with 0 and 20% with 1s
v=df.FBS.values=='?'
df.loc[v, 'FBS'] = numpy.random.choice(('0','1'), v.sum(), p=(0.8,0.2))
print df['FBS'].value_counts()
df['FBS']=df['FBS'].astype('int64')

df['CHOL'].value_counts().head()
#evenly distributed...
#so will replace with mean of the class

df['CHOL']=df['CHOL'].replace('?','-69')#temporarily replacing ? with -69
df['CHOL']=df['CHOL'].astype('int64')
k=int(df[df['CHOL']!=-69]['CHOL'].mean())
df['CHOL']=df['CHOL'].replace(-69,k)


print df['CHOL'].unique() #completed !--!

print df['RESTECG'].value_counts()

#replacing with max occuring value for attribute
df['RESTECG']=df['RESTECG'].replace('?','0')
#print df['RESTECG'].unique()
#print df['RESTECG'].value_counts()
df['RESTECG'] = df['RESTECG'].astype('int64')

df['THALACH'].value_counts().head()

df['THALACH']=df['THALACH'].replace('?','-69')#temporarily replacing ? with -69
df['THALACH']=df['THALACH'].astype('int64')
k=int(df[df['THALACH']!=-69]['THALACH'].mean())
print k
df['THALACH']=df['THALACH'].replace(-69,k)

df['THALACH'].value_counts().head()

#exang:exercise induced angina (1 = yes; 0 = no) 
print df['EXANG'].value_counts()

k=528.0/(337.0+528.0)
print k

v=df.EXANG.values=='?'
df.loc[v,'EXANG'] = numpy.random.choice(('0','1'), v.sum(), p=(0.61,0.39))
print df['EXANG'].value_counts()
df['EXANG']=df["EXANG"].astype('int64')

print df['OLDPEAK'].value_counts().head()

df['OLDPEAK']=df['OLDPEAK'].replace('?','-69')#temporarily replacing ? with -69
df['OLDPEAK']=df['OLDPEAK'].astype('float64')
k=df[df['OLDPEAK']!=-69]['OLDPEAK'].mean()
print k
df['OLDPEAK']=df['OLDPEAK'].replace(-69,numpy.round(k,1))

print df['OLDPEAK'].value_counts()

print df['SLOPE'].value_counts()

#k=203.0/(345.0+203.0+63.0)
#print k

v=df.SLOPE.values=='?'
df.loc[v,'SLOPE'] = numpy.random.choice(('2','1','3'), v.sum(), p=(0.6,0.30,0.10))
print df['SLOPE'].value_counts()
df['SLOPE']=df['SLOPE'].astype('int64')

print df["CA"].value_counts()
k=(41.0)/(181+67+41+20)
print k

v=df.CA.values=='?'
df.loc[v,'CA'] = numpy.random.choice(('0','1','2','3'), v.sum(), p=(0.60,0.20,0.13,0.07))
df['CA']=df['CA'].astype('int64')
print df['CA'].value_counts()

print df['THAL'].value_counts()
#can't use random walk directly here

print df[df['THAL']=='3']['SEX'].value_counts()
print df[df['THAL']=='7']['SEX'].value_counts()

print "THAL:3=====>\n",df[df['THAL']=='3']['CATEGORY'].value_counts()
print "THAL:7=====>\n",df[df['THAL']=='7']['CATEGORY'].value_counts()
print "THAL:6=====>\n",df[df['THAL']=='6']['CATEGORY'].value_counts()

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
print df['THAL'].value_counts()
df['THAL']=df['THAL'].astype('int64')

print df.dtypes

dummies = pandas.get_dummies(df["CP"],prefix="CP")
df = df.join(dummies)

dummies = pandas.get_dummies(df["RESTECG"],prefix="RESTECG")
df      = df.join(dummies)

dummies = pandas.get_dummies(df["SLOPE"],prefix="SLOPE")
df      = df.join(dummies)

dummies = pandas.get_dummies(df["THAL"],prefix="THAL")
df      = df.join(dummies)


del df['CP']
del df['RESTECG']
del df['SLOPE']
del df['THAL']

print df.dtypes

for g in df.columns:
    if df[g].dtype=='uint8':
        df[g]=df[g].astype('int64')

df.dtypes
df.loc[df['CATEGORY']>0,'CATEGORY']=1

stdcols = ["AGE","THRESTBPS","CHOL","THALACH","OLDPEAK"]
nrmcols = ["CA"]
stddf   = df.copy()
stddf[stdcols] = stddf[stdcols].apply(lambda x: (x-x.mean())/x.std())
stddf[nrmcols] = stddf[nrmcols].apply(lambda x: (x-x.mean())/(x.max()-x.min()))

print stddf.dtypes

from sklearn.model_selection import train_test_split


df_copy=stddf.copy()
df_copy=df_copy.drop(['CATEGORY'],axis=1)

dat=df_copy.values
#print dat.shape

print type(dat),dat

labels=df['CATEGORY'].values
print labels,type(labels)

x_train,x_test,y_train,y_test=train_test_split(dat,labels, test_size=0.25, random_state=42)

print "x_train:",x_train.shape
print "y_train:",y_train.shape
print
print "x_test:",x_test.shape
print "y_test:",y_test.shape

#training and testing
#SVM
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=5)
clf.fit(x_train,y_train)
print "SVM:",clf.score(x_test,y_test)*100,"%"


from sklearn import linear_model
lrcv=linear_model.LogisticRegressionCV(fit_intercept=True,penalty='l2',dual=False)
lrcv.fit(x_train,y_train)
print "Logistic Regression:",lrcv.score(x_test,y_test)*100,"%"
f=open("svm_check", 'wb')
pickle.dump(clf,f)
f.close()
