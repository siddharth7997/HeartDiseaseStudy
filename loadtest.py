import pandas
import numpy
import matplotlib.pyplot as plt
import pickle

k=pickle.load(open('svm_check','rb'))
m=pickle.load(open('value','rb'))

print m
print type(m)
kr=[]
kr.append(m)
print type(kr)
print m.shape	
print "prediction:",k.predict(kr)


