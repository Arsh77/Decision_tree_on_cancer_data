from sklearn import tree
import pandas as pd
import numpy as np
from sklearn import cross_validation
import pickle
df= pd.read_csv('breast-cancer-wisconsin.data')
df.drop(['id'] ,1, inplace =True)
df.replace('?', -99999 , inplace =True)
#print(df)
X=df.drop(['class'] , 1)
y=df['class']

# classifier is pickled 
'''
X_train, X_test , y_train ,y_test = cross_validation.train_test_split(X ,y ,test_size=0.2)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
'''
accuracy = clf.score(X_test , y_test)

with open('my_pic' , 'wb') as d:
    pickle.dump(clf , d)
    
pickle_in =open('my_pic' , 'rb')
clf =pickle.load(pickle_in)    
print(accuracy)
Z=[]
for i in range(9):
    a=input()
    Z.append(a)
print(Z)    
t=np.array(Z)
print(t)
predict_true = clf.predict(t.reshape(1,-1))
print(predict_true)    