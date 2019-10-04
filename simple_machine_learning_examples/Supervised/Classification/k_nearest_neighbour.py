#Local dataset
#List of lists of height, weight, shoe size
BodyMeasurements=[[110,30,350], [140,20,490],[120,80,320],[140,80,400],[120,50,400],[141,50,470],[220,100,550],[210,60,250],[310,60,250],[210,70,550]]
#Gender list contains the labels
Gender=[['Male'],['Female'],['Male'],['Female'],['Male'],['Female'],['Male'],['Female'],['Male'],['Female']]

#Split the data
#Use train_test_split function to split the data
#test_size determines the split percentage, here it's a 80:20 Split
#random state parameter makes the data split the same way everytime. Need to do more research on this.

from sklearn.model_selection import train_test_split
BodyMeasurements_train,BodyMeasurements_test,Gender_train,Gender_test=train_test_split(BodyMeasurements,Gender,test_size=0.2,random_state=4)

#Check the shape of the train and test objects
print (BodyMeasurements_train)
print (BodyMeasurements_test)
print (Gender_train)
print (Gender_test)

#from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

#Import metrics model to check the accuracy. Not sure about how and why of this, need to come back later
from sklearn import metrics   
import numpy as np  
#Try running from k=1 through 9 and record testing accuracy
k_range=range(1,9)
scores={}
scores_list=[]

for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(BodyMeasurements_train,np.ravel(Gender_train))
    prediction=knn.predict(BodyMeasurements_test)
    #print (prediction)
    scores[k]=metrics.accuracy_score(Gender_test,prediction)
    scores_list.append(metrics.accuracy_score(Gender_test,prediction))

print (scores_list)

'exec (%matplotlib inline)'
import matplotlib.pyplot as plt

#plot the relationship between K and the testing accuracy
plt.plot(k_range,scores_list)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show(block=False)
input('press <ENTER> to continue')

#Here k was selected as 7 as accuracy according to the graph generated was 100% for k=7. Need to try with large datasets.
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(BodyMeasurements,Gender)

Gender_Predict=knn.predict([[210,70,550]])
print (Gender_Predict)