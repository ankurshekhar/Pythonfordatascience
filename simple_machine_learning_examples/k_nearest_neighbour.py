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

#Import metrics model to check the accuracy
from sklearn import metrics     

#Initiate a new variable of kNearestNeighbour class
clf=KNeighborsClassifier()
clf.fit(BodyMeasurements,Gender)
prediction=clf.predict([[210,70,550]])
print (prediction)
