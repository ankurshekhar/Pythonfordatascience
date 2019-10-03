from sklearn import tree

#List of height, weight, shoe size
BodyMeasurements=[[110,30,350], [140,20,490],[120,80,320],[140,80,400],[120,50,400],[141,50,470],[220,100,550],
[210,60,250],[310,60,250],[210,70,550]]
#Gender list contains the labels
Gender=[['Male'],['Female'],['Male'],['Female'],['Male'],['Female'],['Male'],['Female'],['Male'],['Female']]

#Now initiate a variable of tree.classifier class
clf=tree.DecisionTreeClassifier()

clf=clf.fit(BodyMeasurements,Gender)
prediction=clf.predict([[110,32,78]])
print (prediction)