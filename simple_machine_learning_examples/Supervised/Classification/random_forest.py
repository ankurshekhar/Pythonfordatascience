#Local dataset
#List of lists of height, weight, shoe size
BodyMeasurements=[[110,30,350], [140,20,490],[120,80,320],[140,80,400],[120,50,400],[141,50,470],[220,100,550],[210,60,250],[310,60,250],[210,70,550]]
#Gender list contains the labels
Gender=[['Male'],['Female'],['Male'],['Female'],['Male'],['Female'],['Male'],['Female'],['Male'],['Female']]


from sklearn.ensemble import RandomForestClassifier
#Not sure what is n_estimators here. And why is it selected as 2.
rf=RandomForestClassifier(n_estimators=2)
rf=rf.fit(BodyMeasurements,Gender)
prediction = rf.predict([[210,70,550]])
print (prediction)


