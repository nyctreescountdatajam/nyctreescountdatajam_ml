import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier #try xg boost
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
from sklearn import metrics

print datetime.now()

def one_hot(D, index_dict=None, num_indexes=-1):
    if index_dict is None: # dictionary D.column -> value -> index in output
        idx = 0
        index_dict = {}
        for c in D.columns:
            index_dict[c] = {}
            s = set(D[c])
            if len(s) > 2:
                for col in s:
                    index_dict[c][col] = idx
                    idx += 1
            elif len(s) == 2:
                s = list(s)
                index_dict[c][s[0]] = idx
                index_dict[c][s[1]] = idx + 1
                idx += 2
            else:
                pass # do nothing, this column only has 1 value, can't be used for prediction
        num_indexes = idx

    X = np.zeros((len(D), num_indexes))
    for i in xrange(len(D)):
        row = D.iloc[i]
        for c in D.columns:
            val = row[c]
            if val in index_dict[c]:
                j = index_dict[c][val]
                X[i,j] = 1
    return X, index_dict, num_indexes


#def main():
#----indent here
data = pd.read_csv('data.csv')
t0 = datetime.now()
X, index_dict, num_indexes = one_hot(data[data.columns[:-1]])
print "one hot training set time:", (datetime.now() - t0)
print 'count of features from one hot', X.shape

Y = data['label']

clf = ExtraTreesClassifier()#n_estimators=400, n_jobs=6)# fit here to run to get impFeaturesLIst
#clf = RandomForestClassifier(n_estimators=150, n_jobs=4)

print datetime.now()

t0 = datetime.now()
clf.fit(X, Y)
print "training time:", (datetime.now() - t0)
print "training accuracy:", clf.score(X, Y)

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

impFeaturesList = [] #empty list for feature importances
impThresh       = 0.00001
# higFeaturesList = []
# higThresh       = 0.001
for f in range(X.shape[1]):
    #print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    if importances[indices[f]] > impThresh: #was 0.00000001, changed to 0.000001
        impFeaturesList.append(indices[f])
# for f in range(X.shape[1]):
#     if importances[indices[f]] > higThresh:
#         higFeaturesList.append(indices[f])
print 'count of features in important features list that are above',impThresh,'is', len(impFeaturesList)

X2 = X[:,impFeaturesList]
#what is X2 type? A: an array numpy
# for i in higFeaturesList:
#     df[i] = 
#     y = np.append(a,z,axis=1) #append array column
print X.shape, X2.shape

clf2 = ExtraTreesClassifier(n_estimators=400, n_jobs=6) #n_jobs=1, max_depth=6, fit here
#clf = RandomForestClassifier(n_estimators=150, n_jobs=4)

print datetime.now()

t0 = datetime.now()
clf2.fit(X2, Y)
print "training time:", (datetime.now() - t0)
print "training accuracy:", clf2.score(X2, Y)                   

quiz = pd.read_csv('quiz.csv')
t0 = datetime.now()
Xtest, _, _ = one_hot(quiz, index_dict, num_indexes)
print "one hot test set time:", (datetime.now() - t0)

Xtest2 = Xtest[:,impFeaturesList] #limit text to important features list

prediction = clf2.predict(Xtest2) #do prediction and save 

ftime = str(datetime.now()).replace(' ','-').replace(':','-').replace('.','-')
print ftime

with open('output/myoutput-'+ftime+'.csv', 'w') as f:
    f.write('Id,Prediction\n')
    the_id = 1
    for p in prediction:
        f.write("%s,%s\n" % (the_id, p))
        the_id += 1

# t0 = datetime.now()
# print 'cross validating...'
# scores = cross_val_score(clf, X, Y, cv=5)
# print scores
# print "cross val score time:", (datetime.now() - t0)

# t0 = datetime.now()
# print prediction.score(X, Y)
# print "prediction score time:", (datetime.now() - t0)

#print(clf.feature_importances_)

#----end indent here
# if __name__ == '__main__':
#     main()
