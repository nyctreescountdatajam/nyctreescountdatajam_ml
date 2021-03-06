import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from datetime import datetime

print 'time start:', datetime.now()
t_start = datetime.now()

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

    



data = pd.read_csv('data.csv') #def main():
t0 = datetime.now()
X, index_dict, num_indexes = one_hot(data[data.columns[:-1]])
print index_dict, num_indexes
print "one hot training set time:", (datetime.now() - t0)
print 'count of features from one hot', X.shape

Y = data['label']

# fit here
clf = RandomForestClassifier() #Predict: 0.938370417801
clf = ExtraTreesClassifier() #Predict: 0.610565071234
clf = RandomForestClassifier(n_estimators=100) #Predict: 0.955135798517
clf = RandomForestClassifier(n_estimators=100) #Predict: 0.95509311136, 2nd time run

print datetime.now()

t0 = datetime.now()
clf.fit(X, Y)
print "training time:", (datetime.now() - t0) 
print "training accuracy:", clf.score(X, Y)

quiz = pd.read_csv('quiz.csv')
t0 = datetime.now()
Xtest, _, _ = one_hot(quiz, index_dict, num_indexes)
print "one hot test set time:", (datetime.now() - t0)

t0 = datetime.now()
prediction = clf.predict(Xtest) # do prediction and save it
print "prediction time:", (datetime.now() - t0)

ftime = str(datetime.now()).replace(' ','-').replace(':','-').replace('.','-')
print ftime

with open('output/myoutput-'+ftime+'_POST.csv', 'w') as f:
    f.write('Id,Prediction\n')
    the_id = 1
    for p in prediction:
        f.write("%s,%s\n" % (the_id, p))
        the_id += 1

# print 'cross validating...'
# t0 = datetime.now()
# scores = cross_val_score(clf, X, Y, cv=5)
# print "cross validation time:", (datetime.now() - t0)
# print scores #print prediction.score(X, Y)

importances = clf.feature_importances_
print importances

std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

impFeaturesList = [] #empty list for feature importances
impThresh       = 0.00001

for f in range(X.shape[1]):
    #print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    if importances[indices[f]] > impThresh: #was 0.00000001, changed to 0.000001
        impFeaturesList.append(indices[f])

print 'count of features in important features list that are above',impThresh,'is', len(impFeaturesList)

print "Entire run-time:", (datetime.now() - t_start) #if __name__ == '__main__': #main()
