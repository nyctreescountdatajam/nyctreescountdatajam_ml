import pandas as pd

df = pd.read_csv('quiz_w_label.csv')

df['Id'] = df.index + 1

print df.head(10)
print len(df.index)

dfx=df[['label']]
dfx['count'] = 1
dfg = dfx.groupby('label').sum()
print dfg.head(10)

#dfp = pd.read_csv('output/myoutput-2016-06-20-20-42-10-243406_randfore_9384.csv')
#dfp = pd.read_csv('output/myoutput-2016-06-20-21-07-17-582827.csv')
dfp = pd.read_csv('output/myoutput-2016-06-20-21-41-20-469816.csv')

print dfp.head(10)
df = df.merge(dfp, on='Id', how='left')

df = df[['Id','label','Prediction']]

print df.head(10)

dfCorrect = df[(df.label == df.Prediction)]

print dfCorrect.tail(10)

dfPoor = df[(df.label == 1)]

print 'Machine Learning Classifier'
print 'For 2015 Street Tree Census at Prediciting health == "Poor":', len(dfPoor.index)
print len(dfCorrect), 'were predicted correctly out of', len(df.index)
print 'predicts at:', len(dfCorrect.index)/(len(df.index)*1.0)
