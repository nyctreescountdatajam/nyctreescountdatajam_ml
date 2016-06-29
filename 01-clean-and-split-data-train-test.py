import pandas as pd
import numpy  as np

inCSV = '/Users/danielmsheehan/Downloads/nyctrees2015.csv'

df = pd.read_csv(inCSV)

dropList = ['spc_latin','the_geom','tree_id','created_at','block_id','boroname','st_assem','st_senate','state','cb_num','zip_city','address','latitude','longitude','x_sp','y_sp','uid','nta']
#rem from dropList 'boro_ct',

df = df.drop(dropList, axis=1)

df['label'] = np.where(df['health']=='Poor', 1, 0)

print df.head(10)

msk = np.random.rand(len(df)) < 0.8 #split 80/20, 80% for training, 20% for testing
#http://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas
train = df[msk]
test = df[~msk]

print len(df)
print len(train)
print len(test)

test.to_csv('quiz_w_label.csv', index=False)
test = test.drop(['health','label'],axis=1)

train.to_csv('data.csv',index=False)
test.to_csv('quiz.csv', index=False)

dfx=df[['health']]
dfx['count'] = 1
dfg = dfx.groupby('health').sum()
print dfg.head(10)

dfx=df[['label']]
dfx['count'] = 1
dfg = dfx.groupby('label').sum()
print dfg.head(10)
