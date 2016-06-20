import pandas as pd
import numpy  as np

inCSV = '/Users/danielmsheehan/Dropbox/Projects/treescount/data/processing/to_cartodb/nyctrees2015.csv'

df = pd.read_csv(inCSV)

dropList = ['the_geom','tree_id','created_at','block_id','boroname','st_assem','st_senate','state','latitude','longitude','x_sp','y_sp','uid']

df = df.drop(dropList, axis=1)

df['label'] = np.where(df['health']=='Poor', 1, 0)

print df.head(10)

#split 80/20, 80% for training, 20% for testing

msk = np.random.rand(len(df)) < 0.8 
#http://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas
train = df[msk]
test = df[~msk]

print len(df)
print len(train)
print len(test)

train.to_csv('data.csv')
test.to_csv('quiz.csv')
