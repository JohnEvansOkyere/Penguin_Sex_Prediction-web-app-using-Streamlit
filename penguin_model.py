import pandas as pd
from sklearn.ensemble import RandomForestClassifier

penguin = pd.read_csv("penguins_cleaned.csv")
print(penguin)

# make a copy
df = penguin.copy()
target = 'sex'
encode = ['species', 'island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

target_mapper = {'male':1, 'female':0}
def target_encode(val):
    return target_mapper[val]

df['sex'] = df['sex'].apply(target_encode)

#seperate x and y

x = df.drop(['sex'], axis=1)
y = df['sex']

#Build the model

clf = RandomForestClassifier()

#Fit the model
clf.fit(x,y)

#savig the model
import pickle
pickle.dump(clf, open('penguins_clf.pkl','wb' ))