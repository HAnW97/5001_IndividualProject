import datetime

import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import dataprocessing

# training data path
TRAIN_DATA_PATH = "train.csv"
TEST_DATA_PATH = "test.csv"
# preprocess both train and test data
data_train = pd.read_csv(TRAIN_DATA_PATH, sep=',')
data_test = pd.read_csv(TEST_DATA_PATH, sep=',')

data_train = dataprocessing.preprocessing_prediction(data_train)
data_test = dataprocessing.preprocessing_prediction(data_test)


# drop_columns = dataprocessing.correlation(process_train)
# print(drop_columns)

dataset = pd.concat([data_train, data_test])
dataset = dataprocessing.preprocessing_onehot(dataset)
dataset.reset_index(drop=True, inplace=True)

dataset.to_csv('aaa.csv')
#dataset.drop(labels=drop_columns, axis=1, inplace=True)

columns = list(dataset)
columns.remove('playtime_forever')
predictors = columns

# split train and test data after preprocessing
df_train = dataset.head(356)
# t = dataprocessing.correlation(df_train)
# t.to_csv('corr.csv')

df_test = dataset.tail(90)

X_train = df_train[predictors]
pca = PCA(n_components=90)
#X_train = pca.fit_transform(X_train)

#print(pca.explained_variance_ratio_)

y_train = df_train['playtime_forever']
#X_train, X_test, y_train, y_test = train_test_split(df_train[predictors], df_train['playtime_forever'], test_size=0.34, random_state=42)
LR = linear_model.Ridge().fit(X_train,y_train)
#LR = linear_model.LinearRegression().fit(X_train, y_train)
n_folds = 3
kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
scores = cross_val_score(LR, X_train, y_train, scoring='mean_squared_error', cv = kf)
print(scores)
# X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.34, random_state=42)
# #scores = cross_val_score(LR, X_test, y_test, cv=1)
#
# print('Root Mean squared error: %.3f' % mean_squared_error(y_test2, LR.predict(X_test2))**0.5)##((y_test-LR.predict(X_test))**2).mean()
# print('Variance score: %.3f' % r2_score(y_test2,LR.predict(X_test2)))#1-((y_test-LR.predict(X_test))**2).sum()/((y_test - y_test.mean())**2).sum()
# print('score: %.3f' % LR.score(X_test,y_test))
# plt.scatter(X_test, y_test, color='green')
# plt.plot(X_test, LR.predict(X_test), color='red', linewidth=3)
# plt.show()
#joblib.dump(LR, "%s_linear_train_model.model" % (datetime.datetime.today().strftime("%Y%m%d")))

#output test playtime_forever
X_test = df_test[predictors]
#X_test = pca.fit_transform(X_test)
#print(pca.explained_variance_ratio_)
output = LR.predict(X_test)
result = pd.DataFrame(output)

result.rename(columns={0:'playtime_forever'}, inplace=True)

result[result['playtime_forever']<0] = 0
print(result)
result.to_csv("submission_ridge.csv")

#print(df_test)
#r = result['playtime_forever']
#r.to_csv("submission.csv")