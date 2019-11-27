
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
import dataprocessing


# training data path
TRAIN_DATA_PATH = "train.csv"
TEST_DATA_PATH = "test.csv"
# preprocess both train and test data
data_train = pd.read_csv(TRAIN_DATA_PATH, sep=',')
data_test = pd.read_csv(TEST_DATA_PATH, sep=',')

#data_train = dataprocessing.dropNaN(data_train)
#data_train = dataprocessing.dropOutliers(data_train)
data_train = dataprocessing.preprocessing_prediction(data_train)
train_shape = data_train.shape[0]

data_test = dataprocessing.preprocessing_prediction(data_test)
test_shape = data_test.shape[0]

# drop_columns = dataprocessing.correlation(process_train)
# print(drop_columns)

dataset = pd.concat([data_train, data_test])
dataset = dataprocessing.preprocessing_onehot(dataset)
# dataset.drop(labels=['genres','categories','tags'], axis=1, inplace=True)
dataset.reset_index(drop=True, inplace=True)

#drop_columns = dataprocessing.dropColumns(dataset)
#dataset.drop(labels=drop_columns, axis=1, inplace=True)

columns = list(dataset)
columns.remove('playtime_forever')
predictors = columns

# split train and test data after preprocessing
df_train = dataset.head(train_shape)
# t = dataprocessing.correlation(df_train)
# t.to_csv('corr.csv')

df_test = dataset.tail(test_shape)

X_train = df_train[predictors]

#pca = PCA(n_components=200)
#X_train = pca.fit_transform(X_train)
#print(pca.explained_variance_ratio_)
y_train = df_train['playtime_forever']

#forest = DecisionTreeRegressor(max_depth=100, max_features=0.6, min_samples_leaf=3, random_state=42, presort=True)
#forest = AdaBoostRegressor(n_estimators=50)

#forest = RandomForestRegressor(n_estimators=100,criterion="mse",random_state=42,min_samples_leaf=13, max_features=0.5, n_jobs=-1, oob_score=True)
#model = xgb.XGBRegressor(max_depth=6, learning_rate=0.1, n_estimators=160, silent=False, reg_alpha=0, reg_lambda=1)
#model = xgb.XGBRegressor(max_depth=6, learning_rate=0.1, n_estimators=150, subsample=0.8, reg_alpha=0, reg_lambda=1, silent=False)
model = xgb.XGBRegressor(max_depth=7, learning_rate=0.1, n_estimators=180, subsample=0.8, silent=False, reg_alpha=0, objective='reg:gamma')
model.fit(X_train,y_train)


#X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.34, random_state=0)
# n_folds = 3
# kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
# scores = cross_val_score(model, X_train, y_train, scoring='mean_squared_error', cv = kf)
# print(scores)
#print('Root Mean squared error: %.3f' % mean_squared_error(y_test2, svm.predict(X_test2))**0.5)##((y_test-LR.predict(X_test))**2).mean()
#print('Variance score: %.3f' % r2_score(y_test2,svm.predict(X_test2)))#1-((y_test-LR.predict(X_test))**2).sum()/((y_test - y_test.mean())**2).sum()

X_test = df_test[predictors]
#X_test = svd.fit_transform(X_test)
#X_test = pca.fit_transform(X_test)
#print(pca.explained_variance_ratio_)
output = model.predict(X_test)
result = pd.DataFrame(output)
result.reset_index(drop=True,inplace=True)
result.index.name = 'id'

result.rename(columns={0:'playtime_forever'}, inplace=True)

result[result['playtime_forever']<0] = 0
print(result)
result.to_csv("submission_xgb.csv")