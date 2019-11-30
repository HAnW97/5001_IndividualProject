from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import pandas as pd
import xgboost as xgb

import dataprocessing


TRAIN_DATA_PATH = "../train.csv"
TEST_DATA_PATH = "../test.csv"

# load data
data_train = pd.read_csv(TRAIN_DATA_PATH, sep=',')
data_test = pd.read_csv(TEST_DATA_PATH, sep=',')

# preprocess training data
data_train = dataprocessing.preprocessing(data_train)
train_shape = data_train.shape[0]  # nums of row

# preprocess testing data
data_test = dataprocessing.preprocessing(data_test)
test_shape = data_test.shape[0]

# concat train.csv and test.csv together to do one-hot, ensure they have same columns
dataset = pd.concat([data_train, data_test])
dataset = dataprocessing.preprocessing_onehot(dataset)
dataset.reset_index(drop=True, inplace=True)

# drop columns of low correlation
# corr_train = dataprocessing.preprocessing_onehot(data_train)
# drop_columns = dataprocessing.correlation(corr_train)
# drop_columns = dataprocessing.dropColumns(dataset)
# dataset.drop(labels=drop_columns, axis=1, inplace=True)

# split train and test data after preprocessing
df_train = dataset.head(train_shape)
df_test = dataset.tail(test_shape)

# get predictors
predictors = list(dataset)
predictors.remove('playtime_forever')
X_train = df_train[predictors]

# pca = PCA(n_components=90)
# X_train = pca.fit_transform(X_train)

y_train = df_train['playtime_forever']

# train xgboost model
xgb_model = xgb.XGBRegressor(max_depth=7, learning_rate=0.1, n_estimators=100, subsample=0.8, silent=False, reg_alpha=0, objective='reg:gamma')
xgb_model.fit(X_train, y_train)


# k-fold validation
n_folds = 5
kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
scores = cross_val_score(xgb_model, X_train, y_train, scoring='mean_squared_error', cv = kf)
print(scores)

# predict
X_test = df_test[predictors]
#X_test = pca.fit_transform(X_test)
output = xgb_model.predict(X_test)

# output result to submission.csv
result = pd.DataFrame(output)
result.reset_index(drop=True,inplace=True)
result.index.name = 'id'
result.rename(columns={0:'playtime_forever'}, inplace=True)

result[result['playtime_forever']<0] = 0
print(result)

result.to_csv("submission_xgb.csv")