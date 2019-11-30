import numpy
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler


def correlation(df):
    columns = list(df)
    t = df[columns].corr()
    drop_columns = t[(np.abs(t["playtime_forever"] < 0.0000001))].index
    return drop_columns


def preprocessing_onehot(df):
    # one-hot encoding   category data: 'genres', 'categories' and 'tags'
    df_genre = df['genres'].str.get_dummies(sep=',').add_prefix('genres_')
    df = pd.concat((df, pd.DataFrame(df_genre)), axis=1)
    df_category = df['categories'].str.get_dummies(sep=',').add_prefix('categories')
    df = pd.concat((df, pd.DataFrame(df_category)), axis=1)
    df_tag = df['tags'].str.get_dummies(sep=',').add_prefix('tags_')
    df = pd.concat((df, pd.DataFrame(df_tag)), axis=1)
    df.drop(labels=['genres', 'categories', 'tags'], axis=1, inplace=True)
    return df

def dropOutliers(df):
    # process outliers
    outliers = df[(df['price'] > 100000)].index.tolist()
    print(outliers)
    for i in outliers:
        df.drop(index=i, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def preprocessing(df):
    # fill the rows containing NaN
    mean_val_positive = df['total_positive_reviews'].mean()
    df['total_positive_reviews'].fillna(mean_val_positive, inplace=True)
    mean_val_negative = df['total_negative_reviews'].mean()
    df['total_negative_reviews'].fillna(mean_val_negative, inplace=True)

    # transform boolean 'is_free' to int 0/1
    is_free_transfer = {
        False: 0,
        True: 1
    }
    df.replace(is_free_transfer, inplace=True)

    # transform 'purchase_date' and 'release_date' to datetime64
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['release_date'] = pd.to_datetime(df['release_date'])

    # get today's date
    now = datetime.now()

    df['today_date'] = pd.to_datetime(now.strftime('%Y-%m-%d'))

    # get days number
    df['diff_purchase_date'] = (df['today_date'] - df['purchase_date']).dt.days
    df['diff_release_date'] = (df['today_date'] - df['release_date']).dt.days

    mean_val_purchase = df['diff_purchase_date'].mean()
    df['diff_purchase_date'].fillna(mean_val_purchase, inplace=True)
    mean_val_release = df['diff_release_date'].mean()
    df['diff_release_date'].fillna(mean_val_release, inplace=True)

    df_transform = StandardScaler().fit_transform(df[['price', 'total_positive_reviews','total_negative_reviews',
                                                    'diff_purchase_date','diff_release_date']])
    df[['price', 'total_positive_reviews','total_negative_reviews','diff_purchase_date','diff_release_date']] = df_transform


    df.drop(labels=['id', 'purchase_date','release_date', 'today_date'], axis=1, inplace=True)

    return df

if __name__ == '__main__':
    #os.chdir('../Data')
    df = pd.read_csv('train.csv', low_memory=False)
    df_1 = preprocessing(df)
    df_1.to_csv('cleanfile.csv')