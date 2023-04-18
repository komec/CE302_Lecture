####################################
#  Feature Eng, and Data Pre-Processing
####################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
import matplotlib
matplotlib.use('TKAgg')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = sns.load_dataset('titanic')
df.shape
df.head()

#############################################
# 1. Outliers
#############################################

# plot outliers
sns.boxplot(x=df["age"])
plt.show()

# catch ourliers
q1 = df["age"].quantile(0.25)
q3 = df["age"].quantile(0.75)
iqr = q3 - q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

df[(df["age"] < low) | (df["age"] > up)]

df[(df["age"] < low) | (df["age"] > up)].index

# is outlier exist?
df[(df["age"] < low) | (df["age"] > up)]
df[(df["age"] < low) | (df["age"] > up)].any(axis=None)
df[~((df["age"] < low) | (df["age"] > up))].any(axis=None)

df[(df["age"] < low)].any(axis=None)


# function for outliers

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, "age")
outlier_thresholds(df, "fare")

low, up = outlier_thresholds(df, "fare")

df[(df["fare"] < low) | (df["fare"] > up)].head()
df[(df["fare"] < low) | (df["fare"] > up)].index

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "age")
check_outlier(df, "fare")

###################
# grab_col_names
###################
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols
num_cols

for col in num_cols:
    print(col, check_outlier(df, col))

###################
# Aykırı Değerlerin Kendilerine Erişmek
###################

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "age")

grab_outliers(df, "age", True)

age_index = grab_outliers(df, "age", True)


outlier_thresholds(df, "age")
check_outlier(df, "age")
grab_outliers(df, "age", True)

#############################################
# Solutions for Outlier Problems
#############################################

###################
# Delete
###################

low, up = outlier_thresholds(df, "fare")
df.shape

df[~((df["fare"] < low) | (df["fare"] > up))].shape

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.shape

for col in num_cols:
    new_df = remove_outlier(df, col)

df.shape[0] - new_df.shape[0]

###################
# re-assignment with thresholds
###################
df = sns.load_dataset('titanic')
low, up = outlier_thresholds(df, "fare")

df[((df["fare"] < low) | (df["fare"] > up))]["fare"]

df.loc[((df["fare"] < low) | (df["fare"] > up)), "fare"]

df.loc[(df["fare"] > up), "fare"] = up

df.loc[(df["fare"] < low), "fare"] = low

def replace_with_thresholds (dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]
df.shape

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

###################
# Recap
###################

df = sns.load_dataset('titanic')
outlier_thresholds(df, "age")
check_outlier(df, "age")
grab_outliers(df, "age", index=True)

remove_outlier(df, "age").shape
replace_with_thresholds(df, "age")
check_outlier(df, "age")


#############################################
# Missing Values
#############################################

df = sns.load_dataset('titanic')
df.head()

df.isnull().values.any()

df.isnull().sum()

df.notnull().sum()

df.isnull().sum().sum()

df[df.isnull().any(axis=1)]

df[df.notnull().all(axis=1)]

df.isnull().sum().sort_values(ascending=False)

(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)
missing_values_table(df, True)


#############################################
# Solution to missing value problem
#############################################

missing_values_table(df)

###################
# Delete
###################
df.dropna().shape

###################
# Assign mean, median etc
###################

df["age"].fillna(df["age"].mean()).isnull().sum()
df["age"].fillna(df["age"].median()).isnull().sum()
df["age"].fillna(0).isnull().sum()


df["embarked"].fillna(df["embarked"].mode()[0]).isnull().sum()
df["embarked"].fillna("missing")

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

###################
# Kategorik Değişken Kırılımında Değer Atama
###################

df.groupby("sex")["age"].mean()

df["age"].mean()

df["age"].fillna(df.groupby("sex")["age"].transform("mean")).isnull().sum()

df.groupby("sex")["age"].mean()["female"]

df.loc[(df["age"].isnull()) & (df["sex"]=="female"), "age"] = df.groupby("sex")["age"].mean()["female"]

df.loc[(df["age"].isnull()) & (df["sex"]=="male"), "age"] = df.groupby("sex")["age"].mean()["male"]

df.isnull().sum()


###################
# Recap
###################

df = load()
missing_values_table(df)
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
df["age"].fillna(df.groupby("sex")["age"].transform("mean")).isnull().sum()

