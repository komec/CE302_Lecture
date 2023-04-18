###############################################
# COMPREHENSIONS
###############################################

#######################
# List Comprehension
#######################

import pandas as pd
import numpy as np

salaries = [1000, 2000, 3000, 4000, 5000]

def new_salary(x):
    return x * 20 / 100 + x

for salary in salaries:
    print(new_salary(salary))

null_list = []
for salary in salaries:
    null_list.append(new_salary(salary))

null_list = []
for salary in salaries:
    if salary > 3000:
        null_list.append(new_salary(salary))
    else:
        null_list.append((new_salary(salary * .2)+ salary ))


[salary  * 2 for salary in salaries if salary < 3000 ]
[salary * 2 if salary < 3000 else salary for salary in salaries]
[new_salary(salary * 2) if salary < 3000 else new_salary(salary * 0.2 + salary) for salary in salaries]

# - ---

students = ["John", "Mark", "Venessa", "Mariam"]
churn_stdnts =  ["John", "Venessa"]

[student.lower() if student in churn_stdnts else student.upper() for student in students]

# ----

# before:
# ['total', 'speeding', 'alcohol', 'not_distracted', 'no_previous', 'ins_premium', 'ins_losses', 'abbrev']

# after:
# ['TOTAL', 'SPEEDING', 'ALCOHOL', 'NOT_DISTRACTED', 'NO_PREVIOUS', 'INS_PREMIUM', 'INS_LOSSES', 'ABBREV']

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

for col in df.columns:
    print(col.upper())

df.columns = [col.upper() for col in df.columns]

["FLAG_" + col for col in df.columns if "INS" in col]

["FLAG_"+col if "INS" in col else  "NOFLAG_" +col for col in df.columns]

#######################
# Dict Comprehension
#######################

dictionary = {'a': 1,
              'b': 2,
              'c': 3,
              'd': 4}

dictionary.keys()
dictionary.values()
dictionary.items()

{k : v **2 for (k, v) in dictionary.items()}

{k.upper(): v for (k, v) in dictionary.items()}

{k.upper(): v **2  for (k, v) in dictionary.items()}


#####

df = sns.load_dataset("car_crashes")
df.columns

soz = {}
agg_list = ["mean", "min", "max", "sum"]
num_cols = [col for col in df.columns if df[col].dtype != "O"]
num_cols

for col in num_cols:
    soz[col] = df[col].agg(agg_list)

result = pd.DataFrame(soz)
result


# comprehension
num_cols = [col for col in df.columns if df[col].dtype != "O"]
new_dict = {col : agg_list for col in num_cols}
df[num_cols].agg(new_dict)

# ----------------------------------------------------------------------------------------------------
#############################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
#############################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()

def check_df(dataframe, head =10):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Dtypes #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)


#############################################
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts()
df['sex'].unique()
df["class"].nunique()

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

# df["alive"].dtypes
# str(df["alive"].dtypes)
num_but_cat = [col for col in df.columns if df[col].nunique() < 10  and  df[col].dtypes in ["int", "float"]]

cat_but_car =  [col for col in df.columns if df[col].nunique() > 20  and  str(df[col].dtypes) in ["category", "object"]]

cat_cols =  cat_cols +  num_but_cat

#
100 * df["survived"].value_counts() /len(df)

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name:  dataframe[col_name].value_counts() , "Ratio": 100 * dataframe[col_name].value_counts() /len(dataframe)}))
    print("#############################################")

# cat_summary(df, "survived")

for col in cat_cols:
    cat_summary(df, col)

# PLOT


























