###############################################
# DATA ANALYSIS WITH PYTHON
###############################################
# - NumPy
# - Pandas
# - Matplotlib & Seaborn
#
#############################################

import numpy as np

a = [1, 2, 3, 4]
b = [2, 3, 4, 5]
ab = []

for i in range(0, len(a)):
    ab.append(a[i]*b[i])
type(ab)

# if define them as numpy array
# its easier to do calculation and computation
a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
a * b

#############################################
# Creating Numpy Arrays
#############################################

a = np.array([1, 2, 3,4 ,5])
type(a)
np.zeros(10, dtype=int)
np.random.randint(0, 10, size=10)
np.random.normal(10, 4, (3,4))

a  = np.random.randint(0, 10, size= 5)
a.ndim
a.shape
a.dtype
a.size

np.random.randint(1, 10, size=9)
np.random.randint(1, 10, size=9).reshape(3, 3)

ar = np.random.randint(1, 10, size=9)
ar.reshape(3, 3)

a = np.random.randint(10, size=10)
a
a[0]
a[0:5]
a[0] = 999
a

m = np.random.randint(10, size=(3, 5))
m

m[0, 0]
m[1, 1]
m[2, 3]

m[2, 3] = 299
m

m[:, 0]
m[1, :]
m[0:2, 0:3]
m


#############################################
# Fancy Index
#############################################

v = np.arange(0, 30, 3)
v
v[1]
v[4]

catch = [1, 2, 3]
v[catch]

a = np.arange(0, 30, 5)
a
b = a[[1, 2, -1]]
print(b)

b = np.take(a,[1, 2, -1])
print(b)


#############################################
# Conditions on Numpy
#############################################
import numpy as np
v = np.array([1, 2, 3, 4, 5])

ab = []

for i in v:
    if i < 3:
        ab.append(i)

ab

v < 3
v[v<3]
v[v > 3]
v[v != 3]
v[v == 3]


#############################################
# Mathematical Operations
#############################################
import numpy as np
v = np.array([1, 2, 3, 4, 5])

v / 5
v * 5 / 10
v ** 2
v - 1

np.subtract(v, 1)
np.add(v, 1)
np.mean(v)
np.sum(v)
np.min(v)
np.max(v)
np.var(v)
v = np.subtract(v, 1)
v

# 5*x0 + x1 = 12
# x0 + 3*x1 = 10

a = np.array([[5, 1], [1, 3]])
b = np.array([12, 10])

np.linalg.solve(a, b)

#############################################
# PANDAS
#############################################

import pandas as pd

s = pd.Series([70, 17, 32, 4, 52])
type(s)
s.index
s.size
s.ndim
type(s.values)
s.head(3)
s.tail()


#############################################
# Reading Data
#############################################
import pandas as pd

df = pd.read_csv("/home/komec/LECTURES/000_VBO_BootCamp/measurement_problems/datasets/course_reviews.csv")
df.head()

# pandas cheatsheet

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().sum()
df.isnull().values.any()
df["class"].value_counts()

#############################################
# Selection in Pandas
#############################################

df = sns.load_dataset("titanic")
df.head()

df[0:13]
df.drop(0, axis=0).head()

delete_indexes = [1, 3, 5]
df.drop(delete_indexes, axis=0, inplace = True)
df.head(10)

#######################
# Convert Variable to Index
#######################

df["age"].head()
df.index = df["age"]

df.drop("age", axis = 1, inplace =True)
df

#######################
# Convert Index to Variable
#######################

df.index

df["age"] = df.index

df.head()
df.drop("age", axis=1, inplace=True)

df.reset_index().head()
df = df.reset_index()
df.head()

#######################
# Computations on variables
#######################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

"age" in df

df["age"].head()
df.age.head()
type(df["age"].head())

df[["age"]].head()
type(df[["age"]].head())

df[["age", "alive"]]

cols = ["age", "alive","fare"]
type(df[cols])

df["age2"] = df["age"]**2
df["age3"] = df["age"] / df["age2"]
df

df.loc[:, ~df.columns.str.contains("age")].head()

# tilde operator

df = pd.DataFrame({'InvoiceNo': ['aaC','ff','lC'],'a':[1,2,5]})
print (df)

#check if column contains C
print (df['InvoiceNo'].str.contains('C'))

#inversing mask
print (~df['InvoiceNo'].str.contains('C'))

#######################
# iloc & loc
#######################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df.iloc[0:3]
df.iloc[0, 0]
df.iloc[0:3, 0:4]
df.loc[0:3, "age"]
df.loc[0:3, cols]
df.iloc[0, 3]
df.loc[0:3]

#######################
# Conditional Selection
#######################

df[df["age"] > 50].head()
df[df["age"]> 50]["age"].count()

df.loc[df["age"] > 50, ["age", "class"]].head()

df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["age", "class"]].head()

df["embark_town"].value_counts()

df_new =  df.loc[(df["age"] > 50) & (df["sex"] == "male")  \
    & (df["embark_town"] == "Cherbourg"),\
    ["age", "class", "embark_town"]]

df_new

#############################################
# Aggregation & Grouping
#############################################
df = sns.load_dataset("titanic")
df.head()

df["age"].mean()
df.groupby("sex")["age"].mean()

df.groupby("sex").agg({"age":"mean"})
df.groupby("sex").agg({"age":["mean","sum"]})

df.groupby("sex").agg({"age":["mean","sum"], "survived": "mean"})

df.groupby(["sex", "embark_town"]).agg({"age":["mean","sum"], "survived": "mean"})

df.groupby(["sex", "embark_town","class"]).\
    agg({"age":["mean"], "survived": "mean", "sex" : "count"})


#######################
# Pivot table
#######################

df.pivot_table("survived", "sex", "embarked")

df.pivot_table("survived", "sex", ["embarked", "class"])
df["new_age"] = pd.cut(df["age"], [0, 14, 25, 40, 90])
df["new_age"].head()

df.pivot_table("survived", "sex", ["new_age", "class"])

#############################################
# Apply & Lambda
#############################################
df = sns.load_dataset("titanic")
df.head()

df["age2"] = df["age"]*2
df["age3"] = df["age"]*5
(df["age"]/10).head()

for col in df.columns:
    if "age" in col:
        print((df[col]/10).head())


df[["age", "age2", "age3"]].apply(lambda x: x/10).head()
df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std()).head()



def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()


#############################################
# Join Calculations
#############################################
m = np.random.randint(1, 30, size = (5, 3))
m

df1 = pd. DataFrame(m , columns=["var1", "var2", "var3"])
df2 = df1 * 99

pd.concat([df1, df2], ignore_index=True)

#######################
# Merge
#######################
df1 = pd.DataFrame({'employees': ['john', 'dennis', 'mark', 'maria'],
                    'group': ['accounting', 'engineering', 'engineering', 'hr']})

df2 = pd.DataFrame({'employees': ['mark', 'john', 'dennis', 'maria'],
                    'start_date': [2010, 2009, 2014, 2019]})

pd.merge(df1, df2, on="employees")

df3 = pd.merge(df1, df2)

df4 = pd.DataFrame({'group': ['accounting', 'engineering', 'hr'],
                    'manager': ['Caner', 'Mustafa', 'Berkcan']})


pd.merge(df3, df4)

#############################################
# MATPLOTLIB
#############################################

# Kategorik değişken: sütun grafik. countplot bar
# Sayısal değişken: hist, boxplot

import tkinter
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df['sex'].value_counts().plot(kind='bar')
plt.show()

plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])
plt.show()


#######################
# plot
#######################

x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y)
plt.show()

plt.plot(x, y, 'o')
plt.show()

x = np.array([2, 4, 6, 8, 10])
y = np.array([1, 3, 5, 7, 9])

plt.plot(x, y)
plt.show()

plt.plot(x, y, 'o')
plt.show()


#############################################
# SEABORN
#############################################
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df = sns.load_dataset("tips")
df.head()

df["sex"].value_counts()
sns.countplot(x=df["sex"], data=df)
plt.show()

df['sex'].value_counts().plot(kind='bar')
plt.show()


################################################
# Pandas Exercises
###############################################

import pandas as pd
pd.set_option('display.max_columns', None)

df = sns.load_dataset("tips")
df.info

# Q1  - Find the sum, min, max and average of the total_bill values according to the
#            categories (Dinner, Lunch) of the time variable.
df.groupby("time").agg({'total_bill': ["sum", "min", "max", "mean"] })


# Q2 - Find the sum, min, max and average of total_bill values by days and time.
df.groupby(["day","time"]).agg({'total_bill': ["sum", "min", "max", "mean"] })


# Q3 -Find the sum, min, max and average of the total_bill and "tip" values of the
#           lunch time and female customers according to the day.
df.loc[(df["time"] == "Lunch") &  (df["sex"] == "Female")].groupby(["day"]).\
    agg({'total_bill': ["sum", "min", "max", "mean"], "tip": ["sum", "min", "max", "mean"] })


# Q4 - What is the average of orders with size less than 3 and total_bill greater than 10? (use loc)





















































