########## CE302 Assignment AnswerSheet ################
#
# Use "tips" dataset from seaborn library
# Use EDA lecture notes to analyse the dataset and visualize the dataset.
#           1- Use all the functions: shape, head, dtypes, isnull, describe, info
#           2- Analysis of Categorical/Numerical Variables
#           3- Analysis of Correlation
# Make hypothesis and check your hypothesis. Make probability control by using Shapiro.
# Compute correlation coefficient for "total_bill" column.
# Make correlation reliability test (use pearsonr module)
# Write a short discussion about outputs.

# import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
from scipy import stats
import researchpy as rp
import matplotlib
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)  # changes the default display options
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

#######################################
#  Advanced Functional Exploratory Data Analysis
#######################################

# Load Dataset
df = sns.load_dataset("tips")               # loads "tips" dataset from seaborn library

print(df.head(10))                            # shows first five rows of the dataset
print(df.tail())                                # shows last five rows of the dataset
print(df.shape)                               # results in (number of rows, number of columns)
print(df.info())                                 # gives information about the dataframe and datatypes
print(df.dtypes)                              # returns a Series with the data type of each column
print(df.describe().T)                      # gives information about the count, mean, standard deviation, minimum, maximum and quartiles of quantitative variables
print(df.isnull().values.any())      # gives information whether there is any null value or not, results in True or False
print(df.isnull().sum())                  # gives the sum of number of null values
# or
def check_df(dataframe, head=10):
    print("########## First 10 Data #############")
    print(dataframe.head(head))
    print("########## Info #############")
    print(dataframe.info())
    print("########## Statistical Data #############")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("########## Null Data #############")
    print(dataframe.isnull().sum())
    print("########## Variable Types #############")
    print(dataframe.dtypes)

##########################################
#  Analysis of Categorical/Numerical Variables
##########################################
#
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
print(cat_cols)
# categorical variables are: "sex", "smoker", "day", "time"

print(df["sex"].unique())                   # returns unique values
print(df["sex"].nunique())                  # counts number of distinct values
print(df["sex"].value_counts())             # returns in counts of unique rows
print(df["smoker"].unique())
print(df["smoker"].nunique())
print(df["smoker"].value_counts())
print(df["day"].unique(),df["day"].nunique(),df["day"].value_counts(), sep="\n")
print(df["time"].unique(),df["time"].nunique(),df["time"].value_counts(),sep="\n")

rp.summary_cat(df[["sex","smoker","day","time"]])   # returns a data table including the counts and percentages of each category

def cat_summary(dataframe, col_name):                # creates a categorical summary function including the column count and ratio
    print(pd.DataFrame({col_name:  dataframe[col_name].value_counts(), "ratio": 100 * dataframe[col_name].value_counts() /len(dataframe)}))

cat_summary(df, "sex")
cat_summary(df, "smoker")
cat_summary(df, "day")
cat_summary(df, "time")

##########################################
# Data Visualization
##########################################
import matplotlib
matplotlib.use('TKAgg')
# scatter plot
plt.scatter(df["total_bill"],df["tip"], s=df["size"]*10, c="c", marker=r"1")
plt.title("total bill and tip")
plt.xlabel("total bill")
plt.ylabel("tip")
plt.show()

# histogram
plt.hist(df["sex"], label="gender")
plt.hist(df["smoker"], label="smoking")
plt.hist(df["day"], label="day")
plt.hist(df["time"], label="time")
plt.ylabel("frequency")
plt.legend()
plt.show()

##########################################
# Correlation Analysis
##########################################

df.head()
df.groupby('smoker').agg({'total_bill':'mean'})  # check if there is a mathemetically differance


# Quantitative variables are considered

df[["total_bill","tip","size"]].corr())    # finds the pairwise correlation using "pearson" method

# Description of the output :
#       there is a strong and positive correlation between total_bill and tip variables where the correlation coefficient is 0,67
#       there is a positive correlation between total_bill and size variables where the correlation coefficient is 0.60
#        there is a positive correlation between tip and size where the correlation coefficient is 0.49



##########################################
# AB Testing Case Study
##########################################

df = sns.load_dataset("tips")

##########################################
# HYPHOTHESIS
##########################################
# H0: There is no ccorrelation between two groups  ----   Correlation coefficient = 0
# H1: There is a ccorrelation between two groups   -----  Correlation coefficient ≠ 0

cor_total_bill_tip = stats.pearsonr(df["total_bill"],df["tip"])   # tests the hypothesis that correlation coefficient of "total_bill" and "tip" variables is 0
print("correlation coefficient: %.4f, p value: %.4f" %(cor_total_bill_tip))

# correlation coefficient is 0.6757 and p value < 0.05
# null hypothesis H0 can be rejected
# correlation coefficient of "total_bill" and "tip" variables is statistically significant

cor_total_bill_size = stats.pearsonr(df["total_bill"],df["size"])
print("correlation coefficient: %.4f, p value: %.4f" %cor_total_bill_size)
# correlation coefficient is 0.5983 and p value < 0.05
# null hypothesis H0 can be rejected
# correlation coefficient of "total_bill" and "size" variables is statistically significant

cor_tip_size = stats.pearsonr(df["tip"],df["size"])
print("correlation coefficient: %.4f, p value: %.4f" %cor_tip_size)
# correlation coefficient is 0.4893 and p value < 0.05
# null hypothesis H0 can be rejected
# correlation coefficient of "tip" and "size" variables is statistically significant

# Heatmap

f,ax=plt.subplots(figsize = (18,18))
sns.heatmap(df.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Map')
plt.savefig('graph.png')
plt.show()



##########################################
# Normality Test
##########################################
#
# H0: the sample has normal distribution
# H1: the sample does not have normal distribution

shapiro_total_bill = stats.shapiro(df["total_bill"])
print("shapiro test of normality p value: %.4f" %(shapiro_total_bill[1]))
# p value < 0.05
# H0 can be rejected which assumes the normality of the dataset
# it can be said that total_bill dataset is not normally distributed

shapiro_tip = stats.shapiro(df["tip"])
print("shapiro test of normality p value: %.4f" %(shapiro_tip[1]))
# p value < 0.05
# it can be said that tip dataset is not normally distributed

shapiro_size = stats.shapiro(df["size"])
print("shapiro test of normality p value: %.4f" %(shapiro_size[1]))
# p value < 0.05
# it can be said that size dataset is not normally distributed
