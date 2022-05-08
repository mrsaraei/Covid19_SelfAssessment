# -*- coding: utf-8 -*-
print("------------------------------------------------------")
print("---------------- Metadata Information ----------------")
print("------------------------------------------------------")
print("")

print("In the name of God")
print("Project: Self-Assessment of COVID_19 Using Machine Learning")
print("Creator: Mohammad Reza Saraei")
print("Contact: m.r.saraei@seraj.ac.ir")
print("University: Seraj Institute of Higher Education")
print("Supervisor: Dr. Saman Rajebi")
print("Created Date: February 12, 2022")
print("") 

# print("----------------------------------------------------")
# print("------------------ Import Libraries ----------------")
# print("----------------------------------------------------")
# print("")

import pandas as pd
from pandas import set_option
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# print("----------------------------------------------------")
# print("------------------ Data Ingestion ------------------")
# print("----------------------------------------------------")
# print("")

# Import DataFrame (.csv) by Pandas Library
df = pd.read_csv(r"C:\Users\Mohammad Reza\OneDrive\Documents\Paper\CoV19 Dataset\Cleaned Data\CoV19_PhS_ClS_Data_Cleaend.csv")

# print("----------------------------------------------------")
# print("----------------- Set Option -----------------------")
# print("----------------------------------------------------")
# print("")

set_option('display.max_rows', 500)
set_option('display.max_columns', 500)
set_option('display.width', 1000)

# Select Features (as "f") and Target (as "t") Data
f = df.iloc[:, 0: -1].values
t = df.iloc[:, -1].values     

print("------------------------------------------------------")
print("------------- Data Balancing By SMOTE ----------------")
print("------------------------------------------------------")
print("")

# Summarize Targets Distribution
print('Targets Distribution Before SMOTE:', sorted(Counter(t).items()))

# OverSampling (OS) Fit and Transform the DataFrame
OS = SMOTE()
f, t = OS.fit_resample(f, t)

# Summarize the New Targets Distribution
print('Targets Distribution After SMOTE:', sorted(Counter(t).items()))
print("")

print('Feature Set:', f.shape)
print('Target Set:', t.shape)
print("")

print("------------------------------------------------------")
print("------ Plotting Data Distribution After Balancing ----")
print("------------------------------------------------------")
print("")

# # Data Distribution by Scatter Plot
# layout = plt.figure(figsize = (20, 4))
# layout.add_subplot(151)
# plt.scatter(df.iloc[:, 0].values, df.iloc[:, 1].values, alpha = 0.6)
# layout.add_subplot(152)
# plt.scatter(df.iloc[:, 2].values, df.iloc[:, 3].values, alpha = 0.6)
# layout.add_subplot(153)
# plt.scatter(df.iloc[:, 4].values, df.iloc[:, 5].values, alpha = 0.6)
# layout.add_subplot(154)
# plt.scatter(df.iloc[:, 6].values, df.iloc[:, 7].values, alpha = 0.6)
# layout.add_subplot(155)
# plt.scatter(df.iloc[:, 8].values, df.iloc[:, 9].values, alpha = 0.6)
# plt.show()

# Data Distribution by Pair Plot
sns.pairplot(df, hue = '10')

# Data Distribution by Swarm Plot
layout = plt.figure(figsize = (20, 10))
layout.add_subplot(251)
sns.swarmplot(x = '10', y = df.iloc[:, 0].values, data = df)
layout.add_subplot(252)
sns.swarmplot(x = '10', y = df.iloc[:, 1].values, data = df)
layout.add_subplot(253)
sns.swarmplot(x = '10', y = df.iloc[:, 2].values, data = df)
layout.add_subplot(254)
sns.swarmplot(x = '10', y = df.iloc[:, 3].values, data = df)
layout.add_subplot(255)
sns.swarmplot(x = '10', y = df.iloc[:, 4].values, data = df)
layout.add_subplot(256)
sns.swarmplot(x = '10', y = df.iloc[:, 5].values, data = df)
layout.add_subplot(257)
sns.swarmplot(x = '10', y = df.iloc[:, 6].values, data = df)
layout.add_subplot(258)
sns.swarmplot(x = '10', y = df.iloc[:, 7].values, data = df)
layout.add_subplot(259)
sns.swarmplot(x = '10', y = df.iloc[:, 8].values, data = df)
layout.add_subplot(2, 5, 10)
sns.swarmplot(x = '10', y = df.iloc[:, 9].values, data = df)
plt.show()

# # Data Distribution by Strip Plot
# layout = plt.figure(figsize = (20, 10))
# layout.add_subplot(251)
# sns.stripplot(x = '10', y = df.iloc[:, 0].values, data = df)
# layout.add_subplot(252)
# sns.stripplot(x = '10', y = df.iloc[:, 1].values, data = df)
# layout.add_subplot(253)
# sns.stripplot(x = '10', y = df.iloc[:, 2].values, data = df)
# layout.add_subplot(254)
# sns.stripplot(x = '10', y = df.iloc[:, 3].values, data = df)
# layout.add_subplot(255)
# sns.stripplot(x = '10', y = df.iloc[:, 4].values, data = df)
# layout.add_subplot(256)
# sns.stripplot(x = '10', y = df.iloc[:, 5].values, data = df)
# layout.add_subplot(257)
# sns.stripplot(x = '10', y = df.iloc[:, 6].values, data = df)
# layout.add_subplot(258)
# sns.stripplot(x = '10', y = df.iloc[:, 7].values, data = df)
# layout.add_subplot(259)
# sns.stripplot(x = '10', y = df.iloc[:, 8].values, data = df)
# layout.add_subplot(2, 5, 10)
# sns.stripplot(x = '10', y = df.iloc[:, 9].values, data = df)
# plt.show()

# # Data Distribution by Count Plot
# layout = plt.figure(figsize = (20, 10))
# layout.add_subplot(251)
# sns.countplot(x = df.iloc[:, 0].values, data = df)
# layout.add_subplot(252)
# sns.countplot(x = df.iloc[:, 1].values, data = df)
# layout.add_subplot(253)
# sns.countplot(x = df.iloc[:, 2].values, data = df)
# layout.add_subplot(254)
# sns.countplot(x =  df.iloc[:, 3].values, data = df)
# layout.add_subplot(255)
# sns.countplot(x = df.iloc[:, 4].values, data = df)
# layout.add_subplot(256)
# sns.countplot(x = df.iloc[:, 5].values, data = df)
# layout.add_subplot(257)
# sns.countplot(x = df.iloc[:, 6].values, data = df)
# layout.add_subplot(258)
# sns.countplot(x = df.iloc[:, 7].values, data = df)
# layout.add_subplot(259)
# sns.countplot(x = df.iloc[:, 8].values, data = df)
# layout.add_subplot(2, 5, 10)
# sns.countplot(x = df.iloc[:, 9].values, data = df)
# plt.show()

# # Data Distribution by Histogram Plot
# layout = plt.figure(figsize = (20, 10))
# layout.add_subplot(251)
# sns.histplot(x = df.iloc[:, 0].values, data = df, kde = True, hue = df.iloc[:, -1].values)
# layout.add_subplot(252)
# sns.histplot(x = df.iloc[:, 1].values, data = df, kde = True, hue = df.iloc[:, -1].values)
# layout.add_subplot(253)
# sns.histplot(x = df.iloc[:, 2].values, data = df, kde = True, hue = df.iloc[:, -1].values)
# layout.add_subplot(254)
# sns.histplot(x = df.iloc[:, 3].values, data = df, kde = True, hue = df.iloc[:, -1].values)
# layout.add_subplot(255)
# sns.histplot(x = df.iloc[:, 4].values, data = df, kde = True, hue = df.iloc[:, -1].values)
# layout.add_subplot(256)
# sns.histplot(x = df.iloc[:, 5].values, data = df, kde = True, hue = df.iloc[:, -1].values)
# layout.add_subplot(257)
# sns.histplot(x = df.iloc[:, 6].values, data = df, kde = True, hue = df.iloc[:, -1].values)
# layout.add_subplot(258)
# sns.histplot(x = df.iloc[:, 7].values, data = df, kde = True, hue = df.iloc[:, -1].values)
# layout.add_subplot(259)
# sns.histplot(x = df.iloc[:, 8].values, data = df, kde = True, hue = df.iloc[:, -1].values)
# layout.add_subplot(2, 5, 10)
# sns.histplot(x = df.iloc[:, 9].values, data = df, kde = True, hue = df.iloc[:, -1].values)
# plt.show()

# Save DataFrame (f, t) After Munging
pd.DataFrame(f).to_csv(r"C:\Users\Mohammad Reza\OneDrive\Documents\Paper\CoV19 Dataset\Balanced Data\CoV19_PhS_ClS_Data_Balanced_f.csv", index = False)
pd.DataFrame(t).to_csv(r"C:\Users\Mohammad Reza\OneDrive\Documents\Paper\CoV19 Dataset\Balanced Data\CoV19_PhS_ClS_Data_Balanced_t.csv", index = False)

print("------------------------------------------------------")
print("------------------ Data Combination ------------------")
print("------------------------------------------------------")
print("")

# Import Again DataFrames (f, t) by Pandas Library
df_f = pd.read_csv(r"C:\Users\Mohammad Reza\OneDrive\Documents\Paper\CoV19 Dataset\Balanced Data\CoV19_PhS_ClS_Data_Balanced_f.csv")
df_t = pd.read_csv(r"C:\Users\Mohammad Reza\OneDrive\Documents\Paper\CoV19 Dataset\Balanced Data\CoV19_PhS_ClS_Data_Balanced_t.csv")

# Rename t Column
df_t.rename(columns = {'0': '10'}, inplace = True)

# Combination of DataFrames
df = pd.concat([df_f, df_t], axis = 1)

# Save Combination f and t DataFrames After Munging
pd.DataFrame(df).to_csv(r"C:\Users\Mohammad Reza\OneDrive\Documents\Paper\CoV19 Dataset\Balanced Data\CoV19_PhS_ClS_Data_Balanced.csv", index = False)

print("------------------------------------------------------")
print("-------- Data Understanding After Preparation --------")
print("------------------------------------------------------")
print("")

print("Dataset Overview:")
print("*****************")
print(df.head(10))
print("")

print("General Information:")
print("********************")
print(df.info())
print("")

print("Statistics Information:")
print("***********************")
print(df.describe(include="all"))
print("")

print("nSample & (nFeature + Target):", df.shape)
print("")

print("Samples Range:", df.index)
print("")

print(df.columns)
print("")

print("Missing Values (NaN):")
print("*********************")
print(df.isnull().sum())
print("")

print("Duplicate Records:", df.duplicated().sum())
print("")   

print("Features Correlations:")
print("**********************")
print(df.corr(method='pearson'))
print("")

print("------------------------------------------------------")
print("---------- Thank you for waiting, Good Luck ----------")
print("---------- Signature: Mohammad Reza Saraei -----------")
print("------------------------------------------------------")

