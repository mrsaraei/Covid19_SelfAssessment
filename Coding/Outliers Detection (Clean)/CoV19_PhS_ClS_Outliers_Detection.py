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
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings("ignore")

# print("----------------------------------------------------")
# print("------------------ Data Ingestion ------------------")
# print("----------------------------------------------------")
# print("")

# Import DataFrame (.csv) by Pandas Library
df = pd.read_csv(r"C:\Users\Mohammad Reza\OneDrive\Documents\Paper\CoV19 Dataset\Preprocessed Data\CoV19_PhS_ClS_Data_Preprocessed.csv")

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
print("----------------- Outliers Detection -----------------")
print("------------------------------------------------------")
print("")

# Identify Outliers in the Training Data
ISF = IsolationForest(n_estimators = 100, contamination = 0.1, bootstrap = True, n_jobs = -1)

# Fitting Outliers Algorithms on the Training Data
ISF = ISF.fit_predict(f, t)

# Select All Samples that are not Outliers
Mask = ISF != -1
f, t = f[Mask, :], t[Mask]

print('Feature Train Set:', f.shape)
print('Target Train Set:', t.shape)
print("")

print("------------------------------------------------------")
print("  Plotting Data Distribution After Outliers Detection ")
print("------------------------------------------------------")
print("")

layout = plt.figure(figsize = (20, 4))
layout.add_subplot(151)
plt.scatter(f[:, 0], f[:, 1], c = 'green', alpha = 0.6)
layout.add_subplot(152)
plt.scatter(f[:, 2], f[:, 3], c = 'green', alpha = 0.6)
layout.add_subplot(153)
plt.scatter(f[:, 4], f[:, 5], c = 'green', alpha = 0.6)
layout.add_subplot(154)
plt.scatter(f[:, 6], f[:, 7], c = 'green', alpha = 0.6)
layout.add_subplot(155)
plt.scatter(f[:, 8], f[:, 9], c = 'green', alpha = 0.6)
plt.show()

# Save DataFrame (f, t) After Munging
pd.DataFrame(f).to_csv(r"C:\Users\Mohammad Reza\OneDrive\Documents\Paper\CoV19 Dataset\Cleaned Data\CoV19_PhS_ClS_Data_Cleaend_f.csv", index = False)
pd.DataFrame(t).to_csv(r"C:\Users\Mohammad Reza\OneDrive\Documents\Paper\CoV19 Dataset\Cleaned Data\CoV19_PhS_ClS_Data_Cleaend_t.csv", index = False)

print("------------------------------------------------------")
print("------------------ Data Combination ------------------")
print("------------------------------------------------------")
print("")

# Import Again DataFrames (f, t) by Pandas Library
df_f = pd.read_csv(r"C:\Users\Mohammad Reza\OneDrive\Documents\Paper\CoV19 Dataset\Cleaned Data\CoV19_PhS_ClS_Data_Cleaend_f.csv")
df_t = pd.read_csv(r"C:\Users\Mohammad Reza\OneDrive\Documents\Paper\CoV19 Dataset\Cleaned Data\CoV19_PhS_ClS_Data_Cleaend_t.csv")

# Rename t Column
df_t.rename(columns = {'0': '10'}, inplace = True)

# Combination of DataFrames
df = pd.concat([df_f, df_t], axis = 1)

# Save Combination f and t DataFrames After Munging
pd.DataFrame(df).to_csv(r"C:\Users\Mohammad Reza\OneDrive\Documents\Paper\CoV19 Dataset\Cleaned Data\CoV19_PhS_ClS_Data_Cleaend.csv", index = False)

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

