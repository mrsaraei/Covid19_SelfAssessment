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
import warnings
warnings.filterwarnings("ignore")

# print("----------------------------------------------------")
# print("------------------ Data Ingestion ------------------")
# print("----------------------------------------------------")
# print("")

# Import DataFrame (.csv) by Pandas Library
df = pd.read_csv(r"C:\Users\Mohammad Reza\OneDrive\Documents\Paper\CoV19 Dataset\Balanced Data\CoV19_PhS_ClS_Data_Balanced.csv")

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
print("--------- Dropping Nonsignificant Features -----------")
print("------------------------------------------------------")
print("")

# Get rid of Poor Features from DataFrame 
df = df.drop(['1', '6', '7', '3'], axis = 1)

print("------------------------------------------------------")
print("------ Plotting Data Distribution After Balancing ----")
print("------------------------------------------------------")
print("")

layout = plt.figure(figsize = (20, 4))
layout.add_subplot(231)
plt.scatter(f[:, 0], f[:, 2], c = 'Purple', alpha = 0.6)
layout.add_subplot(232)
plt.scatter(f[:, 4], f[:, 5], c = 'Purple', alpha = 0.6)
layout.add_subplot(233)
plt.scatter(f[:, 8], f[:, 9], c = 'Purple', alpha = 0.6)
plt.show()

# Save Combination f and t DataFrames After Munging
pd.DataFrame(df).to_csv(r"C:\Users\Mohammad Reza\OneDrive\Documents\Paper\CoV19 Dataset\Important Data\CoV19_PhS_ClS_Data_Important.csv", index = False)

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

print("Skewed Distribution of Features:")
print("********************************")
print(df.skew())
print("")
print(df.dtypes)
print("")

print("Target Distribution:")
print("********************")
print(df.groupby(df.iloc[:, -1].values).size())
print("")

print("------------------------------------------------------")
print("---------- Thank you for waiting, Good Luck ----------")
print("---------- Signature: Mohammad Reza Saraei -----------")
print("------------------------------------------------------")

