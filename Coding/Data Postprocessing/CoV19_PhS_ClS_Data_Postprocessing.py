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
import numpy as np
from pandas import set_option
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# print("----------------------------------------------------")
# print("------------------ Data Ingestion ------------------")
# print("----------------------------------------------------")
# print("")

# Import DataFrame (.csv) by Pandas Library
df = pd.read_csv(r"C:\Users\Mohammad Reza\OneDrive\Documents\Paper\CoV19 Dataset\Important Data\CoV19_PhS_ClS_Data_Important.csv")

# print("----------------------------------------------------")
# print("----------------- Set Option -----------------------")
# print("----------------------------------------------------")
# print("")

set_option('display.max_rows', 500)
set_option('display.max_columns', 500)
set_option('display.width', 1000)

print("------------------------------------------------------")
print("---------------- Data Preprocessing ------------------")
print("------------------------------------------------------")
print("")

# Replace Question Mark to NaN:
df.replace("?", np.nan, inplace = True)

# Remove Duplicate Samples
df = df.drop_duplicates()
print("Duplicate Records After Removal:", df.duplicated().sum())
print("")

# Replace Mean instead of Missing Values
imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imp.fit(df)
df = imp.transform(df)
print("Mean Value For NaN Value:", "{:.3f}".format(df.mean()))
print("")

# Reordering Records / Samples / Rows
print("Reordering Records:")
print("*******************")
df = pd.DataFrame(df).reset_index(drop = True)
print(df)
print("")

print("------------------------------------------------------")
print("-------- Data Distribution After Preparation ---------")
print("------------------------------------------------------")
print("")

print("nSample & (nFeature + Target):", df.shape)
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

# Select Features (as "f") and Target (as "t") Data
f = df.iloc[:, 0: -1].values
t = df.iloc[:, -1].values  

print("------------------------------------------------------")
print("----------- Plotting Distribution of Data ------------")
print("------------------------------------------------------")
print("")

layout = plt.figure(figsize = (20, 4))
layout.add_subplot(231)
plt.scatter(f[:, 0], f[:, 1], c = 'Purple', alpha = 0.6)
layout.add_subplot(232)
plt.scatter(f[:, 2], f[:, 3], c = 'Purple', alpha = 0.6)
layout.add_subplot(233)
plt.scatter(f[:, 4], f[:, 5], c = 'Purple', alpha = 0.6)
plt.show()

# Save Combination f and t DataFrames After Munging
pd.DataFrame(df).to_csv(r"C:\Users\Mohammad Reza\OneDrive\Documents\Paper\CoV19 Dataset\Postprocessed Data\CoV19_PhS_ClS_Data_postprocessed.csv", index = False)


print("------------------------------------------------------")
print("---------- Thank you for waiting, Good Luck ----------")
print("---------- Signature: Mohammad Reza Saraei -----------")
print("------------------------------------------------------")


