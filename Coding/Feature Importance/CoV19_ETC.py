## -*- coding: utf-8 -*-
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
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# print("----------------------------------------------------")
# print("------------------ Data Ingestion ------------------")
# print("----------------------------------------------------")
# print("")

# Load DataFrame
df = pd.read_csv(r"C:\Users\Mohammad Reza\OneDrive\Documents\Paper\CoV19 Dataset\Balanced Data\CoV19_PhS_ClS_Data_Balanced.csv")

# Select Features (as "f") and Target (as "t") Data
f = df.iloc[:, 0: -1]
t = df.iloc[:, -1]

# Feature Importance By Extra Trees Classifier
ETC = ExtraTreesClassifier()
ETC.fit(f,t)
print(ETC.feature_importances_)

# Plotting Feature Importance By Extra Trees Classifier
FI = pd.Series(ETC.feature_importances_, index = f.columns)
FI.nlargest(14).plot(kind = 'barh')
plt.show()

print("------------------------------------------------------")
print("---------- Thank you for waiting, Good Luck ----------")
print("---------- Signature: Mohammad Reza Saraei -----------")
print("------------------------------------------------------")

