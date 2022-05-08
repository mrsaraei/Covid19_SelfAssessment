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

# print("----------------------------------------------------")
# print("------------------ Data Ingestion ------------------")
# print("----------------------------------------------------")
# print("")

# Import DataFrame (.csv) by Pandas Library
df_PhS = pd.read_csv(r"C:\Users\Mohammad Reza\OneDrive\Documents\Paper\CoV19 Dataset\Encoded Data\CoV19_PhS_Encoded.csv")
df_ClS = pd.read_csv(r"C:\Users\Mohammad Reza\OneDrive\Documents\Paper\CoV19 Dataset\Encoded Data\CoV19_ClS_Encoded.csv")

# Fusion of DataFrames (.csv) 
df = pd.concat([df_PhS, df_ClS], axis = 1)                         # df = np.concatenate((df_PhS, df_ClS), axis = 1)

print("------------------------------------------------------")
print("----------------- Data Annotation --------------------")
print("------------------------------------------------------")
print("")

# [0 = Unsuspecting]
# [1 = Suspected]
# [2 = Probable]

df['Target'] = 0                                                   # range(0, len(df))

def target(row):
    if (row['LoT_S'] == 2):
        return "2"  
    elif (row['Ch_P'] == 1) | (row['LoS_M_C'] == 1):
        return "1"
    elif ((row['Pyrexia'] == 1) | (row['Pyrexia'] == 2) | (row['Pyrexia'] == 3) | (row['Pyrexia'] == 4)) & (row['Cough'] == 2):
        return "1"
    elif (row['Hypoxemia'] == 3) & (row['Dyspnea'] == 1) & (row['Fatigue'] == 2):
        return "1"    
    elif ((row['Pyrexia'] == 1) | (row['Pyrexia'] == 2) | (row['Pyrexia'] == 3) | (row['Pyrexia'] == 4)) & ((row['Hypoxemia'] == 1) | (row['Hypoxemia'] == 2) | (row['Hypoxemia'] == 3)) & ((row['Cough'] == 1) | (row['Cough'] == 3)):
        return "1"
    elif ((row['Pyrexia'] == 1) | (row['Pyrexia'] == 2) | (row['Pyrexia'] == 3) | (row['Pyrexia'] == 4)) & ((row['Hypoxemia'] == 1) | (row['Hypoxemia'] == 2) | (row['Hypoxemia'] == 3)) & (row['Fatigue'] == 2):
        return "1"
    elif ((row['Pyrexia'] == 1) | (row['Pyrexia'] == 2) | (row['Pyrexia'] == 3) | (row['Pyrexia'] == 4)) & ((row['Hypoxemia'] == 1) | (row['Hypoxemia'] == 2) | (row['Hypoxemia'] == 3)) & (row['Headache'] == 1):
        return "1"
    elif ((row['Pyrexia'] == 1) | (row['Pyrexia'] == 2) | (row['Pyrexia'] == 3) | (row['Pyrexia'] == 4)) & ((row['Hypoxemia'] == 1) | (row['Hypoxemia'] == 2) | (row['Hypoxemia'] == 3)) & (row['GI'] == 1):
        return "1"
    elif ((row['Pyrexia'] == 1) | (row['Pyrexia'] == 2) | (row['Pyrexia'] == 3) | (row['Pyrexia'] == 4)) & ((row['Hypoxemia'] == 1) | (row['Hypoxemia'] == 2) | (row['Hypoxemia'] == 3)) & (row['Dyspnea'] == 1):
        return "1"
    elif ((row['Pyrexia'] == 1) | (row['Pyrexia'] == 2) | (row['Pyrexia'] == 3) | (row['Pyrexia'] == 4)) & ((row['Cough'] == 1) | (row['Cough'] == 3)) & (row['Fatigue'] == 2):
        return "1"
    elif ((row['Pyrexia'] == 1) | (row['Pyrexia'] == 2) | (row['Pyrexia'] == 3) | (row['Pyrexia'] == 4)) & ((row['Cough'] == 1) | (row['Cough'] == 3)) & (row['Headache'] == 1):
        return "1"
    elif ((row['Pyrexia'] == 1) | (row['Pyrexia'] == 2) | (row['Pyrexia'] == 3) | (row['Pyrexia'] == 4)) & ((row['Cough'] == 1) | (row['Cough'] == 3)) & (row['GI'] == 1):
        return "1"
    elif ((row['Pyrexia'] == 1) | (row['Pyrexia'] == 2) | (row['Pyrexia'] == 3) | (row['Pyrexia'] == 4)) & ((row['Cough'] == 1) | (row['Cough'] == 3)) & (row['Dyspnea'] == 1):
        return "1"    
    elif ((row['Pyrexia'] == 1) | (row['Pyrexia'] == 2) | (row['Pyrexia'] == 3) | (row['Pyrexia'] == 4)) & (row['Fatigue'] == 2) & (row['Headache'] == 1):
        return "1"    
    elif ((row['Pyrexia'] == 1) | (row['Pyrexia'] == 2) | (row['Pyrexia'] == 3) | (row['Pyrexia'] == 4)) & (row['Fatigue'] == 2) & (row['GI'] == 1):
        return "1"    
    elif ((row['Pyrexia'] == 1) | (row['Pyrexia'] == 2) | (row['Pyrexia'] == 3) | (row['Pyrexia'] == 4)) & (row['Fatigue'] == 2) & (row['Dyspnea'] == 1):
        return "1"        
    elif ((row['Pyrexia'] == 1) | (row['Pyrexia'] == 2) | (row['Pyrexia'] == 3) | (row['Pyrexia'] == 4)) & (row['Headache'] == 1) & (row['GI'] == 1):
        return "1"       
    elif ((row['Pyrexia'] == 1) | (row['Pyrexia'] == 2) | (row['Pyrexia'] == 3) | (row['Pyrexia'] == 4)) & (row['Headache'] == 1) & (row['Dyspnea'] == 1):
        return "1"       
    elif ((row['Pyrexia'] == 1) | (row['Pyrexia'] == 2) | (row['Pyrexia'] == 3) | (row['Pyrexia'] == 4)) & (row['GI'] == 1) & (row['Dyspnea'] == 1):
        return "1"           
    elif ((row['Hypoxemia'] == 1) | (row['Hypoxemia'] == 2) | (row['Hypoxemia'] == 3)) & ((row['Cough'] == 1) | (row['Cough'] == 3)) & (row['Fatigue'] == 2):
        return "1"            
    elif ((row['Hypoxemia'] == 1) | (row['Hypoxemia'] == 2) | (row['Hypoxemia'] == 3)) & ((row['Cough'] == 1) | (row['Cough'] == 3)) & (row['Headache'] == 1):
        return "1"                
    elif ((row['Hypoxemia'] == 1) | (row['Hypoxemia'] == 2) | (row['Hypoxemia'] == 3)) & ((row['Cough'] == 1) | (row['Cough'] == 3)) & (row['GI'] == 1):
        return "1"                   
    elif ((row['Hypoxemia'] == 1) | (row['Hypoxemia'] == 2) | (row['Hypoxemia'] == 3)) & ((row['Cough'] == 1) | (row['Cough'] == 3)) & (row['Dyspnea'] == 1):
        return "1"                    
    elif ((row['Hypoxemia'] == 1) | (row['Hypoxemia'] == 2) | (row['Hypoxemia'] == 3)) & (row['Fatigue'] == 2) & (row['Headache'] == 1):
        return "1"        
    elif ((row['Hypoxemia'] == 1) | (row['Hypoxemia'] == 2) | (row['Hypoxemia'] == 3)) & (row['Fatigue'] == 2) & (row['GI'] == 1):
        return "1"        
    elif ((row['Hypoxemia'] == 1) | (row['Hypoxemia'] == 2) | (row['Hypoxemia'] == 3)) & (row['Fatigue'] == 2) & (row['Dyspnea'] == 1):
        return "1"        
    elif ((row['Hypoxemia'] == 1) | (row['Hypoxemia'] == 2) | (row['Hypoxemia'] == 3)) & (row['Headache'] == 1) & (row['GI'] == 1):
        return "1"
    elif ((row['Hypoxemia'] == 1) | (row['Hypoxemia'] == 2) | (row['Hypoxemia'] == 3)) & (row['Headache'] == 1) & (row['Dyspnea'] == 1):
        return "1"    
    elif (row['Hypoxemia'] == 1) & (row['GI'] == 1) & (row['Dyspnea'] == 1):
        return "1" 
    elif ((row['Cough'] == 1) | (row['Cough'] == 3)) & (row['Fatigue'] == 2) & (row['Headache'] == 1):
        return "1"
    elif ((row['Cough'] == 1) | (row['Cough'] == 3)) & (row['Fatigue'] == 2) & (row['GI'] == 1):
        return "1"
    elif ((row['Cough'] == 1) | (row['Cough'] == 3)) & (row['Fatigue'] == 2) & (row['Dyspnea'] == 1):
        return "1"
    elif ((row['Cough'] == 1) | (row['Cough'] == 3)) & (row['Headache'] == 1) & (row['GI'] == 1):
        return "1"   
    elif ((row['Cough'] == 1) | (row['Cough'] == 3)) & (row['Headache'] == 1) & (row['Dyspnea'] == 1):
        return "1"   
    elif (row['Fatigue'] == 2) & (row['Headache'] == 1) & (row['GI'] == 1):
        return "1"
    elif (row['Fatigue'] == 2) & (row['Headache'] == 1) & (row['Dyspnea'] == 1):
        return "1"
    elif (row['Fatigue'] == 2) & (row['GI'] == 1) & (row['Dyspnea'] == 1):
        return "1"
    elif (row['Headache'] == 1) & (row['GI'] == 1) & (row['Dyspnea'] == 1):
        return "1"
    else:
        return "0"

df = df.assign(Target=df.apply(target, axis = 1))

print("")

# Save DataFrame After Encoding
df.to_csv(r"C:\Users\Mohammad Reza\OneDrive\Documents\Paper\CoV19 Dataset\Annotated Data\CoV19_PhS_ClS_Annotated.csv", index = False)

print("------------------------------------------------------")
print("---------- Thank you for waiting, Good Luck ----------")
print("---------- Signature: Mohammad Reza Saraei -----------")
print("------------------------------------------------------")