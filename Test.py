import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

import joblib
import numpy as np
import pandas as pd
from model import df1, discrp, ektra7at, rnd_forest

# Load your trained model AFTER it has been trained
loaded_rf = joblib.load("random_forest.joblib")

# Function to predict the disease
def predd(x, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17):
    psymptoms = [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17]
    
    # Define your list 'a' based on the data you want to use for comparison
    a = ['symptom1', 'symptom2', 'symptom3', ...]  # Replace with actual symptom names
    
    b = np.array(df1["weight"])
    
    for j in range(len(psymptoms)):
        for k in range(len(a)):
            if psymptoms[j] == a[k]:
                psymptoms[j] = b[k]
    
    psy = [psymptoms]
    pred2 = x.predict(psy)
    
    disp = discrp[discrp['Disease'] == pred2[0]]
    disp = disp.values[0][1]
    
    recomnd = ektra7at[ektra7at['Disease'] == pred2[0]]
    c = np.where(ektra7at['Disease'] == pred2[0])[0][0]
    
    precuation_list = []
    for i in range(1, len(ektra7at.iloc[c])):
       precuation_list.append(str(ektra7at.iloc[c, i])) 
    
    # Create a message box to display the results
    message = f"The Disease Name: {pred2[0]}\n" \
              f"The Disease Description: {disp}\n" \
              f"Medication Recommendation(Initial Treatment):\n" \
              f"{', '.join(precuation_list)}"
    
    messagebox.showinfo("Prediction Result", message)

# Create the main GUI window
root = tk.Tk()
root.title("Disease Prediction")

symptom_labels = []
symptom_entries = []

for i in range(17):
    label = ttk.Label(root, text=f"Symptom {i + 1}:")
    label.grid(row=i, column=0, padx=10, pady=5)
    symptom_labels.append(label)

    entry = ttk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=5)
    symptom_entries.append(entry)

def predd_gui():
    symptoms = [entry.get() for entry in symptom_entries]
    print("Symptoms entered:", symptoms)  # Add this line for debugging

    # Convert numeric entries to float, replace empty entries with 0
    symptoms = [float(symptom) if symptom.replace(".", "", 1).isdigit() else 0 for symptom in symptoms]
    print("Symptoms after conversion:", symptoms)  # Add this line for debugging

    predd(rnd_forest, *symptoms)

predict_button = ttk.Button(root, text="Predict Disease", command=predd_gui)
predict_button.grid(row=17, columnspan=2, padx=10, pady=10)

result_label = ttk.Label(root, text="")
result_label.grid(row=18, columnspan=2, padx=10, pady=5)

description_label = ttk.Label(root, text="")
description_label.grid(row=19, columnspan=2, padx=10, pady=5)

precautions_label = ttk.Label(root, text="")
precautions_label.grid(row=20, columnspan=2, padx=10, pady=5)

root.mainloop()
