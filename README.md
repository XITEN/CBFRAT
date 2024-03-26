# Project Overview

This project introduces "risk.py," a Python script developed for veterinary purposes, focusing on predicting the risk of bone fractures in dogs. It integrates a sophisticated machine learning pipeline, including preprocessing steps for normalization of numerical data and encoding of categorical data, utilizing a RandomForestClassifier for the prediction mechanism. The primary objective is to assess and present the fracture risk based on various inputs such as the dog's age, gender, breed, and weight. The outcomes, alongside the input data, are systematically compiled into an Excel file named "Assessed fracture risk.xlsx," facilitating further analysis and record-keeping.

## Installation Requirements

To ensure smooth operation of "risk.py," your system needs to have Python 3.x installed, along with the following Python packages:

- `pandas`
- `scikit-learn`
- `openpyxl`

These dependencies can be installed using pip with the following command:

pip install pandas scikit-learn openpyxl

## Running the Script

Execute "risk.py" by opening a terminal or command prompt, navigating to the directory containing the script, and running it through Python. The script prompts for input regarding the dog's details (age, gender, breed, weight). Based on the provided information, it processes and predicts the fracture risk, then saves both the input and predictions in the Excel file "Assessed fracture risk.xlsx".

python risk.py


## Detailed Functionality

- **Data Processing:** The script begins by loading canine data from an Excel file. It proceeds to convert age values into floats, scales numerical features, and applies one-hot encoding to categorical features.
- **Model Training and Prediction:** Utilizes a RandomForestClassifier model trained on the processed data to predict bone fracture risks for new canine profiles.
- **Input Validation:** Includes checks to validate the input data against reasonable ranges for dogs' age, gender, and weight to ensure data integrity.
- **Results Documentation:** Outputs are documented in an Excel file, "Assessed fracture risk.xlsx," compiling both the canine details and the model's fracture risk predictions for easy review and further use.


