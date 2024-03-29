import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from openpyxl import Workbook, load_workbook

# Global variable to hold valid breeds
valid_breeds = []

def load_data(filepath='data.xlsx'):
    df = pd.read_excel(filepath)
    df['Age in years'] = df['Age in years'].apply(lambda x: try_parse_float(x, default=0.5))
    # Load valid breeds from the Excel file into a global variable
    global valid_breeds
    valid_breeds = df['Breed'].str.strip().str.title().unique().tolist()
    return df

def try_parse_float(value, default=0.5):
    try:
        return float(value)
    except ValueError:
        return default

def preprocess_and_split(df):
    numeric_features = ['Age in years', 'Weight']
    numeric_transformer = StandardScaler()

    categorical_features = ['Gender', 'Breed']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ])

    X = df[['Age in years', 'Gender', 'Breed', 'Weight']]
    y = df['Broken bone']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor

def train_model(X_train, y_train, preprocessor):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline

def make_predictions_and_save(model, input_features):
    input_df = pd.DataFrame([input_features], columns=input_features.keys())
    probabilities = model.predict_proba(input_df)
    probabilities_df = pd.DataFrame(probabilities, columns=model.classes_).T
    probabilities_df.reset_index(inplace=True)
    probabilities_df.columns = ['Bone', 'Risk (%)']
    probabilities_df['Risk (%)'] *= 100
    animal_details_df = pd.DataFrame([input_features])
    with pd.ExcelWriter("Assessed fracture risk.xlsx") as writer:
        animal_details_df.to_excel(writer, index=False, sheet_name="Animal Details")
        probabilities_df.to_excel(writer, index=False, sheet_name="Fracture Risk")

def validate_input(input_features):
    global valid_breeds
    valid_genders = ['Male', 'Female']
    gender = input_features['Gender']
    
    if gender not in valid_genders:
        return False, "Invalid gender. Expected 'Male' or 'Female'."
    
    age = input_features['Age in years']
    if not (0 <= age <= 20):
        return False, "Invalid age. Age should be between 0 and 20."
    
    weight = input_features['Weight']
    if not (1 <= weight <= 100):
        return False, "Invalid weight. Weight should be between 1 and 100 kg."
    
    # Normalize and validate breed
    breed = input_features['Breed'].strip().title()
    if breed not in valid_breeds:
        return False, f"Invalid breed: {breed}. Please enter a breed exactly as listed."

    return True, "Valid input."

def main():
    df = load_data('data.xlsx')  # Adjust the path as needed
    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(df)
    model = train_model(X_train, y_train, preprocessor)

    age_years = float(input("Enter the dog's age in years (for less than 1 year, use decimal): "))
    gender = input("Enter the dog's gender (Male/Female): ")
    breed = input("Enter the dog's breed: ").strip().title()  # Normalize input
    weight = float(input("Enter the dog's weight in kg: "))

    input_features = {
        'Age in years': age_years,
        'Gender': gender,
        'Breed': breed,
        'Weight': weight
    }

    is_valid, message = validate_input(input_features)
    if not is_valid:
        print(message)  # Print the validation error
        return
    
    make_predictions_and_save(model, input_features)

if __name__ == "__main__":
    main()
