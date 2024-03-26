import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from openpyxl import Workbook, load_workbook

def load_data(filepath='data.xlsx'):
    df = pd.read_excel(filepath)
    
    # Apply a lambda function to convert each value in 'Age in years', using try_parse_float for conversion
    df['Age in years'] = df['Age in years'].apply(lambda x: try_parse_float(x, default=0.5))
    
    return df

def try_parse_float(value, default=0.5):
    """
    Attempts to convert a value to a float. If the conversion fails, returns the default value.
    
    Parameters:
    - value: The value to convert to float.
    - default: The default value to use if conversion fails.
    
    Returns:
    - The converted value as float, or the default value if conversion is not possible.
    """
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
    # Directly create a DataFrame with the correct structure and column names
    input_df = pd.DataFrame([input_features], columns=input_features.keys())
    
    # Now, use the model to predict probabilities. Since the model is a pipeline
    # that includes the preprocessor, it will handle the transformation.
    probabilities = model.predict_proba(input_df)

    # Generate a DataFrame from the probabilities for better formatting and saving
    probabilities_df = pd.DataFrame(probabilities, columns=model.classes_).T
    probabilities_df.reset_index(inplace=True)
    probabilities_df.columns = ['Bone', 'Risk (%)']
    
    # Convert probabilities to percentages
    probabilities_df['Risk (%)'] *= 100

    # Prepare the animal details for inclusion in the Excel file
    animal_details_df = pd.DataFrame([input_features])

    # Save both the animal details and the probabilities to the Excel file
    with pd.ExcelWriter("Assessed fracture risk.xlsx") as writer:
        animal_details_df.to_excel(writer, index=False, sheet_name="Animal Details")
        probabilities_df.to_excel(writer, index=False, sheet_name="Fracture Risk")

# Adjust the main function to remove the 'preprocessor' variable, as it's now included in the pipeline
        
def validate_input(input_features):
    # Define expected categories for 'Gender'
    valid_genders = ['Male', 'Female']
    gender = input_features['Gender']
    
    # Validate gender
    if gender not in valid_genders:
        return False, "Invalid gender. Expected 'Male' or 'Female'."
    
    # Validate age (assuming dogs less than 20 years)
    age = input_features['Age in years']
    if not (0 <= age <= 20):
        return False, "Invalid age. Age should be between 0 and 20."
    
    # Validate weight (assuming weight should be reasonable for dogs, e.g., between 1 kg and 100 kg)
    weight = input_features['Weight']
    if not (1 <= weight <= 100):
        return False, "Invalid weight. Weight should be between 1 and 100 kg."
    
    # If all checks pass
    return True, "Valid input."

def main():
    df = load_data('data.xlsx')  # Make sure to adjust this path
    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(df)
    model = train_model(X_train, y_train, preprocessor)

    # Collect patient information from the user
    age_years = float(input("Enter the dog's age in years (for less than 1 year, use decimal): "))
    gender = input("Enter the dog's gender (Male/Female): ")
    breed = input("Enter the dog's breed: ")
    weight = float(input("Enter the dog's weight in kg: "))

    input_features = {
        'Age in years': age_years,
        'Gender': gender,
        'Breed': breed,
        'Weight': weight
    }
    
    # Validate input before prediction
    is_valid, message = validate_input(input_features)
    if not is_valid:
        print(message)  # Print why the input is invalid
        return  # Stop execution if the input is invalid
    
    make_predictions_and_save(model, input_features)

# Example usage
if __name__ == "__main__":
    main()






