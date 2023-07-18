import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import statsmodels.api as sm


df = pd.read_csv('<The File Path>')

#df.dtypes

### Treating the database first:

# Removing date and ID_Lead
df.drop(['Data do Lead', 'ID_Lead'], axis=1, inplace=True)

# Create a list of the variables with object data type
variables = df.select_dtypes(include='object').columns

# Initialize an empty dictionary to store the index mappings
index_mappings = {}

# Loop through each variable
for variable in variables:
    unique_values = df[variable].unique()
    index_mappings[variable] = {}

    # Assign indexes starting from 0 to each unique value
    for i, value in enumerate(unique_values):
        index_mappings[variable][value] = i

    # Convert the variable values to indexes
    df[variable] = df[variable].map(index_mappings[variable])

# Implementing the Logistic Regression Model:

# Split the database
X = df.drop('Conversão', axis=1)
y = df['Conversão']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability scores for class 1 convert which means buy (Conversão == 1)

# # Evaluate the model's accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")

# # Print the probability scores for each prediction
# print("Probability scores for each prediction:")
# for i, proba in enumerate(y_pred_proba):
#     print(f"Prediction {i+1}: {proba}")

# # Create a DataFrame with the test variables and probabilities
# results_df = X_test.copy()
# results_df['Conversão_Prediction'] = y_pred
# results_df['Conversão_Probability'] = y_pred_proba

# Create a DataFrame with the test variables and probability scores
result_df = X_test.copy()
result_df['Conversão'] = y_test
result_df['Predicted_Probability'] = y_pred_proba
#result_df['Predicted_Conversion'] = y_pred


# Map the converted indexes back to string names
index_mappings_inverse = {variable: {index: value for value, index in mapping.items()} for variable, mapping in index_mappings.items()}
for variable, mapping_inverse in index_mappings_inverse.items():
    result_df[variable] = result_df[variable].map(mapping_inverse)


# Export the DataFrame to an Excel file
result_df.to_excel('<The File Path and name that you want to save>', index=False)

# Coefficients

# Add a constant column to X_train
X_train = sm.add_constant(X_train)

# Create and train the logistic regression model using StatsModels
model = sm.Logit(y_train, X_train)
result = model.fit()

# Get the coefficients, standard errors, z-values, and p-values
coefficients = result.params[1:]  # Exclude the constant term
std_errors = result.bse[1:]  # Exclude the constant term
z_values = coefficients / std_errors
p_values = result.pvalues[1:]  # Exclude the constant term

# Create a DataFrame to store the coefficient information
coefficient_df = pd.DataFrame({'Variable': X_train.columns[1:], 'Coefficient': coefficients, 'Std. Error': std_errors, 'Z-value': z_values, 'Pr(>|z|)': p_values})

# # Print the coefficient information
# print(coefficient_df)

coefficient_df.to_csv('<The File Path and name that you want to save>', index=False)
