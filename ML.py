import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the dataset
data = pd.read_csv("data/properties.csv")

# Define features and target
X = data.drop(columns=['id', 'price'])  # Features
y = data['price']  # Target

# Define preprocessing steps for numeric and categorical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps for both numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing pipeline
X_preprocessed = preprocessor.fit_transform(X)

# Train RandomForestRegressor model
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_preprocessed, y)
score = model_rf.score(X_preprocessed, y)
print("R^2 score:", score)

# Save the model
# joblib.dump(model_rf, "models/random_forest_model.joblib")

# # Load pre-trained models and scaler
# loaded_rf_model = joblib.load("models/random_forest_model.joblib")
# loaded_scaler = joblib.load("models/scaler.joblib")

# # Scale new data
# new_data = pd.DataFrame({'Construction_Year': [2000], 'Bedroom_Count': [2], 'Terrace_Surface': [20], 
#                           'Habitable_Surface': [120], 'Facade_Count': [2], 
#                          'Swimming_Pool': [0], 'Indoor_Parking': [0], 'Outdoor_Parking': [0]})
# new_data_scaled = loaded_scaler.transform(new_data)

# # Debugging output
# print("Scaled New Data:")
# print(new_data_scaled)

# # Predict prices using Random Forest Regressor
# predicted_price_rf = loaded_rf_model.predict(new_data_scaled)
# print("Predicted Price using Random Forest Regressor:", predicted_price_rf)




# # Train LinearRegression model
# model_lr = LinearRegression()
# model_lr.fit(X_train, y_train)

# # Evaluate RandomForestRegressor model
# y_pred_train_rf = model_rf.predict(X_train)
# y_pred_test_rf = model_rf.predict(X_test)
# mae_train_rf = mean_absolute_error(y_train, y_pred_train_rf)
# mae_test_rf = mean_absolute_error(y_test, y_pred_test_rf)
# print("RandomForestRegressor:")
# print(f"Mean Absolute Error on training set: {mae_train_rf}")
# print(f"Mean Absolute Error on test set: {mae_test_rf}")

# # Evaluate LinearRegression model
# y_pred_train_lr = model_lr.predict(X_train)
# y_pred_test_lr = model_lr.predict(X_test)
# mae_train_lr = mean_absolute_error(y_train, y_pred_train_lr)
# mae_test_lr = mean_absolute_error(y_test, y_pred_test_lr)
# print("\nLinearRegression:")
# print(f"Mean Absolute Error on training set: {mae_train_lr}")
# print(f"Mean Absolute Error on test set: {mae_test_lr}")

# # Plotting Predicted vs Actual Prices
# plt.figure(figsize=(12, 8))

# # Plot for RandomForestRegressor
# plt.subplot(2, 1, 1)
# plt.scatter(y_train, y_pred_train_rf, color='blue', label='Training Data')
# plt.scatter(y_test, y_pred_test_rf, color='red', label='Test Data')
# plt.plot(y_test, y_test, color='green', linestyle='--', label='Actual')
# plt.title('RandomForestRegressor: Predicted vs Actual')
# plt.xlabel('Actual Price')
# plt.ylabel('Predicted Price')
# plt.legend()
# plt.grid(True)

# # Plot for LinearRegression
# plt.subplot(2, 1, 2)
# plt.scatter(y_train, y_pred_train_lr, color='blue', label='Training Data')
# plt.scatter(y_test, y_pred_test_lr, color='red', label='Test Data')
# plt.plot(y_test, y_test, color='green', linestyle='--', label='Actual')
# plt.title('LinearRegression: Predicted vs Actual')
# plt.xlabel('Actual Price')
# plt.ylabel('Predicted Price')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()
