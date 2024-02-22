import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

def train():
    # Load the data
    data = pd.read_csv("data/properties.csv")

    # Define features to use
    num_features = ["nbr_frontages", "construction_year", "total_area_sqm", "surface_land_sqm", "nbr_bedrooms", "primary_energy_consumption_sqm"]
    fl_features_apartment = ["fl_terrace"]
    fl_features_house = ["fl_garden", "fl_swimming_pool"]
    cat_features = ["property_type", "subproperty_type", "region", "province", "locality", "epc", "heating_type"]

    # Split the data into features and target
    X = data[num_features + fl_features_apartment + fl_features_house + cat_features]
    y = data["price"]

    # Handle outliers using Z-score method
    z_scores = np.abs((X[num_features] - X[num_features].mean()) / X[num_features].std())
    z_score_threshold = 3
    outlier_mask = (z_scores < z_score_threshold).all(axis=1)
    X = X[outlier_mask]
    y = y[outlier_mask]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Separate features based on property type
    X_train_apartment = X_train[X_train["property_type"] == "APARTMENT"]
    X_train_house = X_train[X_train["property_type"] == "HOUSE"]

    X_test_apartment = X_test[X_test["property_type"] == "APARTMENT"]
    X_test_house = X_test[X_test["property_type"] == "HOUSE"]

    # Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy="mean")

    # For apartment
    X_train_apartment[num_features] = imputer.fit_transform(X_train_apartment[num_features])
    X_test_apartment[num_features] = imputer.transform(X_test_apartment[num_features])

    # For house
    X_train_house[num_features] = imputer.fit_transform(X_train_house[num_features])
    X_test_house[num_features] = imputer.transform(X_test_house[num_features])

    # Convert categorical columns with one-hot encoding using OneHotEncoder
    enc = OneHotEncoder()

    # For apartment
    enc.fit(X_train_apartment[cat_features])
    X_train_apartment_cat = enc.transform(X_train_apartment[cat_features]).toarray()
    X_test_apartment_cat = enc.transform(X_test_apartment[cat_features]).toarray()

    # For house
    enc.fit(X_train_house[cat_features])
    X_train_house_cat = enc.transform(X_train_house[cat_features]).toarray()
    X_test_house_cat = enc.transform(X_test_house[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns for each property type
    X_train_apartment = pd.concat([
        X_train_apartment[num_features + fl_features_apartment].reset_index(drop=True),
        pd.DataFrame(X_train_apartment_cat, columns=enc.get_feature_names_out())
    ], axis=1)

    X_test_apartment = pd.concat([
        X_test_apartment[num_features + fl_features_apartment].reset_index(drop=True),
        pd.DataFrame(X_test_apartment_cat, columns=enc.get_feature_names_out())
    ], axis=1)

    X_train_house = pd.concat([
        X_train_house[num_features + fl_features_house].reset_index(drop=True),
        pd.DataFrame(X_train_house_cat, columns=enc.get_feature_names_out())
    ], axis=1)

    X_test_house = pd.concat([
        X_test_house[num_features + fl_features_house].reset_index(drop=True),
        pd.DataFrame(X_test_house_cat, columns=enc.get_feature_names_out())
    ], axis=1)

    # Train the linear regression model for apartment
    linear_model_apartment = LinearRegression()
    linear_model_apartment.fit(X_train_apartment, y_train[X_train_apartment.index])

    # Train the random forest regressor model for apartment
    rf_model_apartment = RandomForestRegressor(n_estimators=200, random_state=1)
    rf_model_apartment.fit(X_train_apartment, y_train[X_train_apartment.index])

    # Train the linear regression model for house
    linear_model_house = LinearRegression()
    linear_model_house.fit(X_train_house, y_train[X_train_house.index])

    # Train the random forest regressor model for house
    rf_model_house = RandomForestRegressor(n_estimators=200, random_state=1)
    rf_model_house.fit(X_train_house, y_train[X_train_house.index])

    # Evaluate the models for apartment
    train_linear_score_apartment = r2_score(y_train[X_train_apartment.index], linear_model_apartment.predict(X_train_apartment))
    test_linear_score_apartment = r2_score(y_test[X_test_apartment.index], linear_model_apartment.predict(X_test_apartment))

    train_rf_score_apartment = r2_score(y_train[X_train_apartment.index], rf_model_apartment.predict(X_train_apartment))
    test_rf_score_apartment = r2_score(y_test[X_test_apartment.index], rf_model_apartment.predict(X_test_apartment))

    # Evaluate the models for house
    train_linear_score_house = r2_score(y_train[X_train_house.index], linear_model_house.predict(X_train_house))
    test_linear_score_house = r2_score(y_test[X_test_house.index], linear_model_house.predict(X_test_house))

    train_rf_score_house = r2_score(y_train[X_train_house.index], rf_model_house.predict(X_train_house))
    test_rf_score_house = r2_score(y_test[X_test_house.index], rf_model_house.predict(X_test_house))

    print("Linear Regression Model for Apartment:")
    print(f"Train R² score: {train_linear_score_apartment}")
    print(f"Test R² score: {test_linear_score_apartment}")

    print("\nRandom Forest Regressor Model for Apartment:")
    print(f"Train R² score: {train_rf_score_apartment}")
    print(f"Test R² score: {test_rf_score_apartment}")

    print("\nLinear Regression Model for House:")
    print(f"Train R² score: {train_linear_score_house}")
    print(f"Test R² score: {test_linear_score_house}")

    print("\nRandom Forest Regressor Model for House:")
    print(f"Train R² score: {train_rf_score_house}")
    print(f"Test R² score: {test_rf_score_house}")

    # Save the models
    artifacts = {
        "linear_model_apartment": linear_model_apartment,
        "rf_model_apartment": rf_model_apartment,
        "linear_model_house": linear_model_house,
        "rf_model_house": rf_model_house,
        "features": {
            "num_features": num_features,
            "fl_features_apartment": fl_features_apartment,
            "fl_features_house": fl_features_house,
            "cat_features": cat_features
        },
        "imputer": imputer,
        "enc": enc
    }
    joblib.dump(artifacts, "models/artifacts.joblib")

    # Predict on the test set for apartment
    y_pred_linear_apartment = linear_model_apartment.predict(X_test_apartment)
    y_pred_rf_apartment = rf_model_apartment.predict(X_test_apartment)

    # Predict on the test set for house
    y_pred_linear_house = linear_model_house.predict(X_test_house)
    y_pred_rf_house = rf_model_house.predict(X_test_house)

    # Calculate RMSE for Linear Regression model for apartment
    rmse_linear_apartment = np.sqrt(mean_squared_error(y_test[X_test_apartment.index], y_pred_linear_apartment))

    # Calculate RMSE for Random Forest model for apartment
    rmse_rf_apartment = np.sqrt(mean_squared_error(y_test[X_test_apartment.index], y_pred_rf_apartment))

    # Calculate RMSE for Linear Regression model for house
    rmse_linear_house = np.sqrt(mean_squared_error(y_test[X_test_house.index], y_pred_linear_house))

    # Calculate RMSE for Random Forest model for house
    rmse_rf_house = np.sqrt(mean_squared_error(y_test[X_test_house.index], y_pred_rf_house))

    print("RMSE for Linear Regression Model for Apartment:", rmse_linear_apartment)
    print("RMSE for Random Forest Model for Apartment:", rmse_rf_apartment)
    print("RMSE for Linear Regression Model for House:", rmse_linear_house)
    print("RMSE for Random Forest Model for House:", rmse_rf_house)

    # Plot the predictions for apartment
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].scatter(y_test[X_test_apartment.index], y_pred_linear_apartment, color='blue')
    axes[0].plot([min(y_test[X_test_apartment.index]), max(y_test[X_test_apartment.index])], [min(y_test[X_test_apartment.index]), max(y_test[X_test_apartment.index])], color='red', linestyle='--')
    axes[0].set_title("Linear Regression (Apartment): Actual vs Predicted Prices")
    axes[0].set_xlabel('Actual Prices')
    axes[0].set_ylabel('Predicted Prices')

    axes[1].scatter(y_test[X_test_apartment.index], y_pred_rf_apartment, color='blue')
    axes[1].plot([min(y_test[X_test_apartment.index]), max(y_test[X_test_apartment.index])], [min(y_test[X_test_apartment.index]), max(y_test[X_test_apartment.index])], color='red', linestyle='--')
    axes[1].set_title("Random Forest Regression (Apartment): Actual vs Predicted Prices")
    axes[1].set_xlabel('Actual Prices')
    axes[1].set_ylabel('Predicted Prices')

    plt.tight_layout()
    plt.show()

    # Plot the predictions for house
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].scatter(y_test[X_test_house.index], y_pred_linear_house, color='blue')
    axes[0].plot([min(y_test[X_test_house.index]), max(y_test[X_test_house.index])], [min(y_test[X_test_house.index]), max(y_test[X_test_house.index])], color='red', linestyle='--')
    axes[0].set_title("Linear Regression (House): Actual vs Predicted Prices")
    axes[0].set_xlabel('Actual Prices')
    axes[0].set_ylabel('Predicted Prices')

    axes[1].scatter(y_test[X_test_house.index], y_pred_rf_house, color='blue')
    axes[1].plot([min(y_test[X_test_house.index]), max(y_test[X_test_house.index])], [min(y_test[X_test_house.index]), max(y_test[X_test_house.index])], color='red', linestyle='--')
    axes[1].set_title("Random Forest Regression (House): Actual vs Predicted Prices")
    axes[1].set_xlabel('Actual Prices')
    axes[1].set_ylabel('Predicted Prices')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train()
