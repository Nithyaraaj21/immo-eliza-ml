import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
import numpy as np
import matplotlib.pyplot as plt

def train():
    # Load the data
    data = pd.read_csv("data/properties.csv")

    # Define features to use
    num_features = ["total_area_sqm", "surface_land_sqm", "nbr_bedrooms", "terrace_sqm", "garden_sqm"]
    fl_features = ["fl_terrace", "fl_furnished", "fl_open_fire", "fl_garden", "fl_swimming_pool", "fl_floodzone", "fl_double_glazing"]
    cat_features = ["equipped_kitchen", "property_type", "subproperty_type", "region", "province", "locality", "state_building", "epc", "heating_type"]

    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Handle outliers using Z-score method on training set
    z_scores_train = np.abs((X_train[num_features] - X_train[num_features].mean()) / X_train[num_features].std())
    z_score_threshold = 4
    outlier_mask_train = (z_scores_train < z_score_threshold).all(axis=1)

    X_train = X_train[outlier_mask_train]
    y_train = y_train[outlier_mask_train]

    # Apply the same outlier removal to the testing set
    z_scores_test = np.abs((X_test[num_features] - X_test[num_features].mean()) / X_test[num_features].std())
    outlier_mask_test = (z_scores_test < z_score_threshold).all(axis=1)

    X_test = X_test[outlier_mask_test]
    y_test = y_test[outlier_mask_test]

    # Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train[num_features])

    X_train[num_features] = imputer.transform(X_train[num_features])
    X_test[num_features] = imputer.transform(X_test[num_features])

    # Convert categorical columns with one-hot encoding using OneHotEncoder
    enc = OneHotEncoder()
    enc.fit(X_train[cat_features])

    X_train_cat = enc.transform(X_train[cat_features]).toarray()
    X_test_cat = enc.transform(X_test[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
    X_train = pd.concat([X_train[num_features + fl_features].reset_index(drop=True), pd.DataFrame(X_train_cat, columns=enc.get_feature_names_out())], axis=1)
    X_test = pd.concat([X_test[num_features + fl_features].reset_index(drop=True), pd.DataFrame(X_test_cat, columns=enc.get_feature_names_out())], axis=1)

    print(f"Features: \n {X_train.columns.tolist()}")

    # Train the linear regression model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Train the random forest regressor model
    rf_model = RandomForestRegressor(n_estimators=500, random_state=1)
    rf_model.fit(X_train, y_train)

    # Train XGboost
    XG_model = XGBRegressor()
    XG_model.fit(X_train, y_train)

    # Evaluate the models
    train_linear_score = r2_score(y_train, linear_model.predict(X_train))
    test_linear_score = r2_score(y_test, linear_model.predict(X_test))
    rmse_linear = np.sqrt(mean_squared_error(y_test, linear_model.predict(X_test)))

    train_rf_score = r2_score(y_train, rf_model.predict(X_train))
    test_rf_score = r2_score(y_test, rf_model.predict(X_test))
    rmse_rf = np.sqrt(mean_squared_error(y_test, rf_model.predict(X_test)))

    train_XG_score = r2_score(y_train, XG_model.predict(X_train))
    test_XG_score = r2_score(y_test, XG_model.predict(X_test))
    rmse_XG = np.sqrt(mean_squared_error(y_test, XG_model.predict(X_test)))

    print("Linear Regression Model:")
    print(f"Train R² score: {train_linear_score}")
    print(f"Test R² score: {test_linear_score}")
    print(f"RMSE: {rmse_linear}")

    print("\nRandom Forest Regressor Model:")
    print(f"Train R² score: {train_rf_score}")
    print(f"Test R² score: {test_rf_score}")
    print(f"RMSE: {rmse_rf}")

    print("\nXGboost Model:")
    print(f"Train R² score: {train_XG_score}")
    print(f"Test R² score: {test_XG_score}")
    print(f"RMSE: {rmse_XG}")

    # Save the models
    artifacts = {"linear_model": linear_model, "rf_model": rf_model, "XG_model": XG_model,"features": {"num_features": num_features, "fl_features": fl_features, "cat_features": cat_features}, "imputer": imputer, "enc": enc}
    joblib.dump(artifacts, "models/artifacts.joblib")

    # Predict on the test set
    y_pred_linear = linear_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)
    y_pred_XG = XG_model.predict(X_test)

    # Plot the predictions
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    axes[0].scatter(y_test, y_pred_linear, color='blue')
    axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    axes[0].set_title("Linear Regression: Actual vs Predicted Prices")
    axes[0].set_xlabel('Actual Prices')
    axes[0].set_ylabel('Predicted Prices')

    axes[1].scatter(y_test, y_pred_rf, color='blue')
    axes[1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    axes[1].set_title("Random Forest Regression: Actual vs Predicted Prices")
    axes[1].set_xlabel('Actual Prices')
    axes[1].set_ylabel('Predicted Prices')

    axes[2].scatter(y_test, y_pred_XG, color='blue')
    axes[2].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    axes[2].set_title("XGBoost Regression: Actual vs Predicted Prices")
    axes[2].set_xlabel('Actual Prices')
    axes[2].set_ylabel('Predicted Prices')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train()