import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import OneHotEncoder


def train():
    """Trains a linear regression and random forest regressor model using Leave-One-Out Cross-Validation."""
    # Load the data
    data = pd.read_csv("data/properties.csv")

    # Define features to use
    num_features = ["nbr_frontages", "construction_year", "total_area_sqm", "surface_land_sqm", "nbr_bedrooms", "terrace_sqm", "garden_sqm", "primary_energy_consumption_sqm"]
    fl_features = ["fl_terrace", "fl_furnished", "fl_open_fire", "fl_terrace", "fl_garden", "fl_swimming_pool", "fl_floodzone", "fl_double_glazing"]
    cat_features = ["equipped_kitchen", "property_type", "subproperty_type", "region", "province", "locality", "state_building", "epc", "heating_type"]

    # Initialize lists to store evaluation scores
    linear_scores = []
    rf_scores = []

    # Initialize LOOCV
    loo = LeaveOneOut()

    # Iterate over each fold of LOOCV
    for train_index, test_index in loo.split(data):
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = X_train["price"], X_test["price"]
        X_train, X_test = X_train[num_features + fl_features + cat_features], X_test[num_features + fl_features + cat_features]

        # Skip evaluation if there is less than two samples in the test set
        if len(X_test) < 2:
            continue

        # Impute missing values using SimpleImputer
        imputer = SimpleImputer(strategy="mean")
        X_train[num_features] = imputer.fit_transform(X_train[num_features])
        X_test[num_features] = imputer.transform(X_test[num_features])

        # Convert categorical columns with one-hot encoding using OneHotEncoder
        enc = OneHotEncoder()
        X_train_cat = enc.fit_transform(X_train[cat_features]).toarray()
        X_test_cat = enc.transform(X_test[cat_features]).toarray()

        # Combine the numerical and one-hot encoded categorical columns
        X_train = pd.concat([X_train[num_features + fl_features].reset_index(drop=True), pd.DataFrame(X_train_cat, columns=enc.get_feature_names_out())], axis=1)
        X_test = pd.concat([X_test[num_features + fl_features].reset_index(drop=True), pd.DataFrame(X_test_cat, columns=enc.get_feature_names_out())], axis=1)

        # Train the linear regression model
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)

        # Train the random forest regressor model
        rf_model = RandomForestRegressor(n_estimators=10, random_state=1)
        rf_model.fit(X_train, y_train)

        # Evaluate the models
        linear_score = r2_score(y_test, linear_model.predict(X_test))
        rf_score = r2_score(y_test, rf_model.predict(X_test))

        # Store evaluation scores
        linear_scores.append(linear_score)
        rf_scores.append(rf_score)

    # Calculate mean scores
    mean_linear_score = sum(linear_scores) / len(linear_scores) if linear_scores else 0.0
    mean_rf_score = sum(rf_scores) / len(rf_scores) if rf_scores else 0.0

    print("Mean Linear Regression Model R² score:", mean_linear_score)
    print("Mean Random Forest Regressor Model R² score:", mean_rf_score)



   import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

def train():
    """Trains a linear regression and random forest regressor model on the full dataset and stores output."""
    # Load the data
    data = pd.read_csv("data/properties.csv")

    # Define features to use
    num_features = ["nbr_frontages", "construction_year", "total_area_sqm", "surface_land_sqm", "nbr_bedrooms", "terrace_sqm", "garden_sqm", "primary_energy_consumption_sqm"]
    fl_features = ["fl_terrace", "fl_furnished", "fl_open_fire", "fl_terrace", "fl_garden", "fl_swimming_pool", "fl_floodzone", "fl_double_glazing"]
    cat_features = ["equipped_kitchen", "property_type", "subproperty_type", "region", "province", "locality", "state_building", "epc", "heating_type"]

    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=505)

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
    rf_model = RandomForestRegressor(n_estimators=10, random_state=1)
    rf_model.fit(X_train, y_train)

    #Train XGboost
    XG_model = XGBRegressor()
    XG_model.fit(X_train, y_train)

    # Evaluate the models
    train_linear_score = r2_score(y_train, linear_model.predict(X_train))
    test_linear_score = r2_score(y_test, linear_model.predict(X_test))

    train_rf_score = r2_score(y_train, rf_model.predict(X_train))
    test_rf_score = r2_score(y_test, rf_model.predict(X_test))

    train_XG_score = r2_score(y_train, XG_model.predict(X_train))
    test_XG_score = r2_score(y_test, XG_model.predict(X_test))

    print("Linear Regression Model:")
    print(f"Train R² score: {train_linear_score}")
    print(f"Test R² score: {test_linear_score}")

    print("\nRandom Forest Regressor Model:")
    print(f"Train R² score: {train_rf_score}")
    print(f"Test R² score: {test_rf_score}")

    print("\nXGboost Model:")
    print(f"Train R² score: {train_XG_score}")
    print(f"Test R² score: {test_XG_score}")


    # Save the models
    artifacts = {"linear_model": linear_model, "rf_model": rf_model, "XG_model": XG_model,"features": {"num_features": num_features, "fl_features": fl_features, "cat_features": cat_features}, "imputer": imputer, "enc": enc}
    joblib.dump(artifacts, "models/artifacts.joblib")

    # Predict on the test set
    y_pred_linear = linear_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)
    y_pred_XG = XG_model.predict(X_test)

    # Plot the predictions
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

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
    axes[2].set_title("XGBoost: Actual vs Predicted Prices")
    axes[2].set_xlabel('Actual Prices')
    axes[2].set_ylabel('Predicted Prices')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train()



if __name__ == "__main__":
    train()
