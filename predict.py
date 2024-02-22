import click
import joblib
import pandas as pd

@click.command()
@click.option("-i", "--input-dataset", help="path to input .csv dataset", required=True)
@click.option(
    "-o",
    "--output-prefix",
    default="output/predictions",
    help="prefix for the full path where to store predictions (e.g., output/predictions)",
    required=True,
)
def predict(input_dataset, output_prefix):
    """Predicts house prices from 'input_dataset' using trained models and stores predictions."""
    # Load the data
    data = pd.read_csv(input_dataset)

    # Load the model artifacts
    artifacts = joblib.load("models/artifacts.joblib")
    num_features = artifacts["features"]["num_features"]
    fl_features = artifacts["features"]["fl_features"]
    cat_features = artifacts["features"]["cat_features"]
    imputer = artifacts["imputer"]
    enc = artifacts["enc"]
    linear_model = artifacts["linear_model"]
    rf_model = artifacts["rf_model"]
    XG_model = artifacts["XG_model"]

    # Extract the relevant features from the input data
    data_features = data[num_features + fl_features + cat_features]

    # Apply imputer and encoder on data
    data_features[num_features] = imputer.transform(data_features[num_features])
    data_cat = enc.transform(data_features[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
    data_features = pd.concat(
        [
            data_features[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(data_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    # Make predictions using the linear regression model
    linear_predictions = linear_model.predict(data_features)

    # Make predictions using the random forest regressor model
    rf_predictions = rf_model.predict(data_features)

    # Make predictions using the XGBoost model
    XG_predictions = XG_model.predict(data_features)

    # Save predictions to separate CSV files
    linear_output_path = f"{output_prefix}_linear.csv"
    rf_output_path = f"{output_prefix}_rf.csv"
    XG_output_path = f"{output_prefix}_XG.csv"
    pd.DataFrame({"predictions": linear_predictions}).to_csv(linear_output_path, index=False)
    pd.DataFrame({"predictions": rf_predictions}).to_csv(rf_output_path, index=False)
    pd.DataFrame({"predictions": XG_predictions}).to_csv(XG_output_path, index=False)


    # Print success messages
    click.echo(click.style("Predictions generated successfully!", fg="green"))
    click.echo(f"Saved to {linear_output_path, rf_output_path, XG_output_path}")
    click.echo(
        f"Nbr. observations: {data.shape[0]} | Nbr. predictions: {linear_predictions.shape[0], rf_predictions.shape[0], XG_predictions.shape[0] }"
    )
    ### -------------------------------------------------- ###

"""         # Print success messages
    print(f"Linear regression predictions saved to: {linear_output_path}")
    print(f"Random forest regression predictions saved to: {rf_output_path}")p

        ### -------- DO NOT TOUCH THE FOLLOWING LINES -------- ###
    # Save the predictions to a CSV file (in order of data input! """

if __name__ == "__main__":
    predict()



