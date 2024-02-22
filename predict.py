import click
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

@click.command()
@click.option("-i", "--input-dataset", help="Path to input .csv dataset", required=True)
@click.option(
    "-o",
    "--output-prefix",
    default="output/predictions",
    help="Prefix for the full path where to store predictions (e.g., output/predictions)",
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
    ensemble_model = artifacts["ensemble_model"]

    # Extract the relevant features from the input data
    data_features = data[num_features + fl_features + cat_features]

    # Apply imputer on numerical features
    data_features[num_features] = imputer.transform(data_features[num_features])

    # Apply encoder on categorical features
    data_cat = enc.transform(data_features[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
    data_features = pd.concat(
        [
            data_features[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(data_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    # Make predictions using the ensemble model
    ensemble_predictions = ensemble_model.predict(data_features)

    # Save predictions to a CSV file
    ensemble_output_path = f"{output_prefix}_ensemble.csv"
    pd.DataFrame({"predictions": ensemble_predictions}).to_csv(ensemble_output_path, index=False)

    # Print success messages
    click.echo(click.style("Predictions generated successfully!", fg="green"))
    click.echo(f"Saved to  {ensemble_output_path}")
    click.echo(
        f"Nbr. observations: {data.shape[0]} | Nbr. predictions: {ensemble_predictions.shape[0]}"
    )

if __name__ == "__main__":
    predict()
