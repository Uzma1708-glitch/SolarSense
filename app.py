import os
from flask import Flask, render_template

from load_data import load_data

app = Flask(__name__)

CHARTS_DIR = os.path.join(app.static_folder, "charts")


@app.route("/")
def index():

    # Load dataset
    df = load_data()

    # Dataset information
    n_rows = df.shape[0]
    n_cols = df.shape[1]

    columns = list(df.columns)

    # First 10 rows
    preview = df.head(10).to_dict(orient="records")

    # Missing values
    missing_counts = {
        col: int(df[col].isna().sum())
        for col in df.columns
    }

    return render_template(
        "index.html",
        n_rows=n_rows,
        n_cols=n_cols,
        columns=columns,
        preview=preview,
        missing_counts=missing_counts
    )


@app.route("/eda")
def eda():

    charts = []

    if os.path.exists(CHARTS_DIR):
        charts = [
            file for file in os.listdir(CHARTS_DIR)
            if file.endswith(".png")
        ]

    return render_template(
        "eda.html",
        charts=charts
    )


if __name__ == "__main__":
    app.run(debug=True)