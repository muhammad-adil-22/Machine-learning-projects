import os
import sys

_DLL_DIRECTORY_HANDLES = []
if os.name == "nt":
    site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
    dll_directories = [
        sys.prefix,
        os.path.join(sys.prefix, "Library", "bin"),
        os.path.join(sys.prefix, "DLLs"),
        os.path.join(site_packages, "numpy.libs"),
        os.path.join(site_packages, "scipy.libs"),
        os.path.join(site_packages, "xgboost", "lib"),
    ]
    _DLL_DIRECTORY_HANDLES = [
        os.add_dll_directory(directory)
        for directory in dll_directories
        if os.path.isdir(directory)
    ]

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction App")
st.write("Enter the property details below to estimate its sale price.")

# -------------------- FILE PATHS --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

required_files = [
    "xgboost_model.joblib",
    "scaler.joblib",
    "encoder.joblib",
    "imputer.joblib",
    "categorical_columns.joblib"
]

missing_files = [
    filename for filename in required_files
    if not os.path.isfile(os.path.join(BASE_DIR, filename))
]

if missing_files:
    st.error("These required files are missing from your project folder:")
    for filename in missing_files:
        st.write(f"- {filename}")
    st.info("Keep all five files in the same folder as App code.py.")
    st.stop()

# -------------------- LOAD MODEL AND PREPROCESSORS --------------------
try:
    model = joblib.load(os.path.join(BASE_DIR, "xgboost_model.joblib"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.joblib"))
    encoder = joblib.load(os.path.join(BASE_DIR, "encoder.joblib"))
    imputer = joblib.load(os.path.join(BASE_DIR, "imputer.joblib"))
    cat_cols = joblib.load(
        os.path.join(BASE_DIR, "categorical_columns.joblib")
    )
except Exception as e:
    st.error(f"Could not load the saved model files: {e}")
    st.exception(e)
    st.code(f"Python interpreter: {sys.executable}\nNumPy location: {np.__file__}")
    st.stop()

# These columns must match the order used when training your model.
numeric_cols = [
    "LotArea",
    "GarageCars",
    "TotalBsmtSF",
    "GrLivArea",
    "YearBuilt",
    "OverallQual",
    "FullBath",
    "HalfBath"
]

if list(cat_cols) != [
    "KitchenQual", "BsmtCond", "Neighborhood"
]:
    st.error(
        "The saved categorical column order does not match "
        "the expected training setup."
    )
    st.stop()

# -------------------- USER INPUTS --------------------
st.subheader("Property Details")

with st.form("house_details"):
    col1, col2 = st.columns(2)

    with col1:
        lot_area = st.number_input(
            "Lot Area (sq ft)",
            min_value=1,
            value=8450,
            step=100
        )

        garage_cars = st.number_input(
            "Garage Capacity (cars)",
            min_value=0,
            max_value=10,
            value=2,
            step=1
        )

        basement_area = st.number_input(
            "Basement Area (sq ft)",
            min_value=0,
            value=856,
            step=50
        )

        living_area = st.number_input(
            "Living Area (sq ft)",
            min_value=1,
            value=1710,
            step=50
        )

        year_built = st.number_input(
            "Year Built",
            min_value=1800,
            max_value=2100,
            value=2000,
            step=1
        )

    with col2:
        overall_quality = st.number_input(
            "Overall Quality (1–10)",
            min_value=1,
            max_value=10,
            value=6,
            step=1
        )

        full_bath = st.number_input(
            "Full Bathrooms",
            min_value=0,
            max_value=10,
            value=2,
            step=1
        )

        half_bath = st.number_input(
            "Half Bathrooms",
            min_value=0,
            max_value=10,
            value=1,
            step=1
        )

    st.subheader("Quality and Location")

    # Use the actual categories learned by the saved encoder.
    categories = encoder.categories_

    def display_category(value):
        if pd.isna(value):
            return "Missing"
        return str(value)

    kitchen_qual = st.selectbox(
        "Kitchen Quality",
        options=list(categories[0]),
        format_func=display_category
    )

    basement_cond = st.selectbox(
        "Basement Condition",
        options=list(categories[1]),
        format_func=display_category
    )

    neighborhood = st.selectbox(
        "Neighborhood",
        options=list(categories[2]),
        format_func=display_category
    )

    predict_button = st.form_submit_button(
        "Predict House Price",
        use_container_width=True
    )

# -------------------- PREDICTION --------------------
if predict_button:
    try:
        # Keep the original feature names and order.
        input_data = pd.DataFrame([{
            "LotArea": lot_area,
            "KitchenQual": kitchen_qual,
            "GarageCars": garage_cars,
            "TotalBsmtSF": basement_area,
            "BsmtCond": basement_cond,
            "GrLivArea": living_area,
            "Neighborhood": neighborhood,
            "YearBuilt": year_built,
            "OverallQual": overall_quality,
            "FullBath": full_bath,
            "HalfBath": half_bath
        }])

        # Separate numerical and categorical inputs.
        numeric_data = input_data[numeric_cols]
        categorical_data = input_data[cat_cols]

        # Apply the same preprocessing used during training.
        numeric_imputed = imputer.transform(numeric_data)
        numeric_scaled = scaler.transform(numeric_imputed)
        categorical_encoded = encoder.transform(categorical_data)

        # Combine features in the same order used for model training.
        processed_data = np.hstack([
            numeric_scaled,
            categorical_encoded
        ])

        # Check that the model expects this number of features.
        expected_features = getattr(model, "n_features_in_", None)

        if (
            expected_features is not None
            and processed_data.shape[1] != expected_features
        ):
            st.error(
                f"Feature mismatch: the model expects {expected_features} "
                f"features, but preprocessing produced "
                f"{processed_data.shape[1]}. Check that your saved model "
                "and preprocessing files came from the same training run."
            )
            st.stop()

        prediction = float(model.predict(processed_data)[0])

        if not np.isfinite(prediction):
            st.error("The model returned an invalid prediction.")
            st.stop()

        st.success("Prediction completed!")

        st.metric(
            label="Estimated House Sale Price",
            value=f"${prediction:,.2f}"
        )

        st.caption(
            "This is a model estimate, not a guaranteed selling price."
        )

    except Exception as e:
        st.error(f"Could not load the saved model files: {e}")
        st.exception(e)
        st.code(str(e))
        st.info(
            "Make sure the model, scaler, encoder, imputer, and "
            "categorical-columns files were saved from the same "
            "training run as your XGBoost model."
        )