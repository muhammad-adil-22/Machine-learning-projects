# House Price Prediction Using Machine Learning

## Project Overview

This project predicts residential house prices using machine learning regression models. It uses selected features from the house-sales dataset, applies preprocessing to numerical and categorical data, compares three regression algorithms, and selects XGBoost as the final model based on held-out validation performance.

The project also includes a Streamlit interface so a user can enter house information and receive a predicted price.

## Dataset

**Dataset link:**  
[Add dataset URL here]

The project dataset was supplied in CSV format and included:
- `train.csv` — training data, including the target column `SalePrice`.
- `test.csv` — separate data without `SalePrice`, intended for predictions.
- `sample_submission.csv` — example submission structure.

The validation metrics reported below were calculated using a held-out split from the labelled training data. They were **not** calculated on `test.csv`, because that file does not contain the actual house prices needed for evaluation.

## Features Used

The final model uses these 11 input features:

1. `LotArea`
2. `KitchenQual`
3. `GarageCars`
4. `TotalBsmtSF`
5. `BsmtCond`
6. `GrLivArea`
7. `Neighborhood`
8. `YearBuilt`
9. `OverallQual`
10. `FullBath`
11. `HalfBath`

The target variable is `SalePrice`.

## Workflow

1. **Load and inspect the data**  
   Load the CSV files with pandas, inspect the columns and data types, and separate the target (`SalePrice`) from the input features.

2. **Select relevant features**  
   Keep the 11 selected columns as model inputs. This keeps the project focused and ensures the same input columns are used during training and prediction.

3. **Preprocess the data**
   - Use `SimpleImputer` to handle missing values.
   - Use `OneHotEncoder(handle_unknown="ignore")` to convert categorical features into numerical columns.
   - Use `StandardScaler` to scale numerical values.
   - Fit preprocessing on training data, then use the fitted transformations on validation or new data. Do not fit the preprocessing again on the validation/test input.

4. **Train and compare models**  
   Train Linear Regression, Random Forest Regressor, and XGBoost Regressor using the prepared features.

5. **Evaluate the models**  
   Compare predictions with the actual prices in the held-out validation data using MAE, RMSLE, MSE, and R².

6. **Select the final model**  
   XGBoost achieved the strongest results among the three models in this experiment, so it was selected as the final model.

7. **Save and use the model**  
   Save the trained model and the preprocessing information needed to prepare future inputs consistently, using `joblib`.

8. **Use the Streamlit application**  
   The app accepts house-feature values, applies the same preprocessing used during training, and displays the model's predicted house price.

## Validation Results

| Model | MAE | RMSLE | MSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 21,313.08 | 0.177329 | 1,213,852,000 | 0.841747 |
| Random Forest Regressor | 18,045.41 | 0.162182 | 792,749,200 | 0.896647 |
| XGBoost Regressor | **17,776.39** | **0.157644** | **780,934,900** | **0.898188** |

*Values are rounded for readability. These are held-out validation results from the labelled training data.*

### Metric meanings

- **MAE (Mean Absolute Error):** average absolute difference between predicted and actual prices. Lower is better.
- **RMSLE (Root Mean Squared Logarithmic Error):** measures prediction error on a logarithmic scale, useful when the target values vary considerably. Lower is better.
- **MSE (Mean Squared Error):** average squared prediction error, which penalizes larger errors more heavily. Lower is better.
- **R² (Coefficient of Determination):** indicates how much variation in the target is explained by the model. A value closer to 1 is generally better.

## Running the Streamlit App

Open a terminal in the project directory, install the required packages if needed, and run:

```bash
py -m streamlit run app.py
```

If your Streamlit file has a different name, replace `app.py` with its actual filename.

Typical packages used in this project include:

```bash
pip install pandas numpy scikit-learn xgboost joblib streamlit
```

## Project Structure

A possible project folder structure is:

```text
House-Price-Prediction/
│
├── train.csv
├── test.csv
├── sample_submission.csv
├── app.py
├── README.md
└── model and preprocessing files
```

Keep the saved model and preprocessing files in the locations expected by `app.py`.

## Important Notes and Limitations

- The model is trained using only the 11 selected features; new inputs must provide those same features.
- Categorical values must be represented consistently with the categories expected by the app and preprocessing pipeline.
- The validation scores estimate performance on a held-out portion of the labelled training data. They do not guarantee the same performance on every future dataset.
- `test.csv` has no target labels, so its predictions cannot be evaluated unless the true prices become available.
- The model's predictions are estimates, not guaranteed market valuations.

## Tools and Libraries

- Python
- pandas and NumPy
- scikit-learn
- XGBoost
- joblib
- Streamlit

## Author

**Name:** Mian Muhammad Adil
