# House Price Prediction Model Using Linear Regression: Notes

## 1. Objective
- **Goal**: Build a model to predict house prices (continuous numerical value) based on features like size, number of bedrooms, location, etc.
- **Algorithm**: Linear regression, which assumes a linear relationship between features and the target variable (price).
- **Formula**: \( y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n + \epsilon \)
  - \( y \): House price (target)
  - \( x_1, x_2, \dots, x_n \): Features (e.g., square footage, bedrooms)
  - \( \beta_0 \): Intercept
  - \( \beta_1, \beta_2, \dots, \beta_n \): Coefficients
  - \( \epsilon \): Error term

## 2. Dataset
- **Used**: California Housing dataset from scikit-learn.
- **Features**:
  - `MedInc`: Median income (tens of thousands of dollars)
  - `HouseAge`: Median house age (years)
  - `AveRooms`: Average number of rooms per dwelling
  - `AveBedrms`: Average number of bedrooms per dwelling
  - `Population`: Block group population
  - `AveOccup`: Average household members
  - `Latitude`: Latitude (32–42 for California)
  - `Longitude`: Longitude (-124 to -114 for California)
- **Target**: `price` (house price in $100,000s)
- **Exploration**:
  - Check for missing values: None in this dataset.
  - Use `df.describe()` for data ranges and `df.corr()` for feature correlations.
  - Visualize correlations with a heatmap using seaborn.

## 3. Steps to Build the Model

### 3.1 Environment Setup
- **Libraries**:
  - `pandas`: Data manipulation
  - `scikit-learn`: Linear regression, preprocessing, evaluation
  - `matplotlib`/`seaborn`: Visualization
  - `numpy`: Numerical operations
- Install: `pip install pandas scikit-learn matplotlib seaborn numpy`

### 3.2 Data Preprocessing
- **Handle Missing Values**: None in this dataset, but check with `df.isnull().sum()`.
- **Outliers**: Cap extreme values at the 99th percentile to prevent skewing.
- **Standardization**: Use `StandardScaler` to scale features to mean=0, std=1.
- **Feature Selection**: Check correlations with `price` to identify important features (e.g., `MedInc` often has high correlation).
- **Code Example**:
  ```python
  from sklearn.preprocessing import StandardScaler
  import seaborn as sns
  import matplotlib.pyplot as plt

  # Correlation analysis
  print(df.corr()['price'].sort_values())
  plt.figure(figsize=(10, 8))
  sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
  plt.show()

  # Cap outliers
  for col in df.columns:
      if col != 'price':
          df[col] = df[col].clip(upper=df[col].quantile(0.99))

  # Standardize features
  X = df.drop('price', axis=1)
  y = df['price']
  scaler = StandardScaler()
  X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
  ```

### 3.3 Train-Test Split
- Split data: 80% training, 20% testing.
- Use `random_state=42` for reproducibility.
- **Code**:
  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
  ```

### 3.4 Train the Model
- Use `LinearRegression` to fit the model.
- Output coefficients to understand feature importance.
- **Code**:
  ```python
  from sklearn.linear_model import LinearRegression
  model = LinearRegression()
  model.fit(X_train, y_train)
  for feature, coef in zip(X.columns, model.coef_):
      print(f"{feature}: {coef:.4f}")
  print(f"Intercept: {model.intercept_:.4f}")
  ```

### 3.5 Evaluate the Model
- **Metrics**:
  - Mean Squared Error (MSE): Lower is better.
  - R² Score: Higher is better (0–1, measures variance explained).
- Visualize actual vs. predicted prices with a scatter plot.
- **Code**:
  ```python
  from sklearn.metrics import mean_squared_error, r2_score
  import matplotlib.pyplot as plt
  y_train_pred = model.predict(X_train)
  y_test_pred = model.predict(X_test)
  print(f"Training MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
  print(f"Testing MSE: {mean_squared_error(y_test, y_test_pred):.4f}")
  print(f"Training R²: {r2_score(y_train, y_train_pred):.4f}")
  print(f"Testing R²: {r2_score(y_test, y_test_pred):.4f}")
  plt.scatter(y_test, y_test_pred, alpha=0.5)
  plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
  plt.xlabel("Actual Price")
  plt.ylabel("Predicted Price")
  plt.title("Actual vs Predicted House Prices")
  plt.show()
  ```

### 3.6 Make Predictions
- Predict prices for new houses using scaled features.
- Ensure input data matches training data’s structure.
- Convert predictions to dollars (multiply by 100,000 for California dataset).
- **Code**:
  ```python
  import pandas as pd
  new_house = pd.DataFrame([[8.0, 20, 6.0, 1.0, 1000, 3.0, 34.0, -118.0]], 
                           columns=X.columns)
  new_house_scaled = pd.DataFrame(scaler.transform(new_house), columns=X.columns)
  predicted_price = model.predict(new_house_scaled)
  print(f"Predicted price: ${predicted_price[0] * 100000:.2f}")
  ```

## 4. Issues Encountered and Fixes

### 4.1 Negative Predicted Price (`$-5194325.15`)
- **Problem**: Unrealistic negative price due to invalid input values for `Latitude` (3.0) and `Longitude` (50), which were outside the California dataset’s range (Latitude: 32–42, Longitude: -124 to -114).
- **Cause**: Extrapolation beyond training data range led to unreliable predictions, amplified by large negative coefficients.
- **Fix**:
  - Use realistic values (e.g., `Latitude=34.0`, `Longitude=-118.0` for Los Angeles).
  - Clip predictions to ensure non-negative prices: `np.clip(predicted_price, 0, None)`.
  - Optional: Log-transform target variable (`np.log1p(y)`) during training to stabilize predictions.
- **Code Fix**:
  ```python
  import numpy as np
  new_house = pd.DataFrame([[8.0, 20, 6.0, 1.0, 1000, 3.0, 34.0, -118.0]], 
                           columns=X.columns)
  new_house_scaled = pd.DataFrame(scaler.transform(new_house), columns=X.columns)
  predicted_price = model.predict(new_house_scaled)
  predicted_price = np.clip(predicted_price, 0, None)
  print(f"Predicted price: ${predicted_price[0] * 100000:.2f}")
  ```

### 4.2 Scikit-learn Warning
- **Warning**: `X does not have valid feature names, but LinearRegression was fitted with feature names`.
- **Cause**: Input for prediction was a NumPy array (from `scaler.transform`), but the model was trained on a DataFrame with column names.
- **Fix**: Convert scaled input back to a DataFrame with the same column names.
- **Code Fix**:
  ```python
  new_house_scaled = pd.DataFrame(scaler.transform(new_house), columns=X.columns)
  ```

## 5. Key Learning Points
- **Linear Regression Assumptions**:
  - Linearity between features and target.
  - Independence of features.
  - Homoscedasticity (constant variance of residuals).
  - Normality of residuals.
- **Feature Importance**: Coefficients indicate which features (e.g., `MedInc`) drive predictions.
- **Preprocessing**:
  - Standardize features to ensure similar scales.
  - Handle outliers to prevent model skew.
- **Evaluation**:
  - Compare train vs. test MSE/R² to detect overfitting.
  - R² of 0.6–0.8 is common for real-world datasets.
- **Prediction Tips**:
  - Ensure input features are within training data ranges.
  - Match input structure (e.g., column names) to training data.
  - Clip or transform predictions to avoid unrealistic values.

## 6. Potential Improvements
- **Feature Engineering**:
  - Create new features (e.g., rooms per square foot).
  - Use polynomial features for non-linear relationships:
    ```python
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    polyreg = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    polyreg.fit(X_train, y_train)
    ```
- **Feature Selection**: Drop low-correlation or highly correlated features (multicollinearity).
- **Alternative Models**: Try Random Forest or Gradient Boosting for better performance.
- **Residual Analysis**: Plot residuals to check assumptions.

## 7. Next Steps
- **Custom Dataset**: If using a different dataset, verify feature names, target units, and ranges.
- **Visualization**: Create scatter plots (e.g., `MedInc` vs. `price`) to explore relationships.
- **Model Debugging**: Check coefficients, residuals, or test performance for issues.
- **Advanced Models**: Experiment with non-linear models if linear regression underperforms.