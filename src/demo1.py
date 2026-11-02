# House Price Prediction (Hyderabad)
# Simple, Multiple and Polynomial Regression

# Importing the Libraries

import numpy as np
import pandas as pd
import seaborn as sns
import os
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from settings import RAW_FILE_PATH
from settings import PROCEED_FILE_PATH


# Loading Dataset

df = pd.read_csv(RAW_FILE_PATH)
print(df.head())


# ------------------------------------------------------------
# PART 1: Data Preparation (pandas)
# ------------------------------------------------------------

# 1> Basic operation

print("Shape is:", df.shape)
print()
print("Is empty:", df.empty)
print()
print("Columns Names:", df.columns)
print()
print("Data Type per columns")
print(df.dtypes)
print()
print(df.info())
print()
print(df.describe())
print()


# 2> Checking missing values and duplicates

print("Columns with missing values:", list(df.columns[df.isna().any()]))
print()

summary = pd.DataFrame({
    "Missing_Value": df.isna().sum(),
    "Percentage_Missing_Value": df.isna().sum() / len(df) * 100,
    "Uniques_Values": df.nunique()
})
print(summary)
print()
print("Duplicate rows:", df.duplicated().sum())


# 3> Drop duplicate records

print("Before Drop duplicates:", len(df))
df = df.drop_duplicates()
print("After Drop duplicates:", len(df))
print("Duplicate rows now:", df.duplicated().sum())
print()


# 4> Fix Area_sqft column (text like "1200 sqft" -> number 1200)

df["Area_sqft"] = df["Area_sqft"].astype(str).str.replace(r"[^0-9.]", "", regex=True).str.strip()
print(df["Area_sqft"].dtype)

df["Area_sqft"] = pd.to_numeric(df["Area_sqft"])
print(df["Area_sqft"].dtype)
print()


# 5> Filling missing values

print(df.isna().sum())
print()

# numeric columns -> median
cols = ["Bathrooms", "Age_years"]

for col in cols:
    df[col] = df[col].fillna(df[col].median())

# categorical column -> mode
df["Furnishing"] = df["Furnishing"].fillna(df["Furnishing"].mode()[0])

print("After filling missing values")
print(df.isna().sum())
print()


# 6> Handling outliers (IQR method) for Price_Lakhs and Area_sqft

print("Statistical Summary")
print(df.describe())
print()

# boxplots BEFORE removing outliers
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].boxplot(df["Area_sqft"].values)
ax[0].set_title("Area_sqft (Before)")
ax[1].boxplot(df["Price_Lakhs"].values)
ax[1].set_title("Price_Lakhs (Before)")
plt.show()

print("Rows before handling outliers:", len(df))

cols = ["Price_Lakhs", "Area_sqft"]

for col in cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outlier = (df[col] < lower) | (df[col] > upper)
    print(col, "-> lower:", round(lower, 2), "upper:", round(upper, 2))
    print(col, "-> outliers:", outlier.sum())

    df = df[~outlier]
    print("Rows after handling", col, "outliers:", len(df))
    print()

# boxplots AFTER removing outliers
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].boxplot(df["Area_sqft"].values)
ax[0].set_title("Area_sqft (After)")
ax[1].boxplot(df["Price_Lakhs"].values)
ax[1].set_title("Price_Lakhs (After)")
plt.show()

print(df.isna().sum())
print()


# 7> Save cleaned data

df.to_csv(PROCEED_FILE_PATH, index=False)

df1 = pd.read_csv(PROCEED_FILE_PATH)
print("Cleaned data shape:", df1.shape)
print(df1.head())
print()


# ------------------------------------------------------------
# PART 2: Exploratory Data Analysis (Statistics)
# ------------------------------------------------------------

print("Mean:", df1["Price_Lakhs"].mean())
print("Median:", df1["Price_Lakhs"].median())
print("Standard Deviation:", df1["Price_Lakhs"].std())
print("Min:", df1["Price_Lakhs"].min())
print("Max:", df1["Price_Lakhs"].max())
print()

# average price by location
result = df1.groupby("Location")["Price_Lakhs"].mean().sort_values(ascending=False)
print(result)
print()

# average price by furnishing
print(df1.groupby("Furnishing")["Price_Lakhs"].mean().sort_values(ascending=False))
print()

# correlation
corr = df1.corr(numeric_only=True)
print(corr)
print()
print("Correlation with price:")
print(corr["Price_Lakhs"].sort_values(ascending=False))

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# scatter plot
plt.scatter(df1["Area_sqft"], df1["Price_Lakhs"])
plt.xlabel("Area_sqft")
plt.ylabel("Price_Lakhs")
plt.title("Area_sqft vs Price_Lakhs")
plt.show()


# ------------------------------------------------------------
# PART 3: Build the Models
# ------------------------------------------------------------

# function to calculate all metrics (used for the comparison table)

def evaluate(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "Model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }


# Model 1: Simple Linear Regression (only Area_sqft)

X = df1[["Area_sqft"]].values
y = df1["Price_Lakhs"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train:", y_train.shape, "y_test:", y_test.shape)
print()

model = LinearRegression()
model.fit(X_train, y_train)

print("MODEL 1: Simple Linear Regression")
print("Train R2 score is :", model.score(X_train, y_train))
print("Test R2 score is  :", model.score(X_test, y_test))
print("Slope (Coefficient):", model.coef_[0])
print("Intercept is:", model.intercept_)
print()

pred1 = model.predict(X_test)
y_test_simple = y_test      # keep for the comparison table

# plot the regression line
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test, pred1, "r.", label="Predicted")
plt.xlabel("Area_sqft")
plt.ylabel("Price_Lakhs")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()


# Model 2: Multiple Linear Regression (all features)

df1 = df1.drop("House_ID", axis=1)

df1 = pd.get_dummies(df1, columns=["Location", "Furnishing"], drop_first=True, dtype="int")
print(df1.dtypes)
print()

y = df1["Price_Lakhs"].values
X = df1.drop("Price_Lakhs", axis=1).values
feature_cols = df1.drop("Price_Lakhs", axis=1).columns

# split FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# then make the model and fit
model2 = LinearRegression()
model2.fit(X_train, y_train)

print("MODEL 2: Multiple Linear Regression")
print("Train R2 score is :", model2.score(X_train, y_train))
print("Test R2 score is  :", model2.score(X_test, y_test))
print("Intercept is:", model2.intercept_)
print()

# which feature is important?
coef_table = pd.DataFrame({"Feature": feature_cols, "Coefficient": model2.coef_})
print(coef_table.sort_values("Coefficient", ascending=False))
print()

pred2 = model2.predict(X_test)


# Model 3: Polynomial Regression (degree 2)

a = PolynomialFeatures(degree=2)

X_poly = a.fit_transform(X_train)      # train data -> fit_transform
X_test_poly = a.transform(X_test)      # test data  -> only transform

model3 = LinearRegression()
model3.fit(X_poly, y_train)

print("MODEL 3: Polynomial Regression (degree 2)")
print("R2 for Training:", model3.score(X_poly, y_train))
print("R2 for Testing :", model3.score(X_test_poly, y_test))
print()

pred3 = model3.predict(X_test_poly)


# Model 4: Polynomial Regression (degree 3) -> check overfitting

b = PolynomialFeatures(degree=3)

X_poly3 = b.fit_transform(X_train)
X_test_poly3 = b.transform(X_test)

model4 = LinearRegression()
model4.fit(X_poly3, y_train)

print("MODEL 4: Polynomial Regression (degree 3)")
print("R2 for Training:", model4.score(X_poly3, y_train))
print("R2 for Testing :", model4.score(X_test_poly3, y_test))
print()

pred4 = model4.predict(X_test_poly3)


# ------------------------------------------------------------
# PART 4: Compare and Conclude
# ------------------------------------------------------------

# the split uses the same random_state, so y_test is the same for all models

results = pd.DataFrame([
    evaluate("Simple Linear", y_test_simple, pred1),
    evaluate("Multiple Linear", y_test, pred2),
    evaluate("Polynomial deg 2", y_test, pred3),
    evaluate("Polynomial deg 3", y_test, pred4),
])

print("MODEL COMPARISON TABLE")
print(results.round(3).to_string(index=False))
print()

best = results.loc[results["R2"].idxmax(), "Model"]
print("Best model (highest test R2):", best)
print()

# actual vs predicted for the polynomial degree 2 model
plt.scatter(y_test, pred3, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Price (Lakhs)")
plt.ylabel("Predicted Price (Lakhs)")
plt.title("Actual vs Predicted (Polynomial degree 2)")
plt.show()


# Prediction function (uses Polynomial degree 2 model)

def predict_price(area, bedrooms, bathrooms, age, parking, location, furnishing):
    row = pd.DataFrame(0, index=[0], columns=feature_cols)

    row["Area_sqft"] = area
    row["Bedrooms"] = bedrooms
    row["Bathrooms"] = bathrooms
    row["Age_years"] = age
    row["Parking"] = parking

    # Banjara Hills and Fully-Furnished have no column (drop_first=True),
    # so for them all the dummy columns stay 0
    if "Location_" + location in row.columns:
        row["Location_" + location] = 1
    if "Furnishing_" + furnishing in row.columns:
        row["Furnishing_" + furnishing] = 1

    return model3.predict(a.transform(row.values))[0]


print("1500 sqft, Gachibowli  :", round(predict_price(1500, 3, 2, 5, 1, "Gachibowli", "Semi-Furnished"), 2), "lakhs")
print("2500 sqft, Jubilee Hills:", round(predict_price(2500, 4, 4, 2, 2, "Jubilee Hills", "Fully-Furnished"), 2), "lakhs")
print("900 sqft, Uppal         :", round(predict_price(900, 2, 1, 10, 0, "Uppal", "Unfurnished"), 2), "lakhs")



# ------------------------------------------------------------
# PART 5: Save Model for Streamlit App
# ------------------------------------------------------------

# Get location and furnishing categories from cleaned data
clean_df = pd.read_csv(PROCEED_FILE_PATH)

locations = sorted(clean_df["Location"].dropna().unique().tolist())
furnishings = sorted(clean_df["Furnishing"].dropna().unique().tolist())

# Find minimum and maximum area
area_min = clean_df["Area_sqft"].min()
area_max = clean_df["Area_sqft"].max()

# Average price by location
avg_price_by_location = (
    clean_df.groupby("Location")["Price_Lakhs"]
    .mean()
    .reset_index(name="Average_Price_Lakhs")
)

# Find the project folder and save the model inside src
script_dir = os.path.dirname(os.path.abspath(__file__))

if os.path.basename(script_dir).lower() == "src":
    model_file = os.path.join(script_dir, "house_model.pkl")
else:
    model_file = os.path.join(script_dir, "src", "house_model.pkl")

# Store the trained model and required information
model_data = {
    "model": model3,
    "poly": a,
    "feature_cols": list(feature_cols),
    "locations": locations,
    "furnishings": furnishings,
    "area_min": area_min,
    "area_max": area_max,
    "results": results,
    "avg_price_by_location": avg_price_by_location
}

# Save as a pickle file
with open(model_file, "wb") as f:
    pickle.dump(model_data, f)

print("\nModel saved successfully!")
print("Saved file:", model_file)

