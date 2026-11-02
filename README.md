Hyderabad House Price Prediction

A machine learning project that predicts the price of a house in Hyderabad (in Lakhs) from its details, and a Streamlit web app to use the model.

Problem Statement

Buying or selling a house is a big decision, and it is hard to know a fair price. The price depends on many things together: area, number of rooms, age of the house, parking, location and furnishing. Checking all of them by hand is slow and often wrong.

Goal: Build a machine learning model that takes the details of a house in Hyderabad and predicts its price in Lakhs, and make it easy to use through a simple web app.

Objectives
Clean the raw house data (duplicates, wrong formats, missing values and outliers).
Study the data with EDA to understand what affects the price.
Train and compare four regression models:
Simple Linear Regression
Multiple Linear Regression
Polynomial Regression (degree 2)
Polynomial Regression (degree 3)
Choose the best model using test data, and check for overfitting.
Save the best model with Pickle (pickling).
Build a Streamlit app that loads the saved model (unpickling) and predicts the price.
Dataset

File: data/raw/hyderabad_house_prices.csv

Column	Description
House_ID	Unique ID of the house (dropped before training)
Area_sqft	Area of the house in square feet
Bedrooms	Number of bedrooms
Bathrooms	Number of bathrooms
Age_years	Age of the house in years
Parking	Number of parking spots
Location	Area in Hyderabad (10 locations, for example Banjara Hills, Gachibowli, Jubilee Hills, Uppal)
Furnishing	Fully-Furnished, Semi-Furnished or Unfurnished
Price_Lakhs	Target: price of the house in Lakhs

After cleaning, the cleaned data is saved in data/procced/cleaned.csv.

Project Workflow
1. Data Cleaning (Pandas)
Checked shape, data types, missing values and duplicate rows.
Removed duplicate records.
Fixed Area_sqft: text like "1200 sqft" was converted to the number 1200.
Filled missing values: median for numeric columns (Bathrooms, Age_years) and mode for the categorical column (Furnishing).
Removed outliers in Price_Lakhs and Area_sqft using the IQR method, and checked with boxplots before and after.
2. Exploratory Data Analysis (EDA)
Mean, median, standard deviation, minimum and maximum of the price.
Average price by location and by furnishing.
Correlation table and correlation heatmap (Seaborn).
Scatter plot of Area_sqft vs Price_Lakhs (Matplotlib).
3. Feature Preparation
Dropped House_ID.
Applied one-hot encoding (get_dummies, drop_first=True) on Location and Furnishing.
Split the data into train and test sets (80% / 20%, random_state=42).
4. Model Building (Scikit-learn)
Model	Features used
Simple Linear Regression	Area_sqft only
Multiple Linear Regression	All features
Polynomial Regression (degree 2)	All features
Polynomial Regression (degree 3)	All features (to check overfitting)
5. Model Evaluation

All models were tested on the same test data using MAE, MSE, RMSE and R² score.

6. Saving the Model (Pickle)

The best model is saved in house_model.pkl together with everything the app needs: the PolynomialFeatures object, feature columns, list of locations and furnishings, area range, the model comparison table and average price by location.

7. Streamlit Web App

The app loads the pickle file (unpickling) and gives three tabs: Predict price, Model comparison and Price by location.

Results

Test data results (MAE, MSE and RMSE are in Lakhs, so smaller is better; R² closer to 1 is better):

Model	MAE	MSE	RMSE	R²
Simple Linear	8.693	124.717	11.168	0.596
Multiple Linear	3.580	22.770	4.772	0.926
Polynomial (degree 2)	2.145	7.859	2.803	0.975
Polynomial (degree 3)	3.059	16.047	4.006	0.948

Conclusion

Polynomial Regression (degree 2) is the best model. It explains about 97.5% of the change in price, and its average error is about 2.15 Lakhs.
Simple Linear Regression is weak because area alone cannot explain the price.
Degree 3 scores lower than degree 2 on test data, so a higher degree does not help here and can start to overfit.
App Features
Input form for area, bedrooms, bathrooms, house age, parking, location and furnishing.
Shows the predicted price in Lakhs and in rupees.
Input checks (for example, area must be greater than 0).
Warning when the area is outside the range the model learned from (500 to 3500 sqft).
Warning when the predicted price is below 0.
Clear error messages when the model file is missing or broken.
Model comparison table with R² and RMSE bar charts.
Average price by location table and chart.


Tech Stack
Area	Tools
Language	Python
Data analysis	Pandas, NumPy
Visualization	Matplotlib, Seaborn
Machine learning	Scikit-learn
Model saving	Pickle
Web app	Streamlit


Project Structure
house-price-prediction/
│
├── data/
│   ├── raw/
│   │   └── hyderabad_house_prices.csv   # original data
│   └── procced/
│       └── cleaned.csv                  # cleaned data (created by demo1.py)
│
├── src/
│   ├── app.py                           # Streamlit app
│   └── house_model.pkl                  # saved model (created by demo1.py)
│
├── demo1.py                             # cleaning, EDA, model training, saving
├── settings.py                          # file paths
└── README.md




