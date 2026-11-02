# Streamlit app: House Price Prediction (Hyderabad)
# Run:  streamlit run src\app.py
# First run demo1.py once, so that house_model.pkl is created.

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "house_model.pkl")


# ---------- Unpickling: load the saved model ----------

@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


try:
    data = load_model()
except FileNotFoundError:
    st.error("Model file `house_model.pkl` not found. Please run `demo1.py` first.")
    st.stop()
except (pickle.UnpicklingError, EOFError):
    st.error("Model file is broken. Please run `demo1.py` again.")
    st.stop()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

model = data["model"]
poly = data["poly"]
feature_cols = data["feature_cols"]
locations = data["locations"]
furnishings = data["furnishings"]
area_min = data["area_min"]
area_max = data["area_max"]
results = data["results"]
avg_price = data["avg_price_by_location"]


# ---------- Prediction function ----------

def predict_price(area, bedrooms, bathrooms, age, parking, location, furnishing):
    if area <= 0:
        raise ValueError("Area must be greater than 0")
    if location not in locations:
        raise ValueError("Unknown location: " + str(location))
    if furnishing not in furnishings:
        raise ValueError("Unknown furnishing: " + str(furnishing))

    row = pd.DataFrame(0, index=[0], columns=feature_cols)
    row["Area_sqft"] = area
    row["Bedrooms"] = bedrooms
    row["Bathrooms"] = bathrooms
    row["Age_years"] = age
    row["Parking"] = parking

    # Banjara Hills and Fully-Furnished have no column (drop_first=True)
    if "Location_" + location in row.columns:
        row["Location_" + location] = 1
    if "Furnishing_" + furnishing in row.columns:
        row["Furnishing_" + furnishing] = 1

    return model.predict(poly.transform(row.values))[0]


# ---------- Page ----------

st.title("🏠 Hyderabad House Price Prediction")
st.caption("Model: Polynomial Regression (degree 2). Practice data, not real market prices.")

tab1, tab2, tab3 = st.tabs(["Predict price", "Model comparison", "Price by location"])

# ----- Tab 1: Predict -----
with tab1:
    st.subheader("Enter house details")

    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("Area (sqft)", min_value=100, max_value=10000, value=1500, step=50)
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
        age = st.number_input("Age of house (years)", min_value=0, max_value=100, value=5)
    with col2:
        parking = st.selectbox("Parking spots", [0, 1, 2], index=1)
        location = st.selectbox("Location", locations)
        furnishing = st.selectbox("Furnishing", furnishings)

    if area < area_min or area > area_max:
        st.warning(
            f"The model learned from houses of {area_min} to {area_max} sqft. "
            "For other sizes the prediction may not be correct."
        )

    if st.button("Predict price", type="primary"):
        try:
            price = predict_price(area, bedrooms, bathrooms, age, parking, location, furnishing)
            if price <= 0:
                st.warning("The model gave a price below 0. These inputs are outside what the model learned.")
            else:
                st.metric("Predicted price", f"₹ {price:,.2f} Lakhs")
                st.caption(f"About ₹ {price * 100000:,.0f}")
        except ValueError as e:
            st.error(f"Input error: {e}")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ----- Tab 2: Comparison -----
with tab2:
    st.subheader("All models on test data")
    st.dataframe(results.round(3), hide_index=True, width="stretch")

    st.write("**R² score** (closer to 1 is better)")
    st.bar_chart(results.set_index("Model")["R2"])

    st.write("**RMSE** (average error in lakhs, smaller is better)")
    st.bar_chart(results.set_index("Model")["RMSE"])

    best = results.loc[results["R2"].idxmax(), "Model"]
    st.success(f"Best model: {best}")

# ----- Tab 3: Location -----
with tab3:
    st.subheader("Average price by location (Lakhs)")
    st.dataframe(avg_price, hide_index=True, width="stretch")
    st.bar_chart(avg_price.set_index("Location")["Average_Price_Lakhs"])
