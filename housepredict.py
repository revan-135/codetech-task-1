import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def generate_synthetic_data():
    np.random.seed(42)
    num_samples = 10000
    min_size = 500
    max_size = 5000
    min_rooms = 1
    max_rooms = 10

    size = np.random.randint(min_size, max_size + 1, num_samples)
    rooms = np.random.randint(min_rooms, max_rooms + 1, num_samples)
    price = (size * 0.03) + (rooms * 5) + np.random.normal(0, 10, num_samples)

    df = pd.DataFrame({
        "Size (sq ft)": size,
        "Number of Rooms": rooms,
        "Price (in Lakhs)": price.round(2)
    })
    return df

try:
    df = pd.read_csv("house_price_samples.csv")
except FileNotFoundError:
    print("Dataset file not found. Generating synthetic data instead.")
    df = generate_synthetic_data()

print("Dataset loaded successfully. Here are the first few rows:")
print(df.head())
print("\nDataset summary:")
print(df.describe())

X = df[["Size (sq ft)", "Number of Rooms"]]
y = df["Price (in Lakhs)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

def predict_new_house(size, rooms):
    new_house = pd.DataFrame([[size, rooms]], columns=["Size (sq ft)", "Number of Rooms"])
    predicted_price = model.predict(new_house)
    return predicted_price[0]

size = 1600  
rooms = 3    
predicted_price = predict_new_house(size, rooms)

print(f"\nPredicted Price for a {size} sq ft, {rooms}-room house: {predicted_price:.2f} Lakhs")