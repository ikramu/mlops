import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import joblib

def train_model():
    # Generate synthetic data
    data = pd.DataFrame({
        'area': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
        'bedrooms': [1, 2, 2, 3, 3, 3, 4, 4, 5, 5],
        'price': [100, 150, 180, 220, 250, 280, 320, 350, 400, 450]
    })

    X = data[['area', 'bedrooms']]
    y = data['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = root_mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    joblib.dump(model, 'models/house_price_model.joblib')

if __name__ == "__main__":
    train_model()
    