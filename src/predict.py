import joblib

def predict_price(area, bedrooms):
    model = joblib.load('models/house_price_model.joblib')
    prediction = model.predict([[area, bedrooms]])
    return prediction[0]

if __name__ == "__main__":
    area = 100
    bedrooms = 3
    price = predict_price(area, bedrooms)
    print(f"Predicted price for a house with {area} sq ft and {bedrooms} bedrooms: ${price:.2f}")
