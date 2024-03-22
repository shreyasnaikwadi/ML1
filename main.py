import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Generate synthetic data
np.random.seed(0)  # For reproducibility

# Define the number of samples
n_samples = 1000

# Generate random features
size = np.random.normal(loc=1500, scale=300, size=n_samples)
num_rooms = np.random.randint(1, 6, size=n_samples)
location = np.random.choice(['Urban', 'Suburban', 'Rural'], size=n_samples)

# Generate random prices based on features
price = 1000 * size + 500 * num_rooms + np.random.normal(loc=0, scale=10000, size=n_samples)

# Create a DataFrame
house_data = pd.DataFrame({'size': size, 'num_rooms': num_rooms, 'location': location, 'price': price})

# Step 2: Data Preprocessing

# Define preprocessing steps for numerical and categorical features
numeric_features = ['size', 'num_rooms']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['location']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Step 3: Model Training

# Split the dataset into training and testing sets
X = house_data.drop('price', axis=1)
y = house_data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the regression model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Train the model
model.fit(X_train, y_train)

# Step 4: Evaluation

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
