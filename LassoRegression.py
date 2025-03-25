import pandas as pd
import numpy as np
import pickle
import time  # Added to measure runtime
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from codecarbon import EmissionsTracker

# Load training data
train_file = "train data i0_i _final.csv"
train_data = pd.read_csv(train_file)

# Separate features and target
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Initialize emissions tracker
tracker = EmissionsTracker(project_name="LassoRegression")

# Measure runtime manually
start_time = time.time()
tracker.start()  # Start tracking emissions

# Train Lasso model with different random states
best_model = None
best_r2 = -np.inf

for random_state in range(25, 56):
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=random_state
    )

    lasso_reg = Lasso(alpha=1.0, random_state=random_state)
    lasso_reg.fit(X_train_split, y_train_split)

    r2 = r2_score(y_val_split, lasso_reg.predict(X_val_split))

    if r2 > best_r2:
        best_r2 = r2
        best_model = lasso_reg

tracker.stop()  # Stop emissions tracking
end_time = time.time()

# Compute runtime manually
runtime = end_time - start_time  # Runtime in seconds

# Extract emissions data
carbon_emissions = tracker.final_emissions  # CO2 emissions in kg
energy_consumption = tracker._total_energy.kWh  # Energy in kWh

# Print results
print(f"Total Carbon Emissions: {carbon_emissions:.6f} kg CO2")
print(f"Total Energy Consumption: {energy_consumption:.6f} kWh")
print(f"Total Runtime: {runtime:.6f} seconds")

# Save the best model
with open("lasso_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

# Save the scaler
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print(f"Best Lasso model saved as 'lasso_model.pkl'")
print(f"Scaler saved as 'scaler.pkl'")
