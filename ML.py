import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from codecarbon import EmissionsTracker
import time

# Start tracking emissions
tracker = EmissionsTracker(project_name="DecisionTree_Model")
tracker.start()
start_time = time.time()

# Load training data
train_file = 'train data i0_i _final.csv'
train_data = pd.read_csv(train_file)

# Features and target
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Split data
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_scaled, y_train, test_size=0.3, random_state=42
)

# Train Decision Tree Regressor
decision_tree_model = DecisionTreeRegressor(random_state=42)
decision_tree_model.fit(X_train_split, y_train_split)

# Stop tracking emissions
emissions = tracker.stop() # Carbon Emissions rounded to 4 decimal places
end_time = time.time()
runtime = end_time - start_time  # Runtime rounded to 4 decimal places

# Save the trained model
model_filename = "decision_tree_model.pkl"
with open(model_filename, "wb") as file:
    pickle.dump(decision_tree_model, file)

print(f"Model saved as {model_filename}")

# Save emissions data with rounded values
emissions_data = pd.DataFrame({
    "Model": ["Decision Tree"],
    "Carbon Emissions (kg CO2)": [emissions],
    "Energy Consumption (kWh)": [(float(tracker._total_energy))],  # Energy Consumption rounded to 4 decimal places
    "Runtime (s)": [runtime]
})
emissions_data.to_csv("emissions.csv", index=False)

# LIME Explanation
explainer = LimeTabularExplainer(
    training_data=X_train_split,
    training_labels=y_train_split.values,
    feature_names=X_train.columns,
    mode='regression'
)

# Select an instance for explanation
instance_index = 14
instance = X_val_split[instance_index].reshape(1, -1)

# Generate LIME explanation
explanation = explainer.explain_instance(
    data_row=instance.flatten(),
    predict_fn=decision_tree_model.predict
)

# Get actual and predicted values
actual_value = y_val_split.iloc[instance_index]
predicted_value = decision_tree_model.predict(instance)[0]

# Plot and save LIME explanation
lime_fig = explanation.as_pyplot_figure()
lime_fig.savefig('lime_explanation_plot.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
plt.show()

# SHAP Waterfall Plot
shap_explainer = shap.Explainer(decision_tree_model, X_train_split, feature_names=X_train.columns)
shap_values = shap_explainer(instance)

fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[0], max_display=10, show=False)
plt.savefig("waterfall_plot.png", dpi=600, bbox_inches='tight')
plt.show()

# Save SHAP force plot
force_plot = shap.force_plot(shap_explainer.expected_value, shap_values.values[0],
                             instance.flatten(), feature_names=X_train.columns)
shap.save_html("force_plot.html", force_plot)

# Load and display sustainability data
emissions_df = pd.read_csv("emissions.csv")

# Display rounded sustainability metrics
print("\nSustainability Metrics:")
print(emissions_df.to_string(index=False))

# Display predicted ratio
print(f"\nPredicted Ratio: {predicted_value:.5f}")
