import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def load_data():
    """Load and preprocess the simulation data"""
    df = pd.read_csv("sim_results.csv")
    df = df.dropna(subset=["gain_db", "power_watts"])  # cleaning
    df = df[df["gain_db"] > 0] #filter out negative gain
    df = df[df["power_watts"] < 0.05] 
    df["log_power"] = np.log10(df["power_watts"] + 1e-10)
    
    X = df[["W_M1", "W_M2", "W_M3", "W_M4", "W_M5", "W_M6", "I1", "C1", "C2"]].values
    y_gain = df["gain_db"].values
    y_power = df["log_power"].values # log transform for extremely close values
    
    return X, y_gain, y_power

def create_models():
    """Create and return untrained surrogate models"""
    # Create GP model for gain with advanced kernel
    kernel_gain = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) + WhiteKernel(noise_level=1e-5)
    gp_gain = GaussianProcessRegressor(kernel=kernel_gain, normalize_y=True)
    
    # Create RF model for power
    rf_power = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    
    return gp_gain, rf_power

def train_models(X_train, y_gain_train, y_power_train):
    """Train gain and power surrogate models"""
    gp_gain, rf_power = create_models()
    
    # Train models
    gp_gain.fit(X_train, y_gain_train)
    rf_power.fit(X_train, y_power_train)
    
    return gp_gain, rf_power

def evaluate_models(gp_gain, rf_power, X_test, y_gain_test, y_power_test):
    """Evaluate surrogate models"""
    y_gain_pred = gp_gain.predict(X_test)
    y_power_pred = rf_power.predict(X_test)
    
    # Evaluate gain model
    gain_mse = mean_squared_error(y_gain_test, y_gain_pred)
    gain_r2 = r2_score(y_gain_test, y_gain_pred)
    print("\nGain Model Evaluation:")
    print(f"MSE: {gain_mse:.4f}")
    print(f"R² Score: {gain_r2:.4f}")
    
    # Evaluate power model
    power_mse = mean_squared_error(y_power_test, y_power_pred)
    power_r2 = r2_score(y_power_test, y_power_pred)
    print("\nPower Model Evaluation:")
    print(f"MSE: {power_mse:.4f}")
    print(f"R² Score: {power_r2:.4f}")
    
    return gain_mse, gain_r2, power_mse, power_r2

def get_models():
    """Get trained surrogate models for initial use"""
    X, y_gain, y_power = load_data()
    
    # Split data
    X_train, X_test, y_gain_train, y_gain_test = train_test_split(X, y_gain, test_size=0.2, random_state=42)
    _, _, y_power_train, y_power_test = train_test_split(X, y_power, test_size=0.2, random_state=42)
    
    # Train models
    gp_gain, rf_power = train_models(X_train, y_gain_train, y_power_train)
    
    # Get parameter keys
    param_keys = ["W_M1", "W_M2", "W_M3", "W_M4", "W_M5", "W_M6", "I1", "C1", "C2"]
    
    # Evaluate models
    evaluate_models(gp_gain, rf_power, X_test, y_gain_test, y_power_test)
    
    return gp_gain, rf_power, param_keys

def plot_model_performance():
    """Plot surrogate model performance"""
    X, y_gain, y_power = load_data()
    
    # Split data
    X_train, X_test, y_gain_train, y_gain_test = train_test_split(X, y_gain, test_size=0.2, random_state=42)
    _, _, y_power_train, y_power_test = train_test_split(X, y_power, test_size=0.2, random_state=42)
    
    # Train models
    gp_gain, rf_power = train_models(X_train, y_gain_train, y_power_train)
    
    # Predictions
    y_gain_pred = gp_gain.predict(X_test)
    y_power_pred = rf_power.predict(X_test)
    
    # Plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_gain_test, y_gain_pred, c='blue', alpha=0.6)
    plt.plot([y_gain_test.min(), y_gain_test.max()], [y_gain_test.min(), y_gain_test.max()], 'k--')
    plt.xlabel("Actual Gain (dB)")
    plt.ylabel("Predicted Gain (dB)")
    plt.title("Gain: Actual vs Predicted")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(y_power_test, y_power_pred, c='green', alpha=0.6)
    plt.plot([y_power_test.min(), y_power_test.max()], [y_power_test.min(), y_power_test.max()], 'k--')
    plt.xlabel("Actual Log Power")
    plt.ylabel("Predicted Log Power")
    plt.title("Power: Actual vs Predicted")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("surrogate_model_performance.png")
    plt.show()

if __name__ == "__main__":
    # Demo usage
    plot_model_performance()
    gp_gain, rf_power, param_keys = get_models()


"""classic kernel 
Gain Model Evaluation:
  - MSE: 14.8992
  - R² Score: 0.8471

  Power Model Evaluation:
  - MSE: 0.0859
  - R² Score: -0.0550
   
White kernel 
 Gain Model Evaluation:
  - MSE: 6.6827
  - R² Score: 0.9342

  Power Model Evaluation:
  - MSE: 0.1153
  - R² Score: 0.1199

white kernel for gain and randomforestregressor for power
    Gain Model Evaluation:
  - MSE: 6.6827
  - R² Score: 0.9342

Power Model Evaluation:
  - MSE: 0.0534
  - R² Score: 0.5921

 """