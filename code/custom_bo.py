import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
import csv
import os

from automate import modify_netlist, run_ltspice, extract_results_from_log, generate_random_params
from surrogate_model import get_models, create_models, train_models
from automate import param_space, netlist_template_path, netlist_run_path, log_path


csv_output = "custom_bo_results.csv" 
random_results_file = "sim_results.csv" 

alpha = 0.2
def log_bo_result(params_dict, gain, power, score):
    """function to log results into CSV file"""
    fieldnames = list(params_dict.keys()) + ['gain_db', 'power_watts']
    

    if not os.path.exists(csv_output):
        with open(csv_output, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    # Append the new row
    with open(csv_output, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Prepare row data with proper handling of values (no score)
        row = params_dict.copy()
        row.update({
            "gain_db": gain, 
            "power_watts": power
        })
        
        # Write the row
        writer.writerow(row)

def objective(params_dict):
    """ Objective function that runs LTSpice simulation
     We compute the score for the given parameters, and log the results
    """
    modify_netlist(netlist_template_path, netlist_run_path, params_dict)
    run_ltspice(netlist_run_path)
    time.sleep(1) # sleep so that we can run the LTspice simulation and store the results
    
    gain, power = extract_results_from_log(log_path)
    
    if gain is None or power is None or gain < 0:
        return 1000000 # if we get an invalid result, assign an absurdly high score
    
    score = -gain + 0.2 * power #  weighted-sum methoddecided on this parameter after reference to a paper
    log_bo_result(params_dict, gain, power, score)
    return score

def expected_improvement(points_to_evaluate, sample_points_observed, objective_values_observed, gp_gain, rf_power):
    """ Expected Improvement acquisition function"""
    # Predict gain and power using surrogate models

    gain_pred, gain_std = gp_gain.predict(points_to_evaluate.reshape(1, -1), return_std=True)
    log_power_pred = rf_power.predict(points_to_evaluate.reshape(1, -1))
    power_pred = np.power(10, log_power_pred)
    
    # Calculate predicted score
    score_pred = -gain_pred + alpha * power_pred

    # EI formula
    current_best_score = np.min(objective_values_observed)
    imp = current_best_score - score_pred
    Z = imp / (gain_std + 1e-10)
    ei = imp * norm.cdf(Z) + gain_std * norm.pdf(Z)
    
    # exploration-exploitation
    ei = ei + 0.1 * gain_std # decided on 0.01 exploration-exploitation because we saw that we got the best score within 9 iterations on 0.001
    # hence I decided to explore more points since we have 50 iterations 
    return ei[0]

def propose_next_point(sample_points_observed, objective_values_observed, gp_gain, rf_power, param_keys):
    """
    We generate a large number of random candidate points and then use the expected improvement function to find the next point to sample
    which is then actually ereturned and evaluated in LTSpice.
    """
    points_to_evaluate = []
    
    for _ in range(10000): # 10000 random points to be chosen from 
        params = generate_random_params(param_space)
        candidate = [params[key] for key in param_keys]
        points_to_evaluate.append(candidate)
    
    points_to_evaluate = np.array(points_to_evaluate)
    
    ei_values = []
    for candidate in points_to_evaluate:
        ei = expected_improvement(candidate, sample_points_observed, objective_values_observed, gp_gain, rf_power)
        ei_values.append(ei)

    return points_to_evaluate[np.argmax(ei_values)] # return the best EI point

def custom_bayesian_optimization():
    """ Custom Bayesian Optimization implementation using our surrogate models """

    gp_gain, rf_power, param_keys = get_models()
    
    sample_points_ = []
    y_scores = []
    all_params = []
    gain_values = []
    power_values = []
    valid_indices = [] 
    
    for i in range(10): # 10 random starting points
        params = generate_random_params(param_space)
        params_list = [params[key] for key in param_keys]
        sample_points_.append(params_list)
        all_params.append(params)
        
        modify_netlist(netlist_template_path, netlist_run_path, params)
        run_ltspice(netlist_run_path)
        time.sleep(1)
        
        gain, power = extract_results_from_log(log_path)
        
        if gain is None or power is None or gain < 0:
            score = 1000000  # invalid result
            gain_values.append(0)
            power_values.append(0)
        else: # valid point append it to the list for retraining the surrogate model
            score = -gain + alpha * power
            gain_values.append(gain)
            power_values.append(np.log10(power + 1e-10))  # Log transform as in surrogate_model.py
            valid_indices.append(i) 
            
        log_bo_result(params, gain, power, score)
        y_scores.append(score)
        
        print(f"Random point {i+1}/10 - Score: {score:.4f}")
    
    # Main BO loop
    for i in range(50):
        X_valid = np.array([sample_points_[j] for j in valid_indices])
        gain_valid = np.array([gain_values[j] for j in valid_indices])
        power_valid = np.array([power_values[j] for j in valid_indices])
        
        if len(valid_indices) > 2:
            # Use train_models from surrogate_model.py
            gp_gain, rf_power = train_models(X_valid, gain_valid, power_valid)
        
        next_point = propose_next_point(np.array(sample_points_), np.array(y_scores), gp_gain, rf_power, param_keys)
        next_params = dict(zip(param_keys, next_point))
        
        # get the next point and run a simulation
        modify_netlist(netlist_template_path, netlist_run_path, next_params)
        run_ltspice(netlist_run_path)
        time.sleep(1)
        
        gain, power = extract_results_from_log(log_path)
        
        if gain is None or power is None or gain < 0:
            score = 1000000  # handling a invalid score invalid result
            gain_values.append(0)
            power_values.append(0)
        else:
            score = -gain + alpha * power
            gain_values.append(gain)
            power_values.append(np.log10(power + 1e-10))
            valid_indices.append(len(sample_points_)) 
            
        log_bo_result(next_params, gain, power, score)
        
        sample_points_.append(next_point)
        y_scores.append(score)
        all_params.append(next_params)
        
        best_idx = np.argmin(y_scores)
        best_score = y_scores[best_idx]
        best_params = all_params[best_idx]
        
        # print(f"Current best score: {best_score}")
    
    best_idx = np.argmin(y_scores)
    return all_params[best_idx], y_scores[best_idx]


#thanks chatgpt for the plot function
def plot_results():
    """Plot optimization results"""
    df = pd.read_csv(csv_output)
    df["score"] = -df["gain_db"] + alpha * df["power_watts"]
    
    # Plot score over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(df["score"], marker='o', linestyle='-')
    plt.xlabel("Iteration")
    plt.ylabel("Score (-Gain + α·Power)")
    plt.title("Custom Bayesian Optimization Score Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("custom_bo_score.png")
    plt.show()
    
    # Print best result
    best_result = df.loc[df["score"].idxmin()]
    print("\n Best Design Found:")
    print(best_result)
    
    # Compare with random search if available
    if os.path.exists(random_results_file):
        df_rand = pd.read_csv(random_results_file)
        df_rand = df_rand.dropna(subset=["gain_db", "power_watts"])
        df_rand["score"] = -df_rand["gain_db"] + alpha * df_rand["power_watts"]
        
        best_rand_score = df_rand["score"].min()
        best_bo_score = df["score"].min()
        
        print("\n Comparison with Random Sampling:")
        print(f"  Best Random Score: {best_rand_score:.4f}")
        print(f"  Best Custom BO Score: {best_bo_score:.4f}")
        print(f"  Improvement: {best_rand_score - best_bo_score:.4f} ({100*(best_rand_score - best_bo_score)/abs(best_rand_score):.2f}%)")
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        
        # Plot best score found so far for each approach
        def plot_convergence(data, label):
            best_so_far = []
            best = float('inf')
            for score in data["score"]:
                if score < best:
                    best = score
                best_so_far.append(best)
            plt.plot(best_so_far, label=label)
        
        # Use only the first N points from random sampling (same as BO iterations)
        n_points = min(len(df), len(df_rand))
        plot_convergence(df_rand.head(n_points), "Random Sampling")
        plot_convergence(df, "Custom BO")
        
        plt.xlabel("Iteration")
        plt.ylabel("Best Score Found")
        plt.title("Convergence Comparison: Random vs Custom BO")
        plt.legend()
        plt.grid(True)
        plt.savefig("random_vs_bo_comparison.png")
        plt.show()

def compare_approaches():
    """Compare original BO with custom BO approach"""
    if not os.path.exists(csv_output) or not os.path.exists("bo_results.csv"):
        print("Cannot compare - one or both result files missing")
        return
    
    df_custom = pd.read_csv(csv_output)
    df_original = pd.read_csv("bo_results.csv")
    
    if "score" not in df_custom.columns:
        df_custom["score"] = -df_custom["gain_db"] + alpha * df_custom["power_watts"]
    
    if "score" not in df_original.columns:
        df_original["score"] = -df_original["gain_db"] + alpha * df_original["power_watts"]
    
    best_custom_score = df_custom["score"].min()
    best_original_score = df_original["score"].min()
    
    best_custom = df_custom.loc[df_custom["score"].idxmin()]
    best_original = df_original.loc[df_original["score"].idxmin()]
    
    # Print comparison
    print("\nComparison of BO Approaches")
    print(f"Original BO Best Score: {best_original_score:.4f}")
    print(f"Custom BO Best Score:   {best_custom_score:.4f}")
    print(f"Improvement:            {best_original_score - best_custom_score:.4f} ({100*(best_original_score - best_custom_score)/best_original_score:.2f}%)")
    
    # Plot convergence comparison
    plt.figure(figsize=(12, 6))
    
    # Plot best score found so far for each approach
    def plot_convergence(df, label):
        best_so_far = []
        best = float('inf')
        for score in df["score"]:
            if score < best:
                best = score
            best_so_far.append(best)
        plt.plot(best_so_far, label=label)
    
    plot_convergence(df_original, "Original BO (skopt)")
    plot_convergence(df_custom, "Custom BO (our surrogate models)")
    
    plt.xlabel("Iteration")
    plt.ylabel("Best Score Found")
    plt.title("Convergence Comparison of BO Approaches")
    plt.legend()
    plt.grid(True)
    plt.savefig("bo_comparison.png")
    plt.show()
    
    # Compare final parameters
    print("\nBest Parameter Configurations")
    param_keys = ["W_M1", "W_M2", "W_M3", "W_M4", "W_M5", "W_M6", "I1", "C1", "C2"] 
    
    print("\nOriginal BO:")
    for key in param_keys:
        print(f"  {key}: {best_original[key]}")
        
    print("\nCustom BO:")
    for key in param_keys:
        print(f"  {key}: {best_custom[key]}")

    print("\nParameter differences:")
    for key in param_keys:
        diff = abs(best_custom[key] - best_original[key])
        diff_pct = 100 * diff / ((best_custom[key] + best_original[key])/2)
        print(f"  {key}: {diff:.4e} ({diff_pct:.2f}%)")

if __name__ == "__main__":
    best_params, best_score = custom_bayesian_optimization()
    
    print("\nOptimization complete!")
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score:.4f}")
    
    plot_results()
    compare_approaches() 