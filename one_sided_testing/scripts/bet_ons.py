import argparse
import json
import os
import math
import numpy as np
from methods_ons import betting_experiment


def convert_np_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist() 
    elif isinstance(obj, dict):
        return {key: convert_np_to_list(value) for key, value in obj.items()}  
    elif isinstance(obj, list):
        return [convert_np_to_list(item) for item in obj]  
    else:
        return obj  


def process_files(data1, data2, mu_0, alphas, iters):
    if data1 is None or data2 is None:
        return None
        
    ops_tau, ops_tpr = betting_experiment(data1, mu_0, alphas, iters)
    _, ops_fpr = betting_experiment(data2, mu_0, alphas, iters)

    
    return {
        "method": 'ONS',
        "rejection_time": np.ceil(np.mean(ops_tau, axis=0)), 
        "power": np.mean(ops_tpr, axis=0),#true=1/false=0
        "fpr": np.mean(ops_fpr, axis=0) #type-1 error 
    }


def call_process_ons(data1, data2, mu_0, alphas, iters, output_file=None):
    results = process_files(data1, data2, mu_0, alphas, iters)
    
    
    if os.path.exists(output_file):
            with open(output_file, 'r') as file:
                try:
                    data = json.load(file)
                    if not isinstance(data, list):
                        data = [] 
                except json.JSONDecodeError:
                    data = []  
    else:
        data = []
    
    converted_results = convert_np_to_list(results)
    data.append(converted_results)
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)
    

if __name__ == "__main__":
    import sys
    data1 = np.array(eval(sys.argv[1]))
    data2 = np.array(eval(sys.argv[2]))
    mu_0  = np.array(eval(sys.argv[3]))
    alphas = np.linspace(float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]))
    iters = int(sys.argv[7])
    output_file = None if len(sys.argv) <= 8 else sys.argv[8]

    call_process_ons(data1, data2, mu_0, alphas, iters, output_file)