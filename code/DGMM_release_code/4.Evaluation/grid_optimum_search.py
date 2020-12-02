import argparse
import os
import numpy as np
import glob
import pdb

def main():
    parser = argparse.ArgumentParser(description='Find the optimum solution after grid search.')
    parser.add_argument('input_results_dir_path', help='Path to the directory of input results files.')

    args = parser.parse_args()

    results = []
    # Read the results files
    params = os.listdir(args.input_results_dir_path)
    for p in params:
        s_params = p.split('_')
        file_name = glob.glob(args.input_results_dir_path + p + '/' + '*.txt')[0]
        with open(file_name) as f:
            content = f.readlines()
        auc = content[2].split('\n')[0].split(' ')[-1]
        if len(s_params) == 1:
            results.append((s_params[0], auc))
        else:
            results.append(np.array((s_params[0], s_params[1], auc)))

    results = np.array(results)
    if len(s_params) == 1:
        optim_params = results[np.argmax(results[:,1])]
        print('Optimum solution after grid search: threshold {} -> F1: {}'.format(optim_params[0], optim_params[1]))
    else:
        optim_params = results[np.argmax(results[:,2])]
        worst_params = results[np.argmin(results[:,2])]
        print('Optimum solution after grid search: dynamic minimum {} and area minimum {} -> F1: {}'.format(optim_params[0], optim_params[1], optim_params[2]))
        print('Worst value after grid search: dynamic minimum {} and area minimum {} -> F1: {}'.format(worst_params[0], worst_params[1], worst_params[2]))
    
if __name__ == '__main__':
    main()
