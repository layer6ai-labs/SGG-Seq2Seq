import os
import evaluation_vg
import pickle

if __name__ == '__main__':
    
    det_file = 'out.pkl'

    if os.path.exists(det_file):
        print('Loading results from {}'.format(det_file))
        with open(det_file, 'rb') as f:
            all_results = pickle.load(f)
        print('Starting evaluation now...')
        evaluation_vg.eval_rel_results(all_results)

