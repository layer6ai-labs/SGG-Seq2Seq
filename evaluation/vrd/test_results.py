import os
import pickle
import evaluation_vrd
if __name__ == '__main__':

    det_file = 'out.pkl'
    if os.path.exists(det_file):
        with open(det_file, 'rb') as f:
            all_results = pickle.load(f)
        evaluation_vrd.eval_rel_results(all_results)