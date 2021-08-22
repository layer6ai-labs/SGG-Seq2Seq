import sys
import os
file_to_eval = sys.argv[1]
cmd = "python multicoreconvert_seq.py " + file_to_eval
os.system(cmd)
cmd3 = " python test_results.py"
os.system(cmd3)
