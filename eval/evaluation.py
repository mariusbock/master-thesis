import os
import sys

from evaluateTracking_modified import evaluateTracking
from gcn_clustering.utils.logging import Logger

files_path = 'logs/20200706/114353/'
seqinfo_path = 'data/MOT/MOT17'
seqmap = 'eval/seqmaps/MOT17-09.txt'
evalmap_val = 'eval/evalmaps/MOT17-09_val.txt'
evalmap_test = 'eval/evalmaps/MOT17-09_test.txt'

# saves logs to a file (standard output redirected)
sys.stdout = Logger(os.path.join(files_path, 'eval_log.txt'))

print('EVALUATION RESULTS')
evaluateTracking(files_path, seqinfo_path, seqmap, evalmap_val)

print('TEST RESULTS')
evaluateTracking(files_path, seqinfo_path, seqmap, evalmap_test)
