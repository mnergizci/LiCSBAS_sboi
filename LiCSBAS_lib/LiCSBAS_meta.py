# version control
ver='1.14.2 (dev)'
date='2023-06-28'
author="Dr. Yu Morishita and COMET dev team (ML,QO,JM,LS,..)"

# setting number of threads to small number (e.g. 1), as the multiprocessing appears slow otherwise
# solution found by Richard Rigby, Uni of Leeds
import os
os.environ["OMP_NUM_THREADS"] = 1