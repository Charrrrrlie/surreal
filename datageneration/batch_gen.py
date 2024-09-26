import os
from tqdm import tqdm

# multi threading
from multiprocessing.pool import ThreadPool

def func(idx):
    if idx == 0:
        for ishape in tqdm(range(2000), desc=f"Processing idx {idx}", leave=False):
            os.system("./run.sh \'--idx {} --ishape {} --stride 50\'  > NUL{} 2>&1 ".format(idx, ishape, idx))
    else:
        for ishape in range(2000):
            os.system("./run.sh \'--idx {} --ishape {} --stride 50\'  > NUL{} 2>&1 ".format(idx, ishape, idx))

if __name__ == '__main__':
    pool_size = 10
    
    pool = ThreadPool(pool_size)
    pool.map(func, range(pool_size))
    
    pool.close()
    