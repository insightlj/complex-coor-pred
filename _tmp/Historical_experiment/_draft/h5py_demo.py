import h5py
import time
import random

train_file = h5py.File("AlignCoorConfusion/h5py_data/train_dataset.h5py")

beg = time.time()
for i in range(100):
    protein1 = train_file["protein"+str(i)]["aligned_chains"]
t1 = time.time() - beg

beg = time.time()
for i in range(100):
    num = random.randint(0,20000)
    protein1 = train_file["protein"+str(num)]["aligned_chains"]
t2 = time.time() - beg


print(t1)
print(t2)

train_file.close()