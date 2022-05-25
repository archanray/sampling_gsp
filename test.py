import numpy as np
from timeit import default_timer as timer
from src.utils import scipyMatMul as sMM
from src.utils import computeSquaredDistance as cSD
from tqdm import tqdm

loops = 100

y = np.random.random_sample(5000)
y = np.expand_dims(y, axis=1)
start = timer()
for i in tqdm(range(loops)):
	dists = sMM(y)
end = timer()

print("scipy time:", (end-start)/loops)

start = timer()
for i in tqdm(range(loops)):
	dists = cSD(y,y)
end = timer()

print("my time:", (end-start)/loops)

