# HIL_toy_python
Code to solve first a stochastic shortest path problem by means of value iteration.
Then data are sampled from this optimal solution and Behavioral cloning and Hierarchical Imitation Learning are used to infer the optimal policy.

# Main
- BC_main.py  (for Behavioral Cloning)
- HIL_main (for Hierarchical Imitation Learning)
- main (for regularizers validation)

## BC_main.py

### Dependencies
```python
import tensorflow
from tensorflow import keras
import numpy
import matplotlib.pyplot
import matplotlib.animation
```

### Pipeline
- Generate Expert's policy
- Sample data from Expert
- Train Neural Networks for BC
  - NN1 uses Cross Entropy loss function
  - NN2 Mean Squared Error
  - NN3 Hinge Loss
- Evaluate Performance of NNs with different number of trajectories

## HIL_main

### Dependencies
```python
import tensorflow
from tensorflow import keras
import numpy
import matplotlib.pyplot
import matplotlib.animation
import concurrent.futures
```

### Pipeline
- Generate Expert's policy
- Sample data from Expert
- Initialize Hierarchical inverse learning hyperparameters
- Understanding Regularization 1
- Understanding Regularization 2
- Fix gains for regularization (lambdas, eta)
- Train the Triple of NNs using Baum-Welch with failure algorithm
- Save Trained model
- Load Model
- Evaluate Performance 
- Evaluate Performance given different trajectories for the training

## main

### Dependencies
```python
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as kb
import numpy as np
import matplotlib.pyplot as plt
import Environment as env
import StateSpace as ss
import DynamicProgramming as dp
import Simulation as sim
import BehavioralCloning as bc
import HierarchicalImitationLearning as hil
import concurrent.futures
from joblib import Parallel, delayed
import multiprocessing
```

### Important
This code runs the validation of the regularizers generating multithreads using "joblib.Parallel".
Run might take several minutes and cannot be killed manually once started; therefore, select the hyperparameters appropriately.

