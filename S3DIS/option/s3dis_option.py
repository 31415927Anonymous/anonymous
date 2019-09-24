#!/usr/bin/python3
import math

num_class = 13

sample_num = 2048
sample_num_extra = 1024

batch_size = 8 

num_epochs = 1024

label_weights = [1.0] * num_class

learning_rate_base = 0.0005
decay_steps = 5000
decay_rate = 0.8
learning_rate_min = 1e-6
step_val = 500

weight_decay = 1e-8

jitter = 0.0
jitter_val = 0.0

rotation_range = [0, 0, math.pi/32., 'u']
rotation_range_val = [0, 0, 0, 'u']
rotation_order = 'rxyz'

scaling_range = [0.001, 0.001, 0.001, 'g']
scaling_range_val = [0, 0, 0, 'u']

sample_num_variance = 1 // 8
sample_num_clip = 1 // 4


optimizer = 'adam'
epsilon = 1e-5

data_dim = 6
use_extra_features = True
with_normal_feature = False


