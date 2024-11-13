import os
import numpy as np
import scipy.io as sio
import pandas as pd

# Initialize lists to hold the data
data_all = []
foot = []
speed = []
stride_length = []
step_width = []
cadence = []

# Loop through 50 subjects
for i in range(1, 51):
    mat_file = f'./Subject{i}.mat'
    if not os.path.exists(mat_file):
        print(f"File {mat_file} not found!")
        continue
    
    mat_data = sio.loadmat(mat_file)
    s = mat_data['s'][0, 0]  # Access the structure 's'

    # Loop through trials in the subject data
    for j in range(len(s['Data'][0])):
        trial = s['Data'][0, j]
        
        # Check if the task is "Walking"
        task = trial['Task'][0]
        if task != 'Walking':
            continue
        
        # Extract angular data for indices [4, 7, 10]
        ang_data = trial['Ang'][[3, 6, 9], :] * np.pi / 180 
        data_all.append(ang_data)
        
        # Extract properties
        if trial['Foot'][0] == 'RX':
            foot.append(0)
        else:
            foot.append(1)
        
        speed.append(trial['speed'][0, 0])
        stride_length.append(trial['strideLength'][0, 0])
        step_width.append(trial['stepWidth'][0, 0])
        cadence.append(trial['cadence'][0, 0])

# Convert collected data to NumPy arrays
data_all = np.array(data_all)  # Shape: [N, 3, T]
foot = np.array(foot)
speed = np.array(speed)
stride_length = np.array(stride_length)
step_width = np.array(step_width)
cadence = np.array(cadence)

# Create a Pandas DataFrame for properties
prop_all = pd.DataFrame({
    'foot': foot,
    'speed': speed,
    'stride_length': stride_length,
    'step_width': step_width,
    'cadence': cadence
})

# Shuffle and split data
np.random.seed(1234567890)
indices = np.random.permutation(len(prop_all))

# Sizes for splits
tr_size = 400
va_size = 100

# Training set
train_idx = indices[:tr_size]
data_train = data_all[train_idx]
prop_train = prop_all.iloc[train_idx]

# Validation set
val_idx = indices[tr_size:tr_size + va_size]
data_valid = data_all[val_idx]
prop_valid = prop_all.iloc[val_idx]

# Test set
test_idx = indices[tr_size + va_size:]
data_test = data_all[test_idx]
prop_test = prop_all.iloc[test_idx]

# Save to files
np.save('data_train.npy', data_train)
prop_train.to_csv('prop_train.txt', sep=' ', index=False)

np.save('data_valid.npy', data_valid)
prop_valid.to_csv('prop_valid.txt', sep=' ', index=False)

np.save('data_test.npy', data_test)
prop_test.to_csv('prop_test.txt', sep=' ', index=False)