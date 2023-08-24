# Library Imports
from numpy import mean
from numpy import std
from numpy import dstack
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import math
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.ndimage import uniform_filter1d
from sklearn.model_selection import train_test_split
from google.colab import drive
drive.mount('/content/drive')

# Read in movements, power, and pressure files. 
# Ignore time information, because this data has already been synchronized.
#data prep (2d - 3d ?)
movements_table_0 = np.loadtxt("/content/drive/MyDrive/Research/movements_2020.csv",delimiter=',') #loads data into 2d array
movements_axis_0  = movements_table_0[:,3].astype(int)
movements_dir_0   = movements_table_0[:,4].astype(int)

movements_table_1 = np.loadtxt("/content/drive/MyDrive/Research/movements_2021.csv",delimiter=',')
movements_axis_1  = movements_table_1[:,3].astype(int)
movements_dir_1   = movements_table_1[:,4].astype(int)

movements_table_2 = np.loadtxt("/content/drive/MyDrive/Research/movements_2022_v4.csv",delimiter=',')
movements_axis_2  = movements_table_2[:,3].astype(int) 
movements_dir_2   = movements_table_2[:,4].astype(int)

movements_table_3 = np.loadtxt("/content/drive/MyDrive/Research/movements_2022_2023_v1.csv",delimiter=',')
movements_axis_3  = movements_table_3[:,3].astype(int) 
movements_dir_3   = movements_table_3[:,4].astype(int)



movements_axis = np.hstack([movements_axis_0, movements_axis_1, movements_axis_2, movements_axis_3])
movements_dir = np.hstack([movements_dir_0,movements_dir_1, movements_dir_2, movements_dir_3])


power_table_0 = np.loadtxt("/content/drive/MyDrive/Research/power_2020.csv",delimiter=',')
power_table_1 = np.loadtxt("/content/drive/MyDrive/Research/power_2021.csv",delimiter=',')
power_table_2 = np.loadtxt("/content/drive/MyDrive/Research/power2022_v4.csv",delimiter=',')
power_table_3 = np.loadtxt("/content/drive/MyDrive/Research/power_2022_2023_v1.csv",delimiter=',')
power_table = np.vstack([power_table_0, power_table_1, power_table_2, power_table_3])

len_entries = np.shape(power_table)[1]
num_entries = np.shape(power_table)[0]

pressure_missing_pad_value = 0 # 0.3? This value may change.
pressure_table_0  = np.loadtxt("/content/drive/MyDrive/Research/pressure2022_v4.csv",delimiter=',')
pressure_table_1  = np.loadtxt("/content/drive/MyDrive/Research/pressure_2022_2023_v1.csv",delimiter=',')
pressure_table = np.vstack([pressure_table_0, pressure_table_1])

num_classes = 13

# Encode axis and dir into a single number.
# For now, ignore how many steps are taken.
# Encode output data for Tensorflow.
# Prepping direction numbers
def one_hot(movement_idx):
  axis = movements_axis[movement_idx]
  dir  = movements_dir[movement_idx]
  
  if axis == -1:
    return -1 # y_entry = 20
  elif dir <= -1:
    y_entry = 10 + axis
  elif dir >= 1:
    y_entry = axis
  else:
    return -1
  
  if y_entry == 1:
    return 0
  elif y_entry == 2:
    return 1
  elif y_entry == 3:
    return 2
  elif y_entry == 5:
    return 3
  elif y_entry == 6:
    return 4
  elif y_entry == 7:
    return 5
  elif y_entry == 11:
    return 6
  elif y_entry == 12:
    return 7
  elif y_entry == 13:
    return 8
  elif y_entry == 15:
    return 9
  elif y_entry == 16:
    return 10
  elif y_entry == 17:
    return 11
  elif y_entry == 20:
    return 12

  # Format power and pressure data for Tensorflow. 2d-3d?

truncated_len_entries = 400 #dividing up data set into sequences of 400
# CUTOFF_LENGTH = 100
X = []
y = [] 
fullpowerarr = []
fullpressurearr = []
np.seterr(all='raise')

data_idx = 1
while data_idx < num_entries:  
  power_arr = power_table[data_idx]
  pressure_arr = pressure_table[data_idx]   

  # print(np.count_nonzero(power_arr))

  # if np.count_nonzero(power_arr) < CUTOFF_LENGTH:
  #   data_idx += 2
  #   continue

  # Replace invalid terms.
  for idx in range(len_entries):
    current_power_val = power_arr[idx]
    current_pressure_val = pressure_arr[idx]

    # if current_power_val > -50:
    if any([current_power_val > -50,  np.isinf(current_power_val),    np.isnan(current_power_val), current_power_val <= -60]): #current_power_val<-60
      if idx == 0: 
        power_arr[idx] = np.nanmedian(power_arr)
      else:        
        power_arr[idx] = power_arr[idx-1]
  
    if any([current_pressure_val < 0, np.isinf(current_pressure_val), np.isnan(current_pressure_val)]):
      if idx == 0: 
        pressure_arr[idx] = np.nanmedian(pressure_arr)
      else:        
        pressure_arr[idx] = pressure_arr[idx-1]

  # ::-1 indexing reverses the order of the array.
  power_arr    = power_arr[::-1]
  pressure_arr = pressure_arr[::-1]    #data is stored in reverse time order 
  # Truncate down to last "num_timesteps" samples (closest to coupling movement).
  
  power_arr = power_arr[-truncated_len_entries:]                 
  pressure_arr = pressure_arr[-truncated_len_entries:] 
  
  #want the 400 data points right before the most recent coupling movement     
  # Add power row to new power array (2D) (in format for train/val split)
  # end while loop
  fullpowerarr = np.concatenate((power_arr,fullpowerarr), axis = 0)
  fullpressurearr = np.concatenate((pressure_arr, fullpressurearr), axis = 0)
  data_idx += 2 


# Split new power array into train and val data (using built in random function)
power_train, power_val = train_test_split(fullpowerarr, test_size = 0.3, random_state = 42) 
pressure_train, pressure_val = train_test_split(fullpressurearr, test_size = 0.3, random_state = 42)

# Take np.mean and np.std of JUST train data
power_train_mean = np.mean(power_train)
power_train_std = np.std(power_train)
pressure_train_mean = np.mean(pressure_train)
pressure_train_std = np.std(pressure_train)




# Restart old while loop (per row in new power array), starting with normalization (and use mean and std from new power array)


data_idx = 1
while data_idx < num_entries:
  power_arr = power_table[data_idx]
  pressure_arr = pressure_table[data_idx]   
  for idx in range(len_entries):
    current_power_val = power_arr[idx]
    current_pressure_val = pressure_arr[idx]

    # if current_power_val > -50:
    if any([current_power_val > -50,  np.isinf(current_power_val), np.isnan(current_power_val), current_power_val <= -60]): #current_power_val<-60
      if idx == 0: 
        power_arr[idx] = power_arr[idx+1]
      else:        
        power_arr[idx] = power_arr[idx-1]
  
    if any([current_pressure_val < 0, np.isinf(current_pressure_val), np.isnan(current_pressure_val)]):
      if idx == 0: 
        pressure_arr[idx] = pressure_arr[idx+1]
      else:        
        pressure_arr[idx] = pressure_arr[idx-1]

  # ::-1 indexing reverses the order of the array.
  power_arr    = power_arr[::-1]
  pressure_arr = pressure_arr[::-1]    #data is stored in reverse time order 
  # Truncate down to last "num_timesteps" samples (closest to coupling movement).
  
  power_arr = power_arr[-truncated_len_entries:]                 
  pressure_arr = pressure_arr[-truncated_len_entries:]


  
  if np.std(power_arr) != 0: #np.std(power_arr)
    power_arr -= power_train_mean
    power_arr /= power_train_std
  else:
    data_idx += 2
    continue    
    
    
    
  
  if np.std(pressure_arr) != 0:
    pressure_arr -= pressure_train_mean
    pressure_arr /= pressure_train_std

  
  if one_hot(data_idx//2) == -1:
    data_idx += 2
    continue

  y.append(one_hot(data_idx//2))

  # Organize power and pressure pairs for Tensorflow.
  ## Since you already split into train and test, here you would reformat (if needed) into train_X, train_y, val_X and val_y

  X_entry = []
  for entry_idx in range(truncated_len_entries):
    power_val    = power_arr[entry_idx]
    pressure_val = pressure_arr[entry_idx]
    X_entry.append([power_val, pressure_val]) 
  X.append(X_entry)
  data_idx += 2




X = np.array(X)

#reshape into 2d array
n_input = X.shape[1]*X.shape[2]
X = X.reshape(X.shape[0], n_input)
print(np.shape(X))

# reformat into training data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 42) 

# SPLIT_PROP = .7
num_classes = 13
y_train = to_categorical(y_train,num_classes=num_classes)
y_val   = to_categorical(y_val,num_classes=num_classes)

# fit and evaluate model
verbose = 1

epochs     = 30
batch_size = 3
n_features = 2

print(X_train.shape[0])
n_timesteps = X_train.shape[1]

model = Sequential()
model.add(Dense(32, activation='relu', input_dim = n_input)) # 64
model.add(Dense(8))
model.add(Dense(num_classes, activation='softmax')) #num_classes
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.build((None,n_features)) 

model.summary(
  line_length=None,
  positions=None,
  print_fn=None,
  expand_nested=False,
  show_trainable=False,
)
# fit network
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_data= (X_val,y_val), verbose=verbose)

predictions = model.predict(X_train)
print(predictions.shape)
choices = []
for prediction in predictions:
  choices.append(np.argmax(prediction))


predictions = model.predict(X_val)
choices = []
for prediction in predictions:
  choices.append(np.argmax(prediction))
