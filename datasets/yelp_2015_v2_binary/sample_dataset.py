import pandas as pd
import csv

# Reads the original dataset
data = pd.read_csv('train.csv', header=None, names=['class', 'text'])

# Splits the dataset according to the class
data1 = data[data['class']==1]
data2 = data[data['class']==2]

# Samples 2% of each class
data1sample = data1.sample(frac=0.02, replace=False, random_state=42)
data2sample = data2.sample(frac=0.02, replace=False, random_state=42)

# Concatenates the samples
databoth = pd.concat([data1sample, data2sample], verify_integrity=True)

# Shuffles the entries
datasample = databoth.sample(frac=1, replace=False, random_state=42)

# Writes the sampled dataset
datasample.to_csv('train_2perc.csv', header=False, index=False, quoting=csv.QUOTE_ALL)


# Reads the original dataset
data = pd.read_csv('test.csv', header=None, names=['class', 'text'])

# Splits the dataset according to the class
data1 = data[data['class']==1]
data2 = data[data['class']==2]

# Samples 2% of each class
data1sample = data1.sample(frac=0.02, replace=False, random_state=42)
data2sample = data2.sample(frac=0.02, replace=False, random_state=42)

# Concatenates the samples
databoth = pd.concat([data1sample, data2sample], verify_integrity=True)

# Shuffles the entries
datasample = databoth.sample(frac=1, replace=False, random_state=42)

# Writes the sampled dataset
datasample.to_csv('test_2perc.csv', header=False, index=False, quoting=csv.QUOTE_ALL)
