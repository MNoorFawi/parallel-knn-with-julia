knn in parallel with julia
================

K-Nearest Neighbors Algorithm in Parallel using Julia
-----------------------------------------------------

**K-Nearest Neighbors** algorithm known of being computationally intensive because the distance to every point has to be calculated. 
Here we're going to use **Julia**'s parallel computing power to fit **KNN algorithm** on Census Income dataset to predict whether income exceeds $50K ...
Data can be downloaded from <https://archive.ics.uci.edu/ml/machine-learning-databases/adult/> ; they're two datasets, training and test.

###### We will be using Julia version 0.6.4
``` julia
# run the script from the command line
$ julia knn_income.jl

### READING DATA ###
(:data_shape, (32561, 15), (16281, 15))

# Preparing Data:
1 Deleting any row containing any unknown value.
(:new_data_shape, (30162, 15), (15060, 15))

2 one hot encoding string columns and normalizing numeric ones.
(:encoded_data_shape, (30162, 105), (15060, 105))

3 Splitting data into train and test
# size of train and test data
(104, 30162)(104, 15060)

# Define the Model:
* Adding 4 processes => define euclidean distance function
        => get k nearest neighbors => assign new labels
(:number_of_processes, 5)

# Get predictions and accuracy on train and test data using 40 k
(:train_accuracy, 0.8310456866255553)

(:test_accuracy, 0.8288844621513944)
```
