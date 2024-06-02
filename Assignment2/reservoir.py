import sys
import numpy as np
import network as Network_library
import data as Data
import matplotlib.pyplot as plt

def reservoir_class(test_name, filter_name, classifier, num_nodes, input_probability, reservoir_probability):
    """
    Conducts reservoir computing for classification tasks on specified datasets.

    Parameters:
        test_name (str): Name of the test dataset. Supported values are '5s' and 'lvr'.
        filter_name (str): Name of the frequency band for data filtering.
                           Must be one of the keys in the spectral_bands dictionary.
        classifier (str): Type of classifier to use. Supported values are 'lin', 'log', and '1nn'.
        num_nodes (int): Number of nodes in the reservoir layer.
        input_probability (float): Probability of non-zero input connections in the reservoir.
        reservoir_probability (float): Probability of non-zero recurrent connections in the reservoir.

    Raises:
        ValueError: If the specified test, classifier, or filter is not supported.

    Returns:
        Accuracy
    """
    d = Data.Data(80) #80% training 20% testing

    Network = Network_library.Network()

    #Setting the right data for all the possible combinations of problems and classifiers

    if test_name == '5s':
        d.import_data('dataSorted_allOrientations.mat')
        if classifier == 'lin':
            d.build_train_labels_lin()
            d.build_test_labels_lin()
            
        elif classifier == 'log':
            d.build_train_labels_log()
            d.build_test_labels_log()

        else:
            print("This classifier is not supported for this test.")
            sys.exit(1)

        d.build_training_matrix()
        d.build_test_matrix()
        Network.L = 5

    elif test_name == 'lvr':
        if classifier == 'log' or classifier == '1nn':
            d.import_data('dataSorted_leftAndRight.mat')
            d.leftvsright_mixed()
            Network.L = 1

        else: 
            print("This classifier is not supported for this test.")
            sys.exit(1)

    else:
        print("This test does not exist.")
        sys.exit(1)

    #Filtering the data
    if filter_name not in d.spectral_bands.keys():
        print("The specified frequency band is not supported")
        sys.exit(1)

    d.training_data = d.filter_data(d.training_data,filter_name)
    d.test_data = d.filter_data(d.test_data,filter_name)


    #Computing the absolute value of the data, to get rid of negative numbers
    d.training_data = np.abs(d.training_data)
    d.test_data = np.abs(d.test_data)

    ########################
    # Define the network parameters
    ########################

    Network.T = d.training_data.shape[1] #Number of training time steps
    Network.n_min = 2540 #Number time steps dismissed
    Network.K = 128 #Input layer size
    Network.N = num_nodes #Reservoir layer size


    Network.u = d.training_data
    Network.y_teach = d.training_results

    Network.setup_network(d,num_nodes,input_probability,reservoir_probability,d.data.shape[-1])

    Network.train_network(d.data.shape[-1],classifier,d.num_columns, d.num_trials_train, d.train_labels, Network.N) 

    Network.mean_test_matrix = np.zeros([Network.N,d.num_trials_test,d.data.shape[-1]])

    Network.test_network(d.test_data, d.num_columns,d.num_trials_test, Network.N, d.data.shape[-1], t_autonom=d.test_data.shape[1])

    if classifier == 'lin':
        #print(f'Performance for {test_name} using {classifier} : {d.accuracy_lin(Network.regressor.predict(Network.mean_test_matrix.T),d.test_labels)}')
        return d.accuracy_lin(Network.regressor.predict(Network.mean_test_matrix.T),d.test_labels)
    elif classifier == 'log':
        #print(f'Performance for {test_name} using {classifier} : {Network.regressor.score(Network.mean_test_matrix.T,d.test_labels.T)}')
        return Network.regressor.score(Network.mean_test_matrix.T,d.test_labels.T)
    elif classifier == '1nn':
        #print(f'Performance for {test_name} using {classifier} : {Network.regressor.score(Network.mean_test_matrix.T,d.test_labels)}')
        return Network.regressor.score(Network.mean_test_matrix.T,d.test_labels)
    
def calculate_accuracy_variance(df, group_by_columns):
    """
    Calculate the variance of accuracy based on the specified group-by columns.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    group_by_columns (list): The columns to group by.

    Returns:
    DataFrame: A DataFrame with the variance of accuracy for each group.
    """
    # Group by the specified columns and calculate variance of the accuracy
    accuracy_variance = df.groupby(group_by_columns)['accuracy'].var()
    
    # Reset index to turn the groupby object back into a DataFrame
    accuracy_variance = accuracy_variance.reset_index()
    
    # Rename the variance column for clarity
    column_name = '_'.join(group_by_columns) + '_accuracy_variance'
    accuracy_variance.rename(columns={'accuracy': column_name}, inplace=True)
    
    return accuracy_variance

def plot_accuracy_variance(df, group_by_columns, title):
    """
    Plot the variance of accuracy based on the specified group-by columns.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    group_by_columns (list): The columns to group by.
    title (str): The title for the plot.
    """
    # Calculate the variance using the previously defined function
    variance_df = calculate_accuracy_variance(df, group_by_columns)
    
    # Determine the name of the variance column
    variance_column_name = '_'.join(group_by_columns) + '_accuracy_variance'
    
    # Plot the variance
    plt.figure(figsize=(10, 6))
    plt.plot(variance_df[group_by_columns], variance_df[variance_column_name], marker='o')
    
    plt.xlabel(' & '.join(group_by_columns))
    plt.ylabel('Accuracy Variance')
    plt.title(title)
    plt.grid(True)
    plt.show()
    
def calculate_max_accuracy(df, group_by_columns):
    """
    Calculate the maximum accuracy based on the specified group-by columns.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    group_by_columns (list): The columns to group by.

    Returns:
    DataFrame: A DataFrame with the maximum accuracy for each group.
    """
    # Group by the specified columns and calculate the maximum accuracy
    max_accuracy = df.groupby(group_by_columns)['accuracy'].max()
    
    # Reset index to turn the groupby object back into a DataFrame
    max_accuracy = max_accuracy.reset_index()
    
    # Rename the max column for clarity
    column_name = '_'.join(group_by_columns) + '_max_accuracy'
    max_accuracy.rename(columns={'accuracy': column_name}, inplace=True)
    
    return max_accuracy

def plot_max_accuracy(max_accuracy_df, group_by_columns):
    """
    Plot the maximum accuracy for the specified group-by columns.

    Parameters:
    max_accuracy_df (DataFrame): The DataFrame with the maximum accuracy data.
    group_by_columns (list): The columns that were grouped by.
    
    Returns:
    None
    """
    # Extract the column name for the maximum accuracy
    column_name = '_'.join(group_by_columns) + '_max_accuracy'
    
    plt.figure(figsize=(10, 6))
    
    if len(group_by_columns) == 1:
        # For single column grouping, create a line plot
        plt.plot(max_accuracy_df[group_by_columns[0]], max_accuracy_df[column_name], marker='o')
        plt.xlabel(group_by_columns[0])
    else:
        # For multiple column grouping, create a scatter plot
        # Assume the first column is the x-axis and the second column is the color/hue
        for label, df in max_accuracy_df.groupby(group_by_columns[1]):
            plt.scatter(df[group_by_columns[0]], df[column_name], label=label)
        plt.xlabel(group_by_columns[0])
        plt.legend(title=group_by_columns[1])

    plt.ylabel('Maximum Accuracy')
    plt.title(f'Maximum Accuracy grouped by {" and ".join(group_by_columns)}')
    plt.tight_layout()
    plt.show()