# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 13:33:42 2021

@author: Raska
"""
import numpy as np
import pandas as pd
from numpy.random import default_rng
import matplotlib.pyplot as plt

### INPUT AND INITIALISATION

def input_config():
    """
    Input function obtain values of parameters from user

    Raises
    ------
    ValueError
        Invalid selection.

    Returns
    -------
    df : pandas dataframe
        Input dataset.
    ycol : string
        name of target column.
    xcol : string
        name of predictor column.
    model_config : string
        Configuration of model - options include 'std', 'std_nonlinear', and 'simple'.
    maxDegree : int
        Specified degree of dependence.
    n_iter : int
        Number of iterations.
    lr : float
        Learning rate.
    prop : float
        Proportion of data used as training.
    fname : string
        dataset filename.

    """
    model_configurations = np.array(['simple', 'std', 'std_nonlinear'])
    print('__________________________')
    print('Input selection according to values presented on the left-hand side.')
    data_selection = input('Select dataset for model generation: \n [1] Advertising.csv \n [2] Auto.csv \n \n >>')
    if data_selection == '1':
        fname = 'Advertising.csv'
        df = pd.read_csv(fname)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        print('Select target column from dataset:')
        selectionPrint(df.columns)
        target_selection = int(input(' >>'))-1
        ycol = df.columns[target_selection]
        print('Select model configuration:')
        selectionPrint(model_configurations)
        print('(simple, std, and std_nonlinear correspond to Tasks 1, 2, and 3)')
        config_selection = int(input(' >>'))-1
        model_config = model_configurations[config_selection]
        xcol = 1
        maxDegree = 1
        print('__________________________')
        print('Input numeric value.')
    elif data_selection == '2':
        fname = 'Auto.csv'
        df = pd.read_csv(fname)
        df = df[df.horsepower != '?']
        print('Select target column from dataset:')
        selectionPrint(df.columns[:-1])
        target_selection = int(input(' >>'))-1
        ycol = df.columns[target_selection]
        print('Select predictor column from dataset:')
        selectionPrint(df.columns[:-1])
        predictor_selection = int(input(' >>'))-1
        xcol = df.columns[predictor_selection]
        model_config = 'std'
        print('__________________________')
        print('Input numeric value.')
        maxDegree = int(input('Input number of degree dependence of predictor to target \n \n >>'))
    else:
        raise ValueError("Invalid selection!")
    
    n_iter = int(input('Input number of iterations: \n \n >>')) #100
    lr = float(input('Input learning rate: \n \n >>')) #1e-03
    prop = 0.8
    return df, ycol, xcol, model_config, maxDegree, n_iter, lr, prop, fname

def preprocess_data(df_features):
    """
    Wrapper function to split and normalise features.

    Parameters
    ----------
    df_features : pandas dataframe
        Dataframe of predictor features.

    Returns
    -------
    feature_list : list
        Nnrmalised and sorted list of features.

    """
    feature_list = split_Values(df_features)
    feature_list = normalise_Features(feature_list)
    return feature_list

def convert_listToArray(data_list, data_snippet):
    """
    Wrapper function to convert a list to array according to snippet size

    Parameters
    ----------
    data_list : list
        list of data.
    data_snippet : array
        snippet of data representing a sample single row in array.

    Returns
    -------
    data_array : array
        reshaped and converted array.

    """
    aa = np.concatenate(data_list, axis=0 )
    data_array = np.reshape(aa, [len(data_list), np.size(data_snippet)])
    return data_array

def shuffle_split(df, prop):
    """
    Wrapper function to shuffle and split dataset

    Parameters
    ----------
    df : pandas dataframe
        Dataset.
    prop : float
        Proportion of data allocated for training.

    Returns
    -------
    df_train : pandas dataframe
        Training data.
    df_test : pandas dataframe
        Testing data.

    """
    df = shuffle_Dataset(df)
    df_train, df_test = split_testTrain(df, prop)
    return df_train, df_test

def sort_dataset(df, ycol, xcol, maxDegree, model_config, fname):
    """
    Wrapper function to sort dataset

    Parameters
    ----------
    df : pandas dataframe
        Dataset.
    ycol : string
        Target column name.
    xcol : string
        Predictor column name.
    maxDegree : int
        Specified degree of dependence.
    model_config : string
        Configuration of model - options include 'std', 'std_nonlinear', and 'simple'.
    fname : string
        dataset filename.

    Returns
    -------
    df_features : Pandas dataframe
        Dataset of predictor features.
    data_actual : array
        Array of target values.

    """
    if fname == 'Advertising.csv':
        df_features, data_actual = sort_Dataset_adv(df, ycol, model_config = model_config)
    elif fname == 'Auto.csv':
        df_features, data_actual = sort_Dataset_aut(df, ycol, xcol, maxDegree)
    return df_features, data_actual

def initialise_Theta(df, fname, maxDegree, model_config = 'std', zeros = True):
    """
    Initialises Theta values with respect to model configuration

    Parameters
    ----------
    df : pandas DataFrame
        Input dataset.
    fname : string
        Filename of dataset.
    maxDegree : int
        Input maximum degree dependenve.
    model_config : string, optional
        Configuration of model - options include 'std', 'std_nonlinear', and 'simple'. The default is 'std'.
    zeros : Boolean, optional
        Option to set initialisation values as zero or random values within 0-1 GT. The default is True.

    Returns
    -------
    numpy array
        Allocated array of thetas. Can be substituted with other values.

    """
    if fname == 'Advertising.csv':
        if model_config == 'std' or model_config =='std_nonlinear':
            nTheta = df.shape[1]
        elif model_config == 'simple':
            nTheta = df.shape[1]-1
    elif fname == 'Auto.csv':
        nTheta = maxDegree+1
    if zeros==True:
        return np.zeros((nTheta))
    else:
        return np.random.rand(nTheta)
    
def sort_Dataset_adv(df, ycolumn_name, model_config= 'std'): 
    """
    Divides dataset into features and target for the advertising dataset

    Parameters
    ----------
    df : pandas DataFrame
        Input dataset.
    ycolumn_name : string
        Name of target column name.
    model_config : TYPE, optional
        Configuration of model - options include 'std', 'std_nonlinear', and 'simple'. The default is 'std'.

    Returns
    -------
    df_red : numpy array
        Array of features.
    y : numpy array
        Array of target column.

    """
    y = df[ycolumn_name]
    if model_config == 'std':
        df_red = df.drop([ycolumn_name], axis = 1)
    elif model_config == 'simple' :
        df_red = df.drop([ycolumn_name, 'newspaper'], axis = 1)
    elif model_config == 'std_nonlinear':
        df_red = df.drop([ycolumn_name, 'newspaper'], axis = 1)
        df_red['mult'] = df['TV']*df['radio']
    y = y.to_numpy()
    # print(df_red)
    df_red = df_red.to_numpy()
    return df_red, y

def sort_Dataset_aut(df, ycolumn_name, xcolumn_name, maxDegree):
    """
    Divides dataset into features and target for the auto dataset

    Parameters
    ----------
    df : pandas DataFrame
        Input dataset.
    ycolumn_name : string
        Name of target column name.
    xcolumn_name : string
        Name of predictor column name.
    maxDegree : int
        Input maximum degree dependence.

    Returns
    -------
    df_red : numpy array
        Array of features.
    y : numpy array
        Array of target column.

    """
    y = df[ycolumn_name]
    x = df[xcolumn_name].astype(float)
    maxDegree = maxDegree+1
    df_red = []
    for i in range(maxDegree):
        df_red.append(np.power(x, -i))
    y = y.to_numpy()
    df_red = np.asarray(df_red).T
    df_red = df_red[:, 1:]
    return df_red, y

def shuffle_Dataset(df):
    """
    Shuffles dataset

    Parameters
    ----------
    df : pandas dataframe
        Input dataset.

    Returns
    -------
    df : pandas dataframe
        Shuffled dataset.

    """
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    return df

def split_testTrain(df, prop):
    """
    Splits dataset into testing and training dataset

    Parameters
    ----------
    df : pandas dataframe
        Input dataset.
    prop : float
        Proportion of training dataset from full dataset used.

    Returns
    -------
    df_train : pandas dataframe
        Training dataset.
    df_test : pandas dataframe
        Testing dataset.

    """
    np.random.seed(0)
    mask = np.random.rand(len(df))<prop
    df_train = df[mask].reset_index(drop=True)
    df_test = df[~mask].reset_index(drop=True)
    return df_train, df_test

def split_Values(vals):
    """
    Splits big array into lists for use in training

    Parameters
    ----------
    vals : array
        Big array.

    Returns
    -------
    result : list
        list of arrays.

    """
    size = np.shape(vals)[1]
    result = np.split(vals, size, axis =1)
    return result

def normalise_Features(feature_list):
    """
    Normalises values in each feature column to be within 0 to 1.

    Parameters
    ----------
    feature_list : list
        list of raw features.

    Returns
    -------
    normalised_list : list
        list of normalised features.

    """
    normalised_list = []
    for feature in feature_list:
        numerator = feature - np.min(feature)
        denominator = np.max(feature) - np.min(feature)
        output = numerator/denominator
        normalised_list.append(output)
    return normalised_list

### TRAINING

def get_predictions(theta_vector_init, feature_list):
    """
    Gets predictions based on theta and feature values

    Parameters
    ----------
    theta_vector_init : numpy array
        Array of thetas that can be updated.
    feature_list : list
        List of predictor features.

    Returns
    -------
    predictions : numpy array
        Predicted values.

    """
    theta0 = theta_vector_init[0]
    theta_vector_reduced = theta_vector_init[1:]
    feature_sample = feature_list[0]
    list_size = np.shape(feature_sample)
    ytot = np.zeros(list_size)
    i = 0
    for i in range(len(feature_list)):
        ytot += theta_vector_reduced[i]*feature_list[i] 
        i+=1
    predictions = theta0 + ytot
    return predictions

def get_deltaThetaZero(predictions, y):
    """
    Get deltaThetaZero for use in ThetaZero calculations

    Parameters
    ----------
    predictions : numpy array
        Array of predictions.
    y : numpy array
        Array of target values.

    Returns
    -------
    result : float
        Derivative of theta with respect to the target array and predictions.

    """
    n = len(predictions)
    sum_component = predictions-y
    sum_component = np.sum(sum_component)
    result = 2/n*(sum_component)
    return result

def get_deltaTheta(predictions, y, feature):
    """
    Get delta Theta values for ThetaZero calculations

    Parameters
    ----------
    predictions : numpy array
        Array of predictions.
    y : numpy array
        Array of target values.
    feature : numpy array
        Array of predictor.

    Returns
    -------
    result : float
        Derivative of theta with respect to the target array, predictor array, and predictions.

    """
    n = len(predictions)
    sum_component = predictions-y
    sum_component = np.dot(sum_component, feature)
    sum_component = np.sum(sum_component)
    result = 2/n*(sum_component)
    return result

def get_newTheta(theta_old, lr, deltaTheta):
    """
    Uses delta Theta, learning rate, and previous value of Theta to obtain new Theta value

    Parameters
    ----------
    theta_old : float
        Old value of theta.
    lr : float
        Learning rate of gradient descent algorithm.
    deltaTheta : float
        Derivative of theta with respect to target value.

    Returns
    -------
    result : float
        New value of theta.

    """
    result = theta_old - lr*deltaTheta
    return result

def update_dThetaValues(predictions, y, feature_list):
    """
    Wrapper function that is used to obtain delta Theta values for a whole matrix. Calls get_deltaTheta and get_deltaThetaZero

    Parameters
    ----------
    predictions : numpy array
        Array of predictions.
    y : numpy array
        Array of target values.
    feature_list : list
        List of predictor feature arrays.

    Returns
    -------
    dTheta_list : list
        List of delta Theta values.

    """
    deltaThetaZero = get_deltaThetaZero(predictions, y)
    dTheta_list = []
    dTheta_list.append(deltaThetaZero)
    for feature in feature_list:
        dTheta = get_deltaTheta(predictions, y, feature)
        dTheta_list.append(dTheta)
    return dTheta_list

def update_ThetaValues(theta_vector, lr, deltaTheta_vector):
    """
    Wrapper function to obtain Theta values from delta Theta and learning rate.

    Parameters
    ----------
    theta_vector : numpy array
        Array of old theta values for a given number of predictors.
    lr : float
        Learning rate.
    deltaTheta_vector : numpy array
        DESCRIPTION.

    Returns
    -------
    theta_list : TYPE
        Array of theta values for a given number of predictors..

    """
    theta_list = []
    for (theta_vector, deltaTheta_vector) in zip(theta_vector, deltaTheta_vector):
        theta_new = (get_newTheta(theta_vector, lr, deltaTheta_vector))
        theta_list.append(theta_new)
    return theta_list

def get_costFunction(predictions, actual):
    """
    Obtains cost function of a given prediction array with respect to actual data.

    Parameters
    ----------
    predictions : numpy array
        Array of predicted values.
    actual : numpy array
        Array of target values.

    Returns
    -------
    result : float
        Value of cost function (in this case Mean Square Error) .

    """
    actual = np.reshape(actual, [np.size(actual), 1])
    result = predictions-actual
    result = result**2
    result = np.mean(result)
    return result


def generate_modelHistory(theta_vector, feature_list, data_actual, n_iter, lr):
    """
    Runs training and obtains training history.

    Parameters
    ----------
    theta_vector : numpy array
        Initialised list of thetas.
    feature_list : list
        list of predictor features.
    data_actual : numpy array
        list of target values.
    n_iter : int
        number of iterations.
    lr : float
        learning rate.

    Returns
    -------
    theta_list : numpy array
        List of thetas according to training history.
    delta_list : numpy array
        List of delta thetas according to training history.
    cost_list : numpy array
        List of MSE values according to training history.

    """
    theta_list =[]
    delta_list = []
    cost_list = []
    for i in range(n_iter):
        data_predictions = get_predictions(theta_vector, feature_list)
        deltaTheta_vector = update_dThetaValues(data_predictions, data_actual, feature_list)
        theta_vector = update_ThetaValues(theta_vector, lr, deltaTheta_vector)
        cost = get_costFunction(data_predictions, data_actual)
        delta_list.append(deltaTheta_vector)
        theta_list.append(theta_vector)
        cost_list.append(cost)
    theta_list = convert_listToArray(theta_list, theta_vector)
    delta_list = convert_listToArray(delta_list, deltaTheta_vector)  
    cost_list = np.array(cost_list)
    return theta_list, delta_list, cost_list

### EVALUATION

def get_rsquared(actual, predictions):
    """
    Calculates r-squared value to obtain measurement of best fit with regards to data and predictions

    Parameters
    ----------
    actual : numpy array
        Array of target values.
    predictions : numpy array
        Array of predicted values.

    Returns
    -------
    result : float
        R^2 value.

    """
    actual = np.reshape(actual, [np.size(actual), 1])
    numerator  = actual-predictions
    numerator = numerator**2
    numerator = np.sum(numerator)
    n = len(actual)
    denominator = n*get_variance(actual)
    result = 1 - (numerator/denominator)
    return result

def get_variance(values):
    """
    Obtains variance of data

    Parameters
    ----------
    values : numpy array
        Input data.

    Returns
    -------
    ele : float
        Variance of data.

    """
    ele = 0
    mean = np.mean(values)
    for i in range(len(values)):
        ele += (values[i]-mean)**2
    return ele

def get_minimumValue(array):
    """
    Wrapper function to get minimum value of array and its index

    Parameters
    ----------
    array : numpy array
        Any numpy array.

    Returns
    -------
    val : float
        Magnitude of minimum value.
    idx : int
        Index of minimum value in array.

    """
    val, idx = min((val, idx) for (idx, val) in enumerate(array))
    return val, idx

### PRINTERS - for UI and monitoring only
    
def processPrint(state1, state2 = "", state3 = "", state4 = "", state5 = ""):
    """
    Printer function for processes

    """
    print("[PROCESS]", state1, state2, state3, state4, state5)
    
def infoPrint(state1, state2 = "", state3 = "", state4 = "", state5 = "", state6 = ""):
    """
    Printer function for info

    """
    print("[INFO]", state1, state2, state3, state4, state5, state6)
    
def selectionPrint(selections):
    """
    Printer function for selections

    """
    selections = np.array(selections)
    selection_idx = np.arange(1, len(selections)+1)
    i = 0
    for selection_idx, selections in zip(selection_idx, selections):
        print(' [', selection_idx, ']', ' ', selections, sep = "")
        i+=1
        
def plot_3dregression(xy, z):
    from matplotlib import cm
    xy = np.array(xy)
    z = np.array(z)
    x = xy[:,0]
    y = xy[:,1]
    z = np.squeeze(z)
    # build the figure instance
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    mesh = ax.plot_trisurf(x, y, z, cmap=cm.binary, linewidth=0.5)
    ax.scatter(x, y, z, color='r')
    
    # set your labels
    ax.set_xlabel('TV')
    ax.set_ylabel('Radio')
    ax.set_zlabel('Sales')
    plt.colorbar(mesh)

    
    