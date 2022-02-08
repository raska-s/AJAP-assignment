# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 13:35:42 2021

@author: Raska
"""

# to use the negative powers edit sort_dataset_aut and plot term to use -i and **-1

from functions import *

# Print header and input configurations
print('=INPUT PAGE=')
df, ycol, xcol, model_config, maxDegree, n_iter, lr, prop, fname  = input_config()
print('== Multilinear Gradient Descent Algorithm ==')
infoPrint('MODEL CONFIGURATIONS')
print('Dataset:', fname, ' \nConfiguration:', model_config, '\nTarget:', ycol)
if fname =='Auto.csv':
    print('Predictor: ', xcol, '\nDegree of dependence:', maxDegree)
print('No. if iterations: ', n_iter, '\nLearning rate: ', lr)
print()



## Initialise values
processPrint('Initialising values ...')
df_train, df_test = shuffle_split(df, prop)
df_features, data_actual = sort_dataset(df_train, ycol, xcol, maxDegree, model_config, fname)
feature_list = preprocess_data(df_features)
theta_vector = initialise_Theta(df, fname = fname, maxDegree = maxDegree, model_config = model_config, zeros = True)

# Generate model
processPrint('Training model ...')
theta_list, delta_list, cost_list = generate_modelHistory(theta_vector, feature_list, data_actual, n_iter,lr)
min_cost, idx_cost = get_minimumValue(cost_list)
final_theta = theta_list[idx_cost,:]

# Test predictions
processPrint('Model trained. Obtaining best theta-parameters ...')
df_features_test, data_actual_test = sort_dataset(df_test, ycol, xcol, maxDegree, model_config, fname)
feature_test = preprocess_data(df_features_test)
final_predictions = get_predictions(final_theta, feature_test)
final_rsquared = get_rsquared(data_actual_test, final_predictions)

# Displaying final output
print('\n=FINAL OUTPUT=')
infoPrint('Best values obtained after', idx_cost, '/', n_iter, 'iterations')
infoPrint('Theta parameters:', final_theta)
infoPrint('Cost function:', min_cost)
infoPrint('R-squared value:', final_rsquared)

# # Plotting results
axis = np.arange(0, len(final_predictions))
# import matplotlib.pyplot as plt

if model_config =='simple':
    plot_3dregression(df_features_test, final_predictions)
    plt.title('Advertising dataset regression function plane')
    plt.show()
    
    plot_3dregression(df_features_test, data_actual_test)
    plt.title('Advertising dataset true values')
    plt.show()

if fname == 'Auto.csv':
    fig = plt.figure()
    plt.scatter(df_features_test[:, 0]**-1, final_predictions, marker = 'o', color = 'b', label = 'Predicted values')
    plt.scatter(df_features_test[:, 0]**-1, data_actual_test, marker = 'x', color = 'r', label = 'Actual values')
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.legend()
    plt.title('Predicted vs actual data of test set')
    
    
fig = plt.figure()
plt.plot(axis, final_predictions, '--', linewidth=0.7, label = 'Predicted values')
plt.scatter(axis, data_actual_test, marker='x', color='r', label = 'Actual values')
plt.xlabel('Iterations')
plt.ylabel(ycol)
plt.legend()
plt.title('Predicted vs actual data of test set')
plt.show()

fig = plt.figure()
plt.plot(theta_list)
plt.title('Value of theta with respect to iterations')
plt.xlabel('Iterations')
plt.ylabel('Theta')
plt.show()

fig = plt.figure()
plt.plot(delta_list)
plt.title('Value of delta Theta with respect to iterations')
plt.xlabel('Iterations')
plt.ylabel('delta Theta')
plt.show()

fig = plt.figure()
plt.plot(cost_list)
plt.title('Value of cost function with respect to iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost function')
plt.show()

