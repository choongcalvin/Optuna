# Optuna
Optuna hyperparameter tuning

Optuna is an open source hyperparameter optimization framework to automate hyperparameter search
This code will use Optuna to search through the hyperparameter space and find the best combination of parameters for the GradientBoostingRegressor model. 
The code will print the best parameters, as well as the mean absolute error on the test data.

A study is a container for the trials that are run during the optimization process.
Once you have created a study, you can use it to run trials. 
The trials are the individual runs of the optimization process. 
Each trial will try a different set of hyperparameters and the objective function 
will be evaluated on the test data. The best set of hyperparameters will be the set 
that results in the minimum value of the objective function.

The value of n_trials is the number of times that Optuna will try 
to find a better set of hyperparameters. In our code, we have set n_trials to 100. 
This means that Optuna will try 100 different combinations of hyperparameters to 
find the best set of hyperparameters.
The value of n_trials is a trade-off between the amount of time that you want to 
spend optimizing the hyperparameters and the quality of the results. 
If you set n_trials to a high value, Optuna will be able to search a wider range 
of hyperparameters and find a better set of hyperparameters. 
However, it will also take longer to run the optimization.
