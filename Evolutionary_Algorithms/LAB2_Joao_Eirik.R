# First, install and load the required packages
#install.packages("caTools")
#install.packages("caret")
#install.packages("nnet")
#install.packages("cmaes")
library(caTools)
library(caret)
library(cmaesr)
library(nnet)
library(GA)

# Preprocessing -----------------------------------------------------------

# Function to preprocess the California Housing dataset
preprocess_housing_data <- function() {
  # Read the data into a data frame
  housing_data <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", 
                           header = F, 
                           sep = "", 
                           dec = ".")
  
  # The data does not have column names, so add them manually
  colnames(housing_data) <- c("CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV")
  
  # Split the data into training, validation, and test sets
  set.seed(123)  # Set seed for reproducibility
  set.seed(123)  # Set seed for reproducibility
  
  fractionTraining   <- 0.80
  fractionValidation <- 0.10
  fractionTest       <- 0.10
  
  # Compute sample sizes.
  sampleSizeTraining   <- floor(fractionTraining   * nrow(housing_data))
  sampleSizeValidation <- floor(fractionValidation * nrow(housing_data))
  sampleSizeTest       <- floor(fractionTest       * nrow(housing_data))
  
  # avoid overlapping subsets of indices.
  indicesTraining    <- sort(sample(seq_len(nrow(housing_data)), size=sampleSizeTraining))
  indicesNotTraining <- setdiff(seq_len(nrow(housing_data)), indicesTraining)
  indicesValidation  <- sort(sample(indicesNotTraining, size=sampleSizeValidation))
  indicesTest        <- setdiff(indicesNotTraining, indicesValidation)
  
  # Finally, output the three dataframes for training, validation and test.
  dfTraining   <- housing_data[indicesTraining, ]
  dfValidation <- housing_data[indicesValidation, ]
  dfTest       <- housing_data[indicesTest, ]
  
  
  # Normalize the data
  preprocess <- preProcess(dfTraining, method = "range")  # Change the method to "range"
  train_data_norm <- predict(preprocess, dfTraining)
  validation_data_norm <- predict(preprocess, dfValidation)
  test_data_norm <- predict(preprocess, dfTest)
  
  # Create the final variables
  X_train <- train_data_norm[, -ncol(train_data_norm)]  # all columns except the last one
  y_train <- train_data_norm[, ncol(train_data_norm)]   # last column
  
  X_valid <- validation_data_norm[, -ncol(validation_data_norm)]
  y_valid <- validation_data_norm[, ncol(validation_data_norm)]
  
  X_test <- test_data_norm[, -ncol(test_data_norm)]
  y_test <- test_data_norm[, ncol(test_data_norm)]
  
  # Return the final variables
  return(list(X_train = X_train, y_train = y_train, X_valid = X_valid, y_valid = y_valid,
              X_test = X_test, y_test = y_test))
}


# Validation  ----------------------------------------------------------------

get_performance <- function(t1, model){
  
  
  # stop the timer
  t2 = Sys.time()
  
  # calculate the time taken to train the model
  train_time = t2 - t1
  
  # start the timer
  t1 = Sys.time()
  
  # perform validation
  y_pred_valid = predict(model, X_valid)
  mse_valid = mean((y_pred_valid - y_valid) ^ 2)
  
  # stop the timer
  t2 = Sys.time()
  
  # calculate the time taken to perform validation
  valid_time = t2 - t1
  
  # start the timer
  t1 = Sys.time()
  
  # perform test
  y_pred_test = predict(model, X_test)
  mse_test = mean((y_pred_test - y_test) ^ 2)
  
  # stop the timer
  t2 = Sys.time()
  
  # calculate the time taken to perform test
  test_time = t2 - t1
  return(list(train_time=train_time, mse_valid=mse_valid, valid_time=valid_time, mse_test=mse_test, test_time=test_time))
}
print_result <- function(results){
  # print the data frame
  print(results)
  
  # find the row index of the maximum mse_valid value
  row_index = which.min(results$mse_valid)
  
  # print the row with the highest mse_valid value
  print(results[row_index, ])
}

# BP ---------------------------------------------------------------------

bp_experiments <- function(){
  # initialize an empty data frame to store the results
  bp_results = data.frame()
  
  # train the model with different sizes, alphas, and momentum values
  alphas = c(0.01, 0.1)
  momentums = c(0.7, 0.8, 0.9)
  
  for (size in sizes) {
    for (alpha in alphas) {
      for (momentum in momentums) {
        # start the timer
        t1 = Sys.time()
        
        # train the model with the specified momentum value
        bp_model = nnet(X_train, y_train, size = size, decay = alpha, momentum = momentum, 
                     act.fct = "logsig", linout = FALSE)
    
        performance = get_performance(t1, bp_model)
  
        
        # store the results in the data frame
        bp_results = rbind(bp_results, data.frame(size = size, alpha = alpha, momentum = momentum,
                                            train_time = performance$train_time, valid_time = performance$valid_time,
                                            test_time = performance$test_time, mse_valid = performance$mse_valid,
                                            mse_test = performance$mse_test))
      }
    }
  }
  return(bp_results)
}
# GA  ---------------------------------------------------------------------

ga_experiments <- function(maxiter){
  
  
  # initialize an empty data frame to store the results
  results_ga = data.frame()
  
  # train the model
  
  pop_sizes = c(50, 100, 200, 400)
  mutation_rates = c(0.1, 0.2)
  
  # Function to evaluate the fitness of a neural network using GA
  nn_ga_fitness <- function(w) {
    
    model <- nnet(X_train, y_train, size = size, Wts = w, act.fct = "logsig", linout = FALSE, maxit = 0, trace=FALSE)
    # Make predictions on the testing data
    y_pred <- predict(model, X_test)
    
    # Calculate the MSE on the testing data
    mse <- mean((y_pred - y_test) ^ 2)
    # Return the MSE as the fitness value
    return(-mse)
  }
  
  for (size in sizes) {
    for (pop_size in pop_sizes) {
      for (mutation_rate in mutation_rates) {
        print(cat("Size:",size,"pop_size:",pop_size,"mutation_rate:",mutation_rate))
        # start the timer
        nParams = (nFeatures+2)*size+1
        t1 = Sys.time()
        # train the model with the specified momentum value
        ga_result <- ga(type = "real-valued", fitness = nn_ga_fitness, lower = rep(-2, nParams), upper = rep(-1, nParams), pmutation = mutation_rate, popSize = pop_size, maxiter = maxiter
                        , run = 10)
        ga_weights <- ga_result@solution[1,]
        ga_model <- nnet(X_train, y_train, size = size, act.fct = "logsig", Wts = ga_weights, linout = FALSE, maxit = 0, trace=FALSE)
        performance = get_performance(t1, ga_model)
        
        # store the results in the data frame
        results_ga = rbind(results_ga, data.frame(size = size, pop_size = pop_size , mutation_rate = mutation_rate,
                                               train_time = performance$train_time, valid_time = performance$valid_time,
                                               test_time = performance$test_time, generations = length(ga_result@summary)/length(ga_result@summary[1,]), mse_valid = performance$mse_valid,
                                               mse_test = performance$mse_test))
        #Save the best model for plot
        if(size == 5 & pop_size == 400 & mutation_rate == 0.2){
          best_ga = ga_result
        }
        
      }
    }
  }
  
  return(list(results_ga = results_ga, best_ga = best_ga))
}

# ES  ---------------------------------------------------------------------

es_experiments <- function(maxiter){
  # initialize an empty data frame to store the results
  es_results = data.frame()
  
  # train the model with different sizes, alphas, and momentum values
  offsprings = c(50, 100, 200, 400)
  step_sizes = c(0.02, 0.01)
  
  for (size in sizes) {
    nParams = (nFeatures+2)*size+1
    
    # Function to evaluate the fitness of a neural network using GA
    objective.fun = makeSingleObjectiveFunction(
      name = "ES",
      fn = function(x) {
        # Train the neural network using the weights
        model <- nnet(X_train, y_train, size = size, Wts = x,act.fct = "logsig", linout = FALSE, maxit = 0, trace=FALSE)
        
        # Make predictions on the testing data
        y_pred <- predict(model, X_train)
        
        # Calculate the MSE on the testing data
        mse <- mean((y_pred - y_train) ^ 2)
        
        # Return the MSE as the fitness value
        return(mse)
      }
      ,
      par.set = makeNumericParamSet("x", len = nParams, lower = -1, upper = 1)
      
    )
    
    for (offspring in offsprings) {
      for (step_size in step_sizes) {
        print(cat("Size:",size,"offsprings:",offspring,"step_sizes:",step_size))
        # start the timer
        t1 = Sys.time()
        
        # train the model 
        es_result = cmaes(
          objective.fun, 
          monitor = makeSimpleMonitor(),
          control = list(
            sigma = step_size, # initial step size
            lambda = offspring, # number of offspring
            stop.ons = c(
              list(stopOnMaxIters(maxiter)), # stop after 100 iterations
              getDefaultStoppingConditions() # or after default stopping conditions
            )
          )
        )
        es_weights <- es_result$best.param
        es_model <- nnet(X_train, y_train, size = size, act.fct = "logsig", Wts = es_weights, linout = FALSE, maxit = 0)
        performance = get_performance(t1, es_model)
        
        # store the results in the data frame
        es_results = rbind(es_results, data.frame(size = size, offsprings = offspring , step_size = step_size,
                                            train_time = performance$train_time, valid_time = performance$valid_time,
                                            test_time = performance$test_time, mse_valid = performance$mse_valid,
                                            mse_test = performance$mse_test))
        #Save the best model for plot
        if(size == 5 & offspring == 200 & step_size == 0.01){
          best_es = es_result
        }
      }
    }
  }
  
  return(es_results)
}
# Main  -------------------------------------------------------------------

# Call the function to preprocess the California Housing dataset
housing_data <- preprocess_housing_data()

# Access the final variables
X_train <- housing_data$X_train
y_train <- housing_data$y_train

X_valid <- housing_data$X_valid
y_valid <- housing_data$y_valid

X_test <- housing_data$X_test
y_test <- housing_data$y_test

nFeatures = ncol(housing_data$X_train)

sizes = c(5, 10, 20, 30, 40, 50, 60) # The layer size of the ANN

# Run the experiments
bp_results <- bp_experiments() #Backpropagation
ga_results <- ga_experiments(100) #genetic algorithm Takes time. (100 generations)
es_results <- es_experiments(100) #evolutionary strategy. Takes even more time (100 iterations)

# Print and plot dataframes showing the each parameter combination for each model and plot the best GA model
print_result(bp_results)
print_result(ga_results$results_ga)
plot(ga_results$best_ga)
print_result(es_results)
