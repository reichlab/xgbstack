### Estimate function that gives model weights based on observed inputs

### WE CAN DROP THE INDEX j FROM DISCUSSION HERE, SUMMING BINS ELSEWHERE

### MAKE MATH IN VIGNETTE

### Objective function to maximize is the log score:
### sum_k log{sum_j prob(bin_kj)} = sum_k log{sum_j sum_i w_ik prob_i(bin_kj)}
###   = sum_k log[sum_i  w_ik * {sum_j prob_i(bin_kj)}]     (1)
### k indexes combinations of time at which a prediction is made,
### j indexes the bins adjacent to the observed bin, used in calculating log scores, and
### i indexes the component models;
### prob_i(bin_kj) denotes the probability assigned to bin j at time(s) k by model i.
### w_ik is the weight assigned to model i for observation k, computed as
### w_ik = exp(rho_ik)/(sum_i' exp(rho_i'k)), where
### rho_ik is a function of covariates observed for observation k
### time a prediction is for (which may be each week within the season or just the season)

### In practice, we will implement the calculation of (1) using the logspace summation operator:
### logsum_i a_i = log(sum_i exp(a_i))
### With this notation, continuing from (1), we have
###  (1) = sum_k logsum_i log[w_ik * {sum_j prob_i(bin_kj)}]
###  = sum_k logsum_i [log(w_ik) + log{sum_j prob_i(bin_kj)}]
###  = sum_k logsum_i [log(w_ik) + logsum_j log{prob_i(bin_kj)}].
### Further, we compute log(w_ik) = log[exp(rho_ik)/{sum_i' exp(rho_i'k)}]
###  = log{exp(rho_ik)} - log{sum_i' exp(rho_i'k)}
###  = rho_ik - (logsum_i' rho_i'k)

### We also need to calculate the first and second order derivatives of the
### loss function with respect to the predicted values (rho_ik) at each observation
### (e.g., see https://www.r-bloggers.com/an-introduction-to-xgboost-r-package/
### https://github.com/dmlc/xgboost/blob/ef4dcce7372dbc03b5066a614727f2a6dfcbd3bc/src/objective/multiclass_obj.cc
### https://github.com/dmlc/xgboost/tree/ef4dcce7372dbc03b5066a614727f2a6dfcbd3bc/plugin/example)
### There is a separate gradient calculation for each pair of indices k and i
### in Equation (1)
### partial term_k / partial rho_ik
###  = 



softmax <- function(x) {
  log_denom <- logspace_sum(x)
  return(exp(x - log_denom))
}

#' take preds, a vector in 
## From line 56 at https://github.com/dmlc/xgboost/blob/ef4dcce7372dbc03b5066a614727f2a6dfcbd3bc/src/objective/multiclass_obj.cc,
## it appears that preds is stored in column-major order with
## observations in columns and classes/models in rows
## i.e., preds[k * nclass + i] is prediction for model i at index k
## (note that our use of i and k is exactly reversed from theirs)
## first, convert preds to matrix
## calculate num_obs like this instead of storing since num_obs depends on
## the value of subsample argument to xgb.train
#' 
#' @return preds in matrix form, with num_models columns and num_obs rows
preds_to_matrix <- function(preds, num_models) {
  num_obs <- length(preds) / num_models
  dim(preds) <- c(num_models, num_obs)
  return(t(preds))
#  dim(preds) <- c(num_obs, num_models)
#  return(preds)
}

#' A factory-esque arrangement to manufacture an objective function with
#' needed quantities accessible in its parent environment.
get_obj_fn <- function(component_model_log_scores) {
  ## evaluate arguments so that they're not just empty promises
  component_model_log_scores
  
  ## create function to calculate objective
  obj_fn <- function(preds, dtrain) {
    ## convert preds to matrix form with one row per observation and one column per component model
    preds <- preds_to_matrix(preds = preds, num_models = ncol(component_model_log_scores))
    
    log_weights_denom <- logspace_sum_matrix_rows(preds)
    
    log_weights <- sweep(preds, 1, log_weights_denom, `-`)
    
    ## adding log_weights and component_model_log_scores gets
    ## log(pi_mi) + log(f_m(y_i | x_i)) = log(pi_mi * f_m(y_i | x_i))
    ## in cell [i, m] of the result.  logspace sum the rows to get a vector with
    ## log(sum_m pi_mi * f_m(y_i | x_i)) in position i.
    ## sum that vector to get the objective.
    return(-1 * sum(logspace_sum_matrix_rows(log_weights + component_model_log_scores)))
  }
  
  ## return function to calculate objective
  return(obj_fn)
}

#' A factory-esque arrangement to manufacture an function to calculate first and
#' second order derivatives of the objective function, with
#' needed quantities accessible in its parent environment.
get_obj_deriv_fn <- function(component_model_log_scores) {
  ## evaluate arguments so that they're not just empty promises
  component_model_log_scores
  
  ## create function to calculate objective
  obj_deriv_fn <- function(preds, dtrain) {
    ## convert preds to matrix form with one row per observation and one column per component model
    preds <- preds_to_matrix(preds = preds, num_models = ncol(component_model_log_scores))
    
    log_weights <- sweep(preds, 1, logspace_sum_matrix_rows(preds), `-`)
    
    ## adding preds and component_model_log_scores gets
    ## log(pi_mi) + log(f_m(y_i | x_i)) = log(pi_mi * f_m(y_i | x_i))
    ## in cell [i, m] of the result.  logspace sum the rows to get a vector with
    ## log(sum_m pi_mi * f_m(y_i | x_i)) in position i.
    ## sum that vector to get the objective.
    log_weighted_scores <- log_weights + component_model_log_scores
    log_weighted_score_sums <- logspace_sum_matrix_rows(log_weighted_scores)
    
    ## calculate gradient
    ## is there a way to do exponentiation last in the lines below,
    ## instead of in the calculations of term1 and term2?
    ## think i may need to vectorize logspace_sub?
    grad_term1 <- exp(sweep(log_weighted_scores, 1, log_weighted_score_sums, `-`))
    grad_term2 <- exp(log_weights)
    grad <- grad_term1 - grad_term2
#    print(grad)
    grad <- as.vector(t(grad))
#    grad <- as.vector(grad)
    
    ## calculate hessian
    hess <- grad_term1 - grad_term1^2 - grad_term2 + grad_term2^2
    hess <- as.vector(t(hess))
#    hess <- as.vector(hess)
    
    ## return
    return(list(grad = -1 * grad, hess = -1 * hess))
#    return(list(grad = grad, hess = hess))
  }
  
  ## return function to calculate derivatives of objective
  return(obj_deriv_fn)
}
