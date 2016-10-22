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

## objective function for call to xgboost to do estimation of model weights
## actually only calculate gradient and (diagonal elements of) Hessian
test_obj <- function(preds, dtrain) {
  ## Convert "preds" to weights.
  ## From line 56 at https://github.com/dmlc/xgboost/blob/ef4dcce7372dbc03b5066a614727f2a6dfcbd3bc/src/objective/multiclass_obj.cc,
  ## it appears that preds is stored in column-major order with
  ## observations in columns and classes/models in rows
  ## i.e., preds[k * nclass + i] is prediction for model i at index k
  ## (note that our use of i and k is exactly reversed from theirs)
  
  ## first, convert preds to matrix
  ## calculate num_obs like this instead of storing since num_obs depends on
  ## the value of subsample argument to xgb.train
  num_models <- get("num_models", envir = storage_env)
  num_obs <- length(preds) / num_models
  dim(preds) <- c(num_models, num_obs)
  
  weights <- matrix(NA, nrow = num_models, ncol = num_obs)
  for(k in seq_len(num_obs)) {
    weights[, k] <- softmax(preds[, k])
  }
  
  ## calculate preliminary quantities used in both grad and hess
  prelim_quantities <- calc_obj_prelim()
  
  ## calculate gradient
  grad <- calc_obj_grad()
  
  ## calculate Hessian
  ## Based on lines 66 - 71 of https://github.com/dmlc/xgboost/blob/ef4dcce7372dbc03b5066a614727f2a6dfcbd3bc/src/objective/multiclass_obj.cc#L5
  ## it looks like we only need the diagonal elements of the Hessian
  hess <- calc_obj_hess()
  
  # return
  return(list(grad = grad, hess = hess))
}

calc_obj_prelim <- function() {
  
}

calc_obj_grad <- function() {
  
}

calc_obj_hess <- function() {
  
}
