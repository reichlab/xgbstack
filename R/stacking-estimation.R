### Estimate function that gives model weights based on observed inputs

#' Convert vector of predictions to matrix format
#' From line 56 at https://github.com/dmlc/xgboost/blob/ef4dcce7372dbc03b5066a614727f2a6dfcbd3bc/src/objective/multiclass_obj.cc,
#' it appears that preds is stored in column-major order with
#' observations in columns and classes/models in rows
#' i.e., preds[k * nclass + i] is prediction for model i at index k
#' 
#' @param preds vector of predictions as obtained from xgb.train
#' @param number of models
#' 
#' @return preds in matrix form, with num_models columns and num_obs rows
preds_to_matrix <- function(preds, num_models) {
  num_obs <- length(preds) / num_models
  dim(preds) <- c(num_models, num_obs)
  return(t(preds))
}

#' A factory-esque arrangement (not sure if there is an actual name for this
#' pattern) to manufacture an objective function with needed quantities
#' accessible in its parent environment.  We do this becaues there's no way to
#' use the info attribute of the dtrain object to store the component model
#' log scores (as would be standard in the xgboost package).  But we need to
#' ensure that the objective function has access to these log scores when
#' called in parallel.
#' 
#' @param component_model_log_scores N by M matrix where entry [i, m] has the
#'   log-score for observation i obtained from component model m
#' 
#' @return a function that takes two arguments (preds and dtrain) and computes
#'   the log-score objective function for the stacked model.  i.e., it converts
#'   preds to model weights for each component model m at each observation i
#'   and then combines the component model log scores with those weights to
#'   obtain stacked model log scores
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

#' A factory-esque arrangement (not sure if there is an actual name for this
#' pattern) to manufacture an function to calculate first and second order
#' derivatives of the log-score objective, with needed quantities
#' accessible in its parent environment.  We do this becaues there's no way to
#' use the info attribute of the dtrain object to store the component model
#' log scores (as would be standard in the xgboost package).  But we need to
#' ensure that the objective function has access to these log scores when
#' called in parallel.
#' 
#' @param component_model_log_scores N by M matrix where entry [i, m] has the
#'   log-score for observation i obtained from component model m
#' 
#' @return a function that takes two arguments (preds and dtrain) and computes
#'   the first and second order derivatives of the log-score objective function
#'   for the stacked model.  i.e., it converts preds to model weights for each
#'   component model m at each observation i and then combines the component
#'   model log scores with those weights to obtain stacked model log scores.
#'   See the package vignette for derivations of these calculations.  This
#'   function is suitable for use as the "obj" function in a call to
#'   xgboost::xgb.train
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
    grad <- as.vector(t(grad))

    ## calculate hessian
    hess <- grad_term1 - grad_term1^2 - grad_term2 + grad_term2^2
    hess <- as.vector(t(hess))

    ## return
    return(list(grad = -1 * grad, hess = -1 * hess))
  }
  
  ## return function to calculate derivatives of objective
  return(obj_deriv_fn)
}

#' Fit a stacking model given a measure of performance for each component model
#' on a set of training data, and a set of covariates to use in forming
#' component model weights
#' 
#' @param formula a formula describing the model fit.  left hand side should
#'   give columns in data with scores of models, separated by +.  right hand
#'   side should specify explanatory variables.
#' @data a data frame with variables in formula
#' 
#' @return an estimated xgbstack object, which contains a gradient tree boosted
#'   fit mapping observed variables to component model weights
xgbstack <- function(formula, data) {
  formula <- Formula::Formula(formula)
  
  ## response, as a matrix of type double
  model_scores <- Formula::model.part(formula, data = data, lhs = 1) %>%
    as.matrix() %>%
    `storage.mode<-`("double")
  
  ## predictors, in format used in xgboost
  dtrain <- xgb.DMatrix(
    data = Formula::model.part(formula, data = data, rhs = 1) %>%
      as.matrix() %>%
      `storage.mode<-`("double")
  )
  
  ## get a function to compute first and second order derivatives of objective
  ## function
  obj_deriv_fn <- get_obj_deriv_fn(component_model_log_scores = model_scores)
  
  fit <- xgb.train(
    params = list(
      booster = "gbtree", # change to gblinear to fit lines
      subsample = 1, # use half of the observations in each boosting iteration
      colsample_bytree = 1, # use all of the columns to do prediction (for now, only one column...)
      max_depth = 10,
      min_child_weight = 0,
      gamma = 0,
      num_class = ncol(model_scores)
    ),
    data = dtrain,
    nrounds = 1000,
    obj = obj_deriv_fn,
    verbose = 0
  )
  
  ## return -- should really return an object with call, formula, etc.
  return(fit)
}
