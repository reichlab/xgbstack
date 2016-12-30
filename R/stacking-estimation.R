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

#' Function to compute (log) weights of component models from the "predictions"
#' output by xgboost
#' 
#' @param preds predictions from xgboost in matrix form, with num_models columns
#'   and num_obs rows
#' @param log boolean; return log of component model weights or just the weights
#' 
#' @return a matrix with num_models columns and num_obs rows, with (log) weight
#'   for model m at observation i in entry [i, m]
compute_model_weights_from_preds <- function(preds, log = FALSE) {
  log_weights <- sweep(preds, 1, logspace_sum_matrix_rows(preds), `-`)
  
  if(log) {
    return(log_weights)
  } else {
    return(exp(log_weights))
  }
}

#' Compute (log) weights of component models based on an xgbstack fit and
#' new data.
#' 
#' @param xgbstack_fit a fit xgbstack object
#' @param newdata new x data
#' @param log boolean: return log of weights or original weights?
#' 
#' @return an n_obs by num_models matrix of (log) weights
compute_model_weights <- function(xgbstack_fit, newdata, log = FALSE) {
  if(!identical(class(xgbstack_fit), "xgbstack")) {
    stop("xgbstack_fit must be an object of type xgbstack!")
  }
  
  ## convert newdata to format used in xgboost
  newdata_matrix <- Formula::model.part(xgbstack_fit$formula, data = newdata, rhs = 1) %>%
    as.matrix() %>%
    `storage.mode<-`("double")
  newdata <- xgb.DMatrix(data = newdata_matrix)
  
  ## get something proportional to log(weights)
  xgb_fit <- xgb.load(xgbstack_fit$fit)
  preds <- predict(xgb_fit, newdata = newdata)
  
  ## convert to weights
  preds <- preds_to_matrix(preds, num_models = xgbstack_fit$num_models)
  model_weights <- compute_model_weights_from_preds(preds, log = log)
  
  ## set column names
  colnames(model_weights) <-
    strsplit(as.character(xgbstack_fit$formula)[2], " + ", fixed = TRUE)[[1]]
  
  ## return
  return(model_weights)
}

#' A factory-esque arrangement (not sure if there is an actual name for this
#' pattern) to manufacture an objective function with needed quantities
#' accessible in its parent environment.  We do this because there's no way to
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
    
    ## Compute log of component model weights at each observation
    log_weights <- compute_model_weights_from_preds(preds, log = TRUE)
    
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
get_obj_deriv_fn <- function(component_model_log_scores, dtrain_Rmatrix) {
  ## evaluate arguments so that they're not just empty promises
  component_model_log_scores
  if(!missing(dtrain_Rmatrix)) {
    dtrain_Rmatrix
  }
  
  ## create function to calculate objective
  obj_deriv_fn <- function(preds, dtrain) {
    ## convert preds to matrix form with one row per observation and one column per component model
    preds <- preds_to_matrix(preds = preds, num_models = ncol(component_model_log_scores))
    
    ## Compute log of component model weights at each observation
    log_weights <- compute_model_weights_from_preds(preds, log = TRUE)
    
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
xgbstack <- function(formula,
  data,
  booster = "gbtree",
  subsample = 1,
  colsample_bytree = 1,
  max_depth = 10,
  min_child_weight = -10^10,
  gamma = 0,
  lambda = 0,
  nrounds = 10,
  cv_params = NULL,
  cv_folds = NULL,
  cv_nfolds = 10L,
  nthread) {
  formula <- Formula::Formula(formula)
  
  ## response, as a matrix of type double
  model_scores <- Formula::model.part(formula, data = data, lhs = 1) %>%
    as.matrix() %>%
    `storage.mode<-`("double")
  
  ## predictors, in format used in xgboost
  dtrain_Rmatrix <- Formula::model.part(formula, data = data, rhs = 1) %>%
    as.matrix() %>%
    `storage.mode<-`("double")
  dtrain <- xgb.DMatrix(
    data = dtrain_Rmatrix
  )
  
  base_params <- list(
    booster = booster,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    max_depth = max_depth,
    min_child_weight = min_child_weight,
    gamma = gamma,
    lambda = lambda,
    num_class = ncol(model_scores)
  )
  if(!missing(nthread)) {
    base_params$nthread <- nthread
  }
  
  if(is.null(cv_params)) {
    ## no cross-validation for parameter selection
    ## use base_params
    params <- base_params
  } else {
    ## estimation of some parameters via cross validation was specified
    cv_results <- expand.grid(cv_params, stringsAsFactors = TRUE)
    
    ## if they weren't provided, get sets of observations for cv folds
    ## otherwise, set cv_nfolds = number of folds provided
    if(is.null(cv_folds)) {
      cv_fold_memberships <- cut(seq_len(nrow(dtrain_Rmatrix)), cv_nfolds) %>%
        as.numeric() %>%
        sample(size = nrow(dtrain_Rmatrix), replace = FALSE)
      cv_folds <- lapply(seq_len(cv_nfolds),
        function(fold_ind) {
          which(cv_fold_memberships == fold_ind)
        }
      )
    } else {
      cv_nfolds <- length(cv_folds)
    }
    
    ## if nrounds is a parameter to choose by cross-validation,
    ## only fit the models with largest number of nrounds,
    ## then get predictions with fewer rounds by using ntreelimit argument
    cv_results <- cbind(cv_results,
      matrix(NA, nrow = nrow(cv_results), ncol = cv_nfolds) %>%
        `colnames<-`(paste0("cv_log_score_fold_", seq_len(cv_nfolds)))
    )
    if("nrounds" %in% names(cv_params)) {
      model_inds_to_fit <- which(cv_results$nrounds == max(cv_params$nrounds))
    } else {
      model_inds_to_fit <- seq_len(nrow(cv_results))
    }
    
    for(cv_ind in model_inds_to_fit) {
      params <- cv_results[cv_ind, ]
      attr(params, "out.attrs") <- NULL
      params <- as.list(params)
      params <- c(params,
        base_params[!(names(base_params) %in% names(params))])
      
      ## get all rows for parameter combinations where everything other than
      ## nrounds matches what is in the current set of parameters
      if("nrounds" %in% names(cv_params)) {
        if(length(cv_params) == 1) {
          similar_param_rows <- seq_len(nrow(cv_results))
        } else {
          similar_param_rows <- which(sapply(seq_len(nrow(cv_results)),
            function(possible_ind) {
              all(cv_results[possible_ind, -(colnames(cv_results) == "nrounds")] == cv_results[cv_ind, -(colnames(cv_results) == "nrounds")], na.rm = TRUE)
            }
          ))
        }
      } else {
        similar_param_rows <- cv_ind
      }
      
      ## for each k = 1, ..., cv_nfolds,
      ##  a) get xgb fit leaving out fold k
      ##  b) get log score for fold k (possibly for multiple values of nrounds)
      for(k in seq_len(cv_nfolds)) {
        ## step a) get xgb fit leaving out fold k
        dtrain_Rmatrix_k <- dtrain_Rmatrix[-cv_folds[[k]], , drop = FALSE]
        dtrain_k <- xgb.DMatrix(
          data = dtrain_Rmatrix_k
        )
        
        obj_deriv_fn_train_k <- get_obj_deriv_fn(
          component_model_log_scores = model_scores[-cv_folds[[k]], , drop = FALSE],
          dtrain_Rmatrix = dtrain_Rmatrix_k)
        
        fit_k <- xgb.train(
          params = params,
          data = dtrain_k,
          nrounds = params$nrounds,
          obj = obj_deriv_fn_train_k,
          verbose = 0
        )
        
        ## step b) get log score for fold k (val for validation)
        dval_Rmatrix_k <- dtrain_Rmatrix[cv_folds[[k]], , drop = FALSE]
        dval_k <- xgb.DMatrix(
          data = dval_Rmatrix_k
        )
        
        obj_fn_val_k <- get_obj_fn(
          component_model_log_scores = model_scores[cv_folds[[k]], , drop = FALSE])
        
        ## obj_fn returns -1* log score (thing to minimize)
        ## to avoid confusion (?) I'll return log score
        for(row_to_eval in similar_param_rows) {
          cv_results[row_to_eval, paste0("cv_log_score_fold_", k)] <-
            -1 * obj_fn_val_k(
              preds = predict(fit_k,
                newdata = dval_k,
                ntreelimit = cv_results[row_to_eval, "nrounds"])
            )
        }
      } # end code to get log score for each k-fold
    } # end code to get log score for each parameter combination
    
    cv_results$cv_log_score <- apply(cv_results[, paste0("cv_log_score_fold_", seq_len(cv_nfolds))], 1, sum)
    
    ## best ind has highest log score
    ## (see comment about multiplication by -1 above)
    best_params_ind <- which.max(cv_results$cv_log_score)
    
    params <- cv_results[best_params_ind, seq_len(ncol(cv_results) - cv_nfolds - 1)]
    attr(params, "out.attrs") <- NULL
    params <- as.list(params)
    params <- c(params,
      base_params[!(names(base_params) %in% names(params))])
  } # end code for cross-validation for parameter selection
  
  ## get fit with all training data based on selected parameters
  obj_deriv_fn <- get_obj_deriv_fn(
    component_model_log_scores = model_scores,
    dtrain_Rmatrix = dtrain_Rmatrix)
  
  fit <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = nrounds,
    obj = obj_deriv_fn,
    verbose = 0
  )
  
  ## return
  if(is.null(cv_params)) {
    return(structure(
      list(fit = xgb.save.raw(fit),
        formula = formula,
        params = params,
        num_models = ncol(model_scores)),
      class = "xgbstack"
    ))
  } else {
    return(structure(
      list(fit = xgb.save.raw(fit),
        formula = formula,
        params = params,
        num_models = ncol(model_scores),
        cv_results = cv_results),
      class = "xgbstack"
    ))
  }
}
