#' Create model to calibrate probabilities with isotonic regression
#' 
#' The creates an object that can be used to calibrate probabilities. The most 
#' common use case will be to provide a set of probability predictions 
#' from model training with the correct labels to this function. Then that 
#' output can be used to calibrate the probabilities generated on test data. 
#'
#' @param pred_tbl A tibble of predictions, such as that created by 
#'     \code{tune::collect_predictions()}
#' @param pred_col Unquoted name of the column containing the probability 
#'     predictions. 
#' @param truth_col Unquoted name of the column containing the correct outcome 
#'     labels. Must be a factor. 
#' @param truth_val If left as \code{NULL}, the default, this will be the first 
#'     level of \code{truth_col}. If supplied it must be a string corresponding 
#'     to the positive level of \code{truth_col}. 
#'
#' @return An object of class \code{iso_fit}, which can then be used in 
#'     \code{predict()} to calibrate predictions.
#' @export

calibrate_fit_iso <- function(pred_tbl, 
                              pred_col, 
                              truth_col, 
                              truth_val = NULL) {
    if (!(is.factor(pred_tbl %>% pull({{ truth_col }})))) {
        stop(
            "truth_col must be a factor"
        )
    }
    levs <- pred_tbl %>% pull({{ truth_col }}) %>% levels()
    if (is.null(truth_val)) {
        truth_val <- levs[1]
    } else if (!(truth_val %in% levs)) {
        stop(
            "truth_val not in levels of truth_col"
        )
    }
    
    preds <- c(0, pred_tbl %>% pull({{ pred_col }}), 1)
    truth <- c(0L, 
               as.integer(pred_tbl %>% 
                              pull({{ truth_col }}) == levs[1]), 
               1L)
    
    iso <- isoreg(preds, truth)
    fit <- as.stepfun(iso)
    res <- list(
        fit = fit, 
        prob_col = rlang::as_name(enquo(pred_col)), 
        truth_val = truth_val, 
        false_val = levs[!(levs == truth_val)]
    )
    class(res) <- c(class(res), "iso_fit")
    res
}

#' Isotonic model predictions
#'
#' @param object An object of class \code{iso_fit}
#' @param new_data A tibble
#' @param ... Not used: given here to give users informative errors if they 
#'     pass extra parameters. 
#'
#' @return A tibble of calibrated probability predictions
#' @export

predict.iso_fit <- function(object,
                            new_data, 
                            ...) {
    the_dots <- enquos(...)
    if (any(names(the_dots) == "newdata"))
        rlang::abort("Did you mean to use `new_data` instead of `newdata`?")
    other_args <- c("level", "std_error", "quantile")
    is_pred_arg <- names(the_dots) %in% other_args
    if (any(!is_pred_arg)) {
        bad_args <- names(the_dots)[!is_pred_arg]
        bad_args <- paste0("`", bad_args, "`", collapse = ", ")
        rlang::abort(
            glue::glue(
                "The ellipses are not used to pass args to the model function's ",
                "predict function. These arguments cannot be used: {bad_args}",
            )
        )
    }
    prob_col <- sym(object$prob_col)
    vec <- new_data %>% pull({{ prob_col }})
    preds <- object$fit(vec)
    res <- tibble(
        !!str_c(".pred_", object$truth_val) := preds, 
        !!str_c(".pred_", object$false_val) := (1 - preds)
    )
    res
}
