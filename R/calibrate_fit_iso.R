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
