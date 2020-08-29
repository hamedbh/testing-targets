#' Create model to calibrate probabilities with Platt scaling
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
#'
#' @return A parsnip \code{model_fit} object. This has an additional class, 
#'     \code{platt_fit}, to enable future methods.  
#' @export

calibrate_fit_platt <- function(pred_tbl, 
                                pred_col, 
                                truth_col) {
    fit <- logistic_reg() %>% 
        set_engine("glm") %>% 
        fit_xy(
            x = pred_tbl %>% select({{ pred_col }}), 
            y = pred_tbl %>% select({{ truth_col }})
        )
    class(fit) <- c(class(fit), "platt_fit")
    fit
}
