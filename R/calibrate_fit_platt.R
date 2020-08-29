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
