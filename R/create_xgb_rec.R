create_xgb_rec <- function(split) {
    recipe(
        cut ~ ., 
        data = training(split)
    ) %>%
        step_dummy(
            all_nominal(), 
            -all_outcomes(), 
            one_hot = TRUE
        )
}
