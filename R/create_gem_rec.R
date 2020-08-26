create_gem_rec <- function(split) {
    recipe(
        cut ~ ., 
        data = training(split)
    ) %>%
        step_normalize(all_numeric(), 
                       -all_outcomes()) %>% 
        step_dummy(
            all_nominal(), 
            -all_outcomes(), 
            one_hot = TRUE
        )
}

