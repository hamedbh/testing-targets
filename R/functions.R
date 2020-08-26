clean_diamonds <- function(raw_d) {
    raw_d %>%
        mutate(
            color = factor(
                color,
                levels = c("D", "E", "F", "G", "H", "I", "J"),
                ordered = TRUE
            ),
            clarity = factor(
                clarity,
                levels = c("I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"),
                ordered = TRUE
            ), 
            cut = (if_else(cut == "Ideal", "ideal", "other") %>% 
                       factor(levels = c("ideal", "other")))
        )
}

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

create_mars_rec <- function(split) {
    recipe(
        cut ~ ., 
        data = training(split)
    ) %>%
        step_normalize(all_numeric(), 
                       -all_outcomes()) %>% 
        step_dummy(
            all_nominal(), 
            -all_outcomes()
        )
}