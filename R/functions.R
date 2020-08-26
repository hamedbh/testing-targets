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

calibrate_prob_model <- function(pred_tbl, 
                                 pred_col, 
                                 actual_col, 
                                 method = c("platt", "isotonic")) {
    method <- match.arg(method)
    if (method == "platt") {
        d <- pred_tbl %>% 
            select({{ pred_col }}, 
                   {{ actual_col }})
        folds <- vfold_cv(d, v = 5, strata = {{ actual_col }})
        rec <- recipe(d) %>% 
            update_role({{ pred_col }}, 
                        new_role = "predictor") %>% 
            update_role({{ actual_col }}, 
                        new_role = "outcome")
        mod <- logistic_reg() %>% 
            set_engine("glm")
        fit_resamples(
            mod, 
            preprocessor = rec, 
            resamples = folds, 
            metrics = metric_set(mn_log_loss), 
            control = control_grid(verbose = TRUE, 
                                   save_pred = TRUE)
        ) %>% 
            collect_predictions(summarize = TRUE) %>% 
            select(
                .row, 
                {{ pred_col }}, 
                {{ actual_col }}
            )
    } else {
        levs <- pred_tbl %>% pull({{ actual_col }}) %>% levels()
        preds <- pred_tbl %>% pull({{ pred_col }})
        truth <- as.integer(pred_tbl %>% pull({{ actual_col }}) == levs[1])
        iso <- isoreg(preds, truth)
        tibble(
            .row = pred_tbl %>% 
                arrange({{ pred_col }}) %>% 
                pull(.row), 
            {{ pred_col }} := iso[["yf"]]
        ) %>% 
            inner_join(
                pred_tbl %>% 
                    select(.row, 
                           {{ actual_col }}), 
                by = ".row"
            )
    }
}
