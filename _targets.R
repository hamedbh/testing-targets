library(targets)
# Using a separate packages script seems to work fine, as it did under drake
source("packages.R")
source("R/functions.R")
options(tidymodels.dark = TRUE)
library(future)
plan(multisession)

tar_pipeline(
    # data wrangling steps
    tar_target(
        raw_data_file, 
        "data/diamonds.csv", 
        format = "file"
    ), 
    tar_target(
        raw_d,
        read_csv(raw_data_file),
        format = "fst_tbl"
    ),
    tar_target(
        clean_d,
        clean_diamonds(raw_d),
        format = "fst_tbl"
    ),
    # shared steps for modelling
    tar_target(
        gem_split,
        initial_split(clean_d, prop = 1/2, strata = cut)
    ),
    tar_target(
        gem_CV_folds,
        vfold_cv(training(gem_split), v = 5, strata = cut)
    ),
    tar_target(
        gem_rec,
        create_gem_rec(gem_split)
    ),
    # build an elastic net logistic regression model
    tar_target(
        gem_elnet_mod,
        logistic_reg(
            penalty = tune(),
            mixture = tune()
        ) %>%
            set_engine(
                "glmnet"
            )
    ),
    tar_target(
        gem_elnet_wfl,
        workflow() %>% add_recipe(gem_rec) %>% add_model(gem_elnet_mod)
    ),
    tar_target(
        gem_elnet_param,
        gem_elnet_wfl %>%
            parameters() %>% 
            update(mixture = mixture(range = c(0, 1)))
    ),
    tar_target(
        gem_elnet_grid,
        gem_elnet_param %>% grid_regular(levels = c(20, 11))
    ),
    tar_target(
        gem_elnet_tune,
        tune_grid(
            gem_elnet_wfl,
            resamples = gem_CV_folds,
            param_info = gem_elnet_param,
            grid = gem_elnet_grid,
            metrics = metric_set(roc_auc, pr_auc, gain_capture, mn_log_loss),
            control = control_grid(verbose = TRUE, save_pred = TRUE)
        )
    ),
    # build a MARS model
    tar_target(
        gem_mars_rec,
        create_mars_rec(gem_split)
    ),
    tar_target(
        gem_mars_mod,
        mars(
            mode = "classification",
            num_terms = tune(),
            prod_degree = 2
        ) %>%
            set_engine(
                "earth"
            )
    ),
    tar_target(
        gem_mars_wfl,
        workflow() %>% add_recipe(gem_mars_rec) %>% add_model(gem_mars_mod)
    ),
    tar_target(
        gem_mars_param,
        gem_mars_wfl %>%
            parameters() %>%
            update(
                num_terms = finalize(num_terms(),
                                     gem_mars_rec %>%
                                         prep() %>%
                                         juice())
            )
    ),
    tar_target(
        gem_mars_grid,
        gem_mars_param %>% grid_regular(levels = 4)
    ),
    tar_target(
        gem_mars_tune,
        tune_grid(
            gem_mars_wfl,
            resamples = gem_CV_folds,
            param_info = gem_mars_param,
            grid = gem_mars_grid,
            metrics = metric_set(roc_auc, pr_auc, gain_capture, mn_log_loss),
            control = control_grid(verbose = TRUE, save_pred = TRUE)
        )
    )
)
