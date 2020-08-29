library(targets)
library(tarchetypes)
# Using a separate packages script seems to work fine, as it did under drake
tar_option_set(packages = c("tidyverse", "tidymodels"))
purrr::walk(list.files("R", full.names = TRUE), 
            ~ source(.x))
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
        workflow() %>% add_recipe(gem_rec) %>% add_model(gem_mars_mod)
    ),
    tar_target(
        gem_mars_param,
        gem_mars_wfl %>%
            parameters() %>%
            update(
                num_terms = finalize(num_terms(),
                                     gem_rec %>%
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
    ), 
    # build an XGBoost model
    tar_target(
        gem_xgb_mod,
        boost_tree(
            mode = "classification",
            mtry = tune(), 
            trees = 100L, 
            tree_depth = tune(), 
            learn_rate = 0.1, 
            sample_size = tune()
        ) %>%
            set_engine(
                "xgboost"
            )
    ),
    tar_target(
        gem_xgb_wfl,
        workflow() %>% add_recipe(gem_rec) %>% add_model(gem_xgb_mod)
    ),
    tar_target(
        gem_xgb_param,
        gem_xgb_wfl %>%
            parameters() %>%
            update(
                tree_depth = tree_depth(range = c(4, 10)), 
                mtry = mtry(range = 
                                c(
                                    round((gem_rec %>% 
                                               prep() %>% 
                                               juice() %>% 
                                               ncol()) * 0.4), 
                                    (gem_rec %>% 
                                         prep() %>% 
                                         juice() %>% 
                                         ncol) - 1L
                                )), 
                sample_size = sample_prop(range = c(0.5, 1))
            )
    ),
    tar_target(
        gem_xgb_grid,
        gem_xgb_param %>% grid_regular(levels = c(4, 4, 3))
    ),
    tar_target(
        gem_xgb_tune,
        tune_grid(
            gem_xgb_wfl,
            resamples = gem_CV_folds,
            param_info = gem_xgb_param,
            grid = gem_xgb_grid,
            metrics = metric_set(roc_auc, pr_auc, gain_capture, mn_log_loss),
            control = control_grid(verbose = TRUE, save_pred = TRUE)
        )
    ),
    # pull out the best parameters for each model type, fit a model on the 
    # training data, and collect predictions made with those parameters
    # first elastic net
    tar_target(
        gem_elnet_best_params, 
        gem_elnet_tune %>%
            select_by_pct_loss(desc(penalty),
                               metric = "roc_auc",
                               limit = 1)
    ), 
    tar_target(
        gem_elnet_fit, 
        gem_elnet_wfl %>%
            finalize_workflow(gem_elnet_best_params) %>%
            fit(data = training(gem_split))
    ), 
    tar_target(
        gem_elnet_preds, 
        gem_elnet_tune %>% 
            collect_predictions() %>% 
            semi_join(gem_elnet_best_params, 
                      by = c("penalty", "mixture"))
    ), 
    # then MARS
    tar_target(
        gem_mars_best_params, 
        gem_mars_tune %>%
            select_by_pct_loss(num_terms,
                               metric = "roc_auc",
                               limit = 1)
    ), 
    tar_target(
        gem_mars_fit, 
        gem_mars_wfl %>%
            finalize_workflow(gem_mars_best_params) %>%
            fit(data = training(gem_split))
    ), 
    tar_target(
        gem_mars_preds, 
        gem_mars_tune %>% 
            collect_predictions() %>% 
            semi_join(gem_mars_best_params, 
                      by = c("num_terms"))
    ), 
    # finally XGBoost
    tar_target(
        gem_xgb_best_params, 
        gem_xgb_tune %>% 
            select_by_pct_loss(tree_depth, sample_size, mtry, 
                               metric = "roc_auc", 
                               limit = 1)
    ), 
    tar_target(
        gem_xgb_fit, 
        gem_xgb_wfl %>%
            finalize_workflow(gem_xgb_best_params) %>%
            fit(data = training(gem_split))
    ), 
    tar_target(
        gem_xgb_preds, 
        gem_xgb_tune %>% 
            collect_predictions() %>% 
            semi_join(gem_xgb_best_params, 
                      by = c("mtry", "sample_size", "tree_depth"))
    ), 
    # Now build calibrated versions of each of the models using their 
    # predictions
    # first put the models and predictions in lists for branching
    tar_target(
        fit_list, 
        list(
            elnet = gem_elnet_fit, 
            mars = gem_mars_fit, 
            xgb = gem_xgb_fit
        ), 
        iteration = "list"
    ), 
    tar_target(
        pred_list, 
        list(
            elnet = gem_elnet_preds, 
            mars = gem_mars_preds, 
            xgb = gem_xgb_preds
        ), 
        iteration = "list"
    ), 
    # then build the platt and isotonic versions
    tar_target(
        platt_mods,
        calibrate_fit_platt(
            pred_list, 
            .pred_ideal, 
            cut
        ),
        pattern = map(pred_list), 
        iteration = "list"
    ), 
    tar_target(
        iso_mods,
        calibrate_fit_iso(
            pred_list, 
            .pred_ideal, 
            cut
        ),
        pattern = map(pred_list), 
        iteration = "list"
    ), 
    # Now generate predictions on the test data
    tar_target(
        unscaled_preds, 
        predict(
            fit_list, 
            new_data = testing(gem_split), 
            type = "prob"
        ) %>% 
            bind_cols(
                testing(gem_split) %>% 
                    select(cut)
            ), 
        pattern = map(fit_list), 
        iteration = "list"
    ), 
    # Then scale with Platt and iso methods
    tar_target(
        platt_scaled_preds, 
        predict(
            platt_mods, 
            unscaled_preds, 
            type = "prob"
        ) %>% 
            bind_cols(
                testing(gem_split) %>% 
                    select(cut)
            ), 
        pattern = map(platt_mods, unscaled_preds), 
        iteration = "list"
    ), 
    tar_target(
        iso_scaled_preds, 
        predict(
            iso_mods, 
            unscaled_preds
        ) %>% 
            bind_cols(
                testing(gem_split) %>% 
                    select(cut)
            ), 
        pattern = map(iso_mods, unscaled_preds), 
        iteration = "list"
    )
)
