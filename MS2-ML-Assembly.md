## Required Packages

``` r
library(tidymodels)
library(forcats)
library(doFuture)
library(vip)
```

## Import data, clean, split into training and test set

``` r
ms2 <- readr::read_csv("./Data/wtMS2_final_ts.csv") #load in wtMS2 training set
ms2_holdouts <- readr::read_csv("./Data/wtMS2_final_hs.csv") #load in wtMS2 holdout set

ms2_ready <- ms2 %>% mutate_if(is.character, factor) %>%
  mutate(across(Original, as.factor))

set.seed(111)
ms2_split <- initial_split(ms2_ready, prop = 3/4, strata = Result)
ms2_train <- training(ms2_split)
ms2_test <- testing(ms2_split)

##Setup 10 fold cross validation
set.seed(111)
ms2_folds <- vfold_cv(ms2_train, strata = Result)

ms2_rec <- 
  recipe(Result ~ ., data = ms2_train) %>% 
  step_rm(AFS, Position) %>% 
  step_normalize(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_zv(all_predictors())
```

## Instantiate the models to be tested

``` r
boost_spec <- boost_tree(
  trees = 1000) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

glm_spec <- logistic_reg() %>%
  set_engine("glm")

rf_spec <- rand_forest(trees = 1000) %>%
  set_mode("classification") %>%
  set_engine("ranger")

knn_spec <- nearest_neighbor() %>%
  set_engine("kknn") %>%
  set_mode("classification")

ms2_wf <- workflow() %>%
  add_recipe(ms2_rec)
```

## Run default parameter models

``` r
class_metrics <- metric_set(pr_auc, roc_auc, accuracy, mcc)
control <- control_resamples(save_pred = TRUE)

ms2_xgb_res <- 
  ms2_wf %>%
  add_model(boost_spec) %>%
  fit_resamples(resamples = ms2_folds, control = control, metrics = class_metrics)

ms2_glm_res <-
  ms2_wf %>%
  add_model(glm_spec) %>%
  fit_resamples(resamples = ms2_folds, control = control, metrics = class_metrics)

ms2_rf_res <- 
  ms2_wf %>%
  add_model(rf_spec) %>%
  fit_resamples(resamples = ms2_folds, control = control, metrics = class_metrics)

ms2_knn_res <- 
  ms2_wf %>%
  add_model(knn_spec) %>%
  fit_resamples(resamples = ms2_folds, control = control, metrics = class_metrics)

ms2_wf <- workflow() %>%
  add_recipe(ms2_rec)
```

## Collect model metrics and plot fold results

``` r
xgb_mcc <- 
  collect_metrics(ms2_xgb_res, summarize = FALSE) %>% 
  filter(.metric == "mcc") %>% 
  select(id, `XGBoost` = .estimate) 

glm_mcc <- 
  collect_metrics(ms2_glm_res, summarize = FALSE) %>% 
  filter(.metric == "mcc") %>% 
  select(id, `Logistic_Regression` = .estimate)

rf_mcc <- 
  collect_metrics(ms2_rf_res, summarize = FALSE) %>% 
  filter(.metric == "mcc") %>% 
  select(id, `Random_Forest` = .estimate)

knn_mcc <- 
  collect_metrics(ms2_knn_res, summarize = FALSE) %>% 
  filter(.metric == "mcc") %>% 
  select(id, `K_Nearest_Neighbors` = .estimate)

mcc_estimates <- inner_join(xgb_mcc, glm_mcc, by = "id") %>%
  inner_join(rf_mcc, by = "id") %>%
  inner_join(knn_mcc, by = "id")

#Plot MCC comparison of tested algorithms
mcc_long <- pivot_longer(mcc_estimates, -id, names_to = "model", values_to = "mcc")

mcc_long %>%
    mutate(model = fct_reorder(model, mcc, .fun='mean')) %>%
ggplot(aes(x = model, y = mcc, fill=model)) + stat_boxplot(geom = "errorbar", width = 0.1) +
  geom_boxplot() + theme(legend.position="none")
```

## Set the specifications of XGBoost hyperparameters to be tuned

``` r
xgb_spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(), min_n = tune(), loss_reduction = tune(), mtry = tune(), sample_size = tune(), learn_rate = tune(), stop_iter = 10
) %>%
  set_engine("xgboost", eval_metric = "auc") %>%
  set_mode("classification")

#Create a workflow to test grid, finalize parameters
xgb_wf <- workflow() %>%
  add_recipe(ms2_rec) %>% 
  add_model(xgb_spec)
  
xgb_set <- parameters(xgb_wf)

ms2_train_preprocessed <- ms2_rec %>%
  prep(ms2_train) %>% 
  juice()

xgb_set <- xgb_set %>% 
  update(mtry = finalize(mtry(), ms2_train_preprocessed))
```

## Perform Bayesian Hyperparameter Tuning - XGBoost

``` r
#Build a parallel processing cluster
all_cores <- parallel::detectCores(logical = FALSE)

registerDoFuture()
cl <- parallel::makeCluster(all_cores)
plan(cluster, workers = cl)
set.seed(777)

#change verbose to TRUE below to see model performance over each iteration

search_res <-
  xgb_wf %>% 
  tune_bayes(
    resamples = ms2_folds,
    # To use non-default parameter ranges
    param_info = xgb_set,
    # Generate ten initial parameter sets
    initial = 10,
    iter = 100,
    # How to measure performance?
    metrics = metric_set(roc_auc, mcc),
    control = control_bayes(no_improve = 20, verbose = FALSE, event_level = "first") #sets which case is the 'positive' result
  )
```

## Finalize parameters and train final model using train set, evaluate on test split

``` r
xgb_wf_final <- finalize_workflow(
  xgb_wf,
  select_best(search_res, "mcc")
)

#fit final model
final_res <- last_fit(xgb_wf_final, ms2_split)

#metrics of final model
final_performance <- final_res %>% collect_metrics()

#Collect the predictions from final model
preds <- final_res %>% 
  collect_predictions()

#confusion matrix of model performance
cmat <- conf_mat(preds, truth = Result, estimate = .pred_class)
xgb_metrics <- summary(cmat, event_level = "first")
xgb_metrics
```

## Model training on full dataset to make predictions about the unknown holdout set of mutations

``` r
xgb_model <- xgb_wf_final %>%
  fit(data = ms2_ready)

ms2_holdouts <- ms2_holdouts %>% mutate_if(is.character, factor) %>%
  mutate(across(Original, as.factor))

xgb_holdout_preds <- predict(xgb_model, new_data = ms2_holdouts)

xgb_predicted_dataset <- ms2_holdouts %>% 
  mutate(Result = xgb_holdout_preds$.pred_class)

xgb_predicted_dataset
```

## Feature importance of the final model

``` r
xgb_obj <- extract_fit_parsnip(xgb_model)$fit

xgb_obj %>% 
  vip(geom = "col", num_features = 10)
```

## Set the specifications of Random Forest hyperparameters to be tuned

``` r
rf_spec <- rand_forest(
  trees = tune(),
  min_n = tune(), mtry = tune()) %>%
  set_engine("ranger", importance = "permutation") %>%
  set_mode("classification")

##Create a workflow to test grid, finalize parameters
rf_wf <- workflow() %>%
  add_recipe(ms2_rec) %>% 
  add_model(rf_spec)
  
rf_set <- parameters(rf_wf)

ms2_train_preprocessed <- ms2_rec %>%
  prep(ms2_train) %>% 
  juice()

rf_set <- rf_set %>% 
  update(mtry = finalize(mtry(), ms2_train_preprocessed))
```

## Perform Bayesian Hyperparameter Tuning - Random Forest

``` r
set.seed(777)

#change verbose to TRUE below to see model performance over each iteration

search_res <-
  rf_wf %>% 
  tune_bayes(
    resamples = ms2_folds,
    # To use non-default parameter ranges
    param_info = rf_set,
    # Generate ten initial parameter sets
    initial = 10,
    iter = 100,
    # How to measure performance?
    metrics = metric_set(roc_auc, mcc),
    control = control_bayes(no_improve = 20, verbose = FALSE, event_level = "first") #sets which case is the 'positive' result
  )
```

## Finalize parameters and train final model using train set, evaluate on test split

``` r
rf_wf_final <- finalize_workflow(
  rf_wf,
  select_best(search_res, "mcc")
)

#fit final model
rffinal_res <- last_fit(rf_wf_final, ms2_split)

#metrics of final model
rffinal_performance <- rffinal_res %>% collect_metrics()

#Collect the predictions from final model
rfpreds <- rffinal_res %>% 
  collect_predictions()

#confusion matrix of model performance
rfcmat <- conf_mat(preds, truth = Result, estimate = .pred_class)
rf_metrics <- summary(rfcmat, event_level = "first")
rf_metrics
```

## Random Forest model training on full dataset to make predictions about the unknown holdout set of mutations

``` r
rf_model <- rf_wf_final %>%
  fit(data = ms2_ready)

rf_holdout_preds <- predict(rf_model, new_data = ms2_holdouts)

rf_predicted_dataset <- ms2_holdouts %>% 
  mutate(Result = rf_holdout_preds$.pred_class)

rf_predicted_dataset
```

## Feature importance of the final model

``` r
rf_obj <- extract_fit_parsnip(rf_model)$fit

rf_obj %>% 
  vip(geom = "col", num_features = 10)
```
