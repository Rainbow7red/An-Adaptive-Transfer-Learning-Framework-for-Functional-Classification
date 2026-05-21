# An Adaptive Transfer Learning Framework for Functional Classification

Welcome! This repository is the code companion for our paper **"An Adaptive
Transfer Learning Framework for Functional Classification"**, by Caihong Qin,
Jinhan Xie, Ting Li, and Yang Bai.

The paper was published in *Journal of the American Statistical Association*,
Volume 120, Issue 550, pages 1201--1213, with DOI:
https://doi.org/10.1080/01621459.2024.2403788.

The paper studies binary classification when the target functional dataset is
small and several source datasets may be available. The main idea is to build a
functional DWD classifier, evaluate how helpful each source dataset is for the
target task, and transfer source information adaptively to reduce negative
transfer.

## Example

The code below generates an example dataset and applies the transfer learning
functions.

Install `MASS` if it is not already available:

```r
install.packages("MASS")
```

```r
# Load the functions for data generation, functional DWD, and transfer learning.
source("functional_transfer_learning.R")

# Generate target training/test data and candidate sources.
# The example has two positive sources, two negative sources, and one random source.
sample_data <- generate_transfer_learning_sample_data(seed = 2036)

# Target data are lists with x and y.
# x is an n by grid_size matrix of curves; y is a vector coded as -1 or 1.
target_data <- sample_data$target_train
target_test <- sample_data$target_test

# Sources are stored as a list; each source has the same list(x, y) structure.
sources <- sample_data$sources

# Check the source types generated for this example.
sample_data$source_info

# Use the positive sources when the informative source set is known.
positive_sources <- sources[sample_data$positive_source_indices]

# Fit transfer learning when the positive source set is known.
known_fit <- fit_atl_known_sources(
  target_data = target_data,              # target training data
  source_data_list = positive_sources     # known positive sources
)

# Average transfer classifiers when the informative source set is unknown.
# The weights are estimated from internal cross-validation decision matrices.
averaged_fit <- average_transfer_fits(
  target_data = target_data,              # target training data
  source_data_list = sources,             # all candidate sources
  metric = "correlation",                 # rank sources by correlation
  mode = "debiased",                      # average debiased transfer fits
  adaptive = FALSE                        # fit the tsf1/tsf2 path used below
)

# Known positive-source case: target-only, transfer, debiased transfer, and ATL.
atl_choice <- known_fit$adaptive$choices[[1]]$correlation_choice
atl_fit <- if (atl_choice == "debiased") known_fit$tsf2 else known_fit$tsf1
known_errors <- c(
  target_only = misclassification_rate(known_fit$target_only, target_test),
  tsf1 = misclassification_rate(known_fit$tsf1, target_test),
  tsf2 = misclassification_rate(known_fit$tsf2, target_test),
  atl = misclassification_rate(atl_fit, target_test)
)

# Unknown source case: use ranked sources and DWD-loss model averaging.
unknown_final_fit <- averaged_fit$final_fit
unknown_errors <- c(
  target_only = misclassification_rate(unknown_final_fit$target_only, target_test),
  tsf1 = misclassification_rate(unknown_final_fit$tsf1, target_test),
  tsf2 = misclassification_rate(unknown_final_fit$tsf2, target_test),
  average = misclassification_rate(averaged_fit$fit, target_test)
)

known_errors
unknown_errors
```

To use your own data, prepare the target and each source dataset as
`list(x = x_matrix, y = y_vector)`. The matrix `x` should have one curve per
row and one grid point per column. The vector `y` should use labels `-1` and
`1`. All target and source matrices should be observed on the same grid. The
code assumes this structure directly and does not recode labels or reshape data.

## Details

The repository includes one script:

- `functional_transfer_learning.R`: synthetic data generation, functional DWD,
  known-source adaptive transfer learning, source ranking, and ranked-path model
  averaging.

Required R packages:

- `MASS` is used for `MASS::ginv()` in the model-averaging weight update.
- `stats` is used from standard R.

Main functions:

- `generate_synthetic_functional_data()`: generates one target, positive
  source, or random source dataset.
- `generate_transfer_learning_sample_data()`: generates target training,
  test, positive source, negative source, and random source datasets in one call
  using one seed.
- `fit_functional_dwd()`: fits the target-only functional DWD classifier.
- `fit_atl_known_sources()`: implements the known-informative-source transfer
  step (`tsf1`), debiased transfer step (`tsf2`), and adaptive choice between
  them.
- `average_transfer_fits()`: implements ranked-source transfer paths followed
  by the DWD-loss model-averaging weight update using internal cross-validation
  folds.
