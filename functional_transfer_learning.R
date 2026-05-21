# Build the kernel and low-order terms used by the functional DWD solver.
build_dwd_operators <- function(x) {
  grid_size <- ncol(x)
  psi1 <- rep(1, grid_size)
  psi2 <- (0:grid_size) / grid_size - 0.5
  k2 <- ((psi2^2 - 1 / 12) / 2)[-1]
  k4 <- ((psi2^4 - psi2^2 / 2 + 7 / 240) / 24)[-(grid_size + 1)]
  k4st <- matrix(NA_real_, nrow = grid_size, ncol = grid_size)
  for (i in seq_len(grid_size)) {
    k4st[i, i:grid_size] <- k4[seq_len(grid_size - i + 1)]
    k4st[i:grid_size, i] <- k4[seq_len(grid_size - i + 1)]
  }
  kst <- k2 %*% t(k2) - k4st
  s0 <- x %*% cbind(psi1, psi2[-1]) / grid_size
  r0 <- x %*% kst %*% t(x) / grid_size^2
  list(grid_size = grid_size, psi1 = psi1, psi2 = psi2, kst = kst, s0 = s0, r0 = r0)
}

# Solve the functional DWD estimating equation for fixed tuning parameters.
solve_dwd_theta <- function(r_matrix, s_matrix, y, q_value, lambda_value, addi, max_iter, tol, lambda_min) {
  n <- nrow(r_matrix)
  n_basis <- ncol(s_matrix)
  theta <- rep(0, 1 + n_basis + n)

  r_svd <- svd(r_matrix)
  keep <- seq_len(sum(r_svd$d > lambda_min))

  b_matrix <- cbind(c(n, colSums(s_matrix)), rbind(colSums(s_matrix), t(s_matrix) %*% s_matrix))
  c_matrix <- cbind(colSums(r_matrix), r_matrix %*% s_matrix)
  b_inv <- solve(b_matrix)
  c_b_inv <- c_matrix %*% b_inv

  pi_values <- (r_svd$d[keep])^2 + (r_svd$d[keep]) * 2 * n * q_value * lambda_value / (q_value + 1)^2
  d_inv <- r_svd$u[, keep, drop = FALSE] %*%
    diag(1 / pi_values, nrow = length(pi_values)) %*%
    t(r_svd$v[, keep, drop = FALSE])
  d_inv_c <- d_inv %*% c_matrix
  middle <- solve(b_matrix - t(c_matrix) %*% d_inv_c)
  p_matrix <- d_inv + d_inv_c %*% middle %*% t(d_inv_c)
  p_c_b_inv <- p_matrix %*% c_b_inv
  a_inv <- cbind(
    rbind(b_inv + t(c_b_inv) %*% p_c_b_inv, -p_c_b_inv),
    rbind(-t(p_c_b_inv), p_matrix)
  )

  for (iter in seq_len(max_iter)) {
    u <- as.vector(y * (cbind(rep(1, n), s_matrix, r_matrix) %*% theta + addi))
    active <- which(u > q_value / (1 + q_value))
    d1loss <- rep(-1, length(u))
    if (length(active) > 0) {
      d1loss[active] <- -(q_value / u[active] / (q_value + 1))^(q_value + 1)
    }
    r_vec <- y * d1loss / n
    d1theta <- c(
      sum(r_vec),
      t(s_matrix) %*% r_vec,
      r_matrix %*% r_vec + 2 * lambda_value * r_matrix %*% theta[-seq_len(1 + n_basis)]
    )
    theta_new <- as.vector(theta - n * q_value / (q_value + 1)^2 * a_inv %*% d1theta)
    if (mean((theta_new - theta)^2) <= tol) {
      theta <- theta_new
      break
    }
    theta <- theta_new
  }

  theta
}

# Convert a solved DWD theta vector into alpha and beta.
build_dwd_fit <- function(theta, x_train, ops, q_value, lambda_value, cv_score = NA_real_) {
  hat_alpha <- theta[1]
  hat_beta <- as.vector(
    cbind(ops$psi1, ops$psi2[-1]) %*% theta[2:3] +
      ops$kst %*% t(x_train) %*% theta[-seq_len(3)]
  )
  list(
    alpha = hat_alpha,
    beta = hat_beta,
    q = q_value,
    lambda = lambda_value,
    cv_score = cv_score,
    grid_size = ops$grid_size
  )
}

# Fit functional DWD once for fixed tuning parameters.
fit_dwd_fixed_tuning <- function(x, y, ops, add_offset, q_value, lambda_value,
                                 max_iter, tol, lambda_min, cv_score = NA_real_) {
  theta <- solve_dwd_theta(
    r_matrix = ops$r0,
    s_matrix = ops$s0,
    y = y,
    q_value = q_value,
    lambda_value = lambda_value,
    addi = add_offset,
    max_iter = max_iter,
    tol = tol,
    lambda_min = lambda_min
  )
  build_dwd_fit(theta, x, ops, q_value, lambda_value, cv_score = cv_score)
}

# Fit a functional DWD classifier with cross-validated tuning parameters.
fit_functional_dwd <- function(
  x,
  y = NULL,
  q_grid = c(seq(0.01, 0.21, 0.05), 0.5, 1, 5),
  lambda_grid = seq(0.01, 0.22, 0.03),
  add_fit = NULL,
  folds = 5,
  max_iter = 100,
  tol = 1e-5,
  lambda_min = 1e-5
) {
  if (is.null(y) && is.list(x)) {
    y <- x$y
    x <- x$x
  }

  ops <- build_dwd_operators(x)
  add_offset <- rep(0, nrow(x))
  if (!is.null(add_fit)) {
    add_offset <- as.vector(add_fit$alpha + x %*% add_fit$beta / ops$grid_size)
  }

  n_folds <- min(folds, nrow(x))
  fold_ids <- lapply(seq_len(n_folds), function(fold_index) {
    start <- floor((fold_index - 1) * nrow(x) / n_folds) + 1
    end <- floor(fold_index * nrow(x) / n_folds)
    start:end
  })
  scores <- matrix(NA_real_, nrow = length(q_grid), ncol = length(lambda_grid))
  for (q_index in seq_along(q_grid)) {
    for (lambda_index in seq_along(lambda_grid)) {
      scores[q_index, lambda_index] <- mean(vapply(fold_ids, function(valid_idx) {
        train_idx <- setdiff(seq_len(nrow(x)), valid_idx)
        theta <- solve_dwd_theta(
          r_matrix = ops$r0[train_idx, train_idx, drop = FALSE],
          s_matrix = ops$s0[train_idx, , drop = FALSE],
          y = y[train_idx],
          q_value = q_grid[q_index],
          lambda_value = lambda_grid[lambda_index],
          addi = add_offset[train_idx],
          max_iter = max_iter,
          tol = tol,
          lambda_min = lambda_min
        )
        fold_fit <- build_dwd_fit(
          theta,
          x[train_idx, , drop = FALSE],
          ops,
          q_grid[q_index],
          lambda_grid[lambda_index]
        )
        mean(decision_values(fold_fit, x[valid_idx, , drop = FALSE]) * y[valid_idx] > 0)
      }, numeric(1)))
    }
  }

  flat_scores <- as.vector(t(scores))
  best_index <- order(flat_scores, decreasing = TRUE)[1]
  best <- c(
    ceiling(best_index / length(lambda_grid)),
    best_index - length(lambda_grid) * (ceiling(best_index / length(lambda_grid)) - 1)
  )
  fit <- fit_dwd_fixed_tuning(
    x = x,
    y = y,
    ops = ops,
    add_offset = add_offset,
    q_value = q_grid[best[1]],
    lambda_value = lambda_grid[best[2]],
    max_iter = max_iter,
    tol = tol,
    lambda_min = lambda_min,
    cv_score = scores[best[1], best[2]]
  )
  class(fit) <- "functional_dwd"
  fit
}

# Combine a pooled classifier with a target-data debiasing update.
combine_fits <- function(base_fit, update_fit) {
  structure(
    list(
      alpha = base_fit$alpha + update_fit$alpha,
      beta = base_fit$beta + update_fit$beta,
      q = update_fit$q,
      lambda = update_fit$lambda,
      cv_score = update_fit$cv_score,
      grid_size = base_fit$grid_size
    ),
    class = "functional_dwd"
  )
}

# Compute fitted decision scores for new functional observations.
decision_values <- function(fit, x) {
  as.vector(fit$alpha + x %*% fit$beta / fit$grid_size)
}

# Compute the misclassification rate on a dataset.
misclassification_rate <- function(fit, data) {
  mean(ifelse(decision_values(fit, data$x) > 0, 1, -1) != data$y)
}

# Compute the correlation score used for source transferability ranking.
transfer_correlation <- function(x1, x2) {
  x1 <- as.numeric(x1)
  x2 <- as.numeric(x2)
  if (stats::sd(x2) == 0) {
    return(sum(x1 * x2) / sum(abs(x1)))
  }
  stats::cor(x1, x2)
}

# Use bootstrap resampling to compare pooled and debiased transfer fits.
bootstrap_adaptive_transfer <- function(
  target_data,
  pooled_fit,
  alpha_grid = seq(0.05, 0.5, 0.05),
  n_boot = 50,
  q_grid = c(seq(0.01, 0.21, 0.05), 0.5, 1, 5),
  lambda_grid = seq(0.01, 0.22, 0.03),
  folds = 5
) {
  pooled_scores <- decision_values(pooled_fit, target_data$x)
  pooled_acc <- mean(ifelse(pooled_scores > 0, 1, -1) == target_data$y)
  pooled_cor <- transfer_correlation(pooled_scores, target_data$y)

  boot_metrics <- replicate(n_boot, {
    idx <- sample(seq_len(length(target_data$y)), replace = TRUE)
    boot_data <- list(x = target_data$x[idx, , drop = FALSE], y = target_data$y[idx])
    boot_update <- fit_functional_dwd(
      boot_data$x,
      boot_data$y,
      q_grid = q_grid,
      lambda_grid = lambda_grid,
      add_fit = pooled_fit,
      folds = min(folds, nrow(boot_data$x))
    )
    boot_fit <- combine_fits(pooled_fit, boot_update)
    scores <- decision_values(boot_fit, boot_data$x)
    c(
      acc = mean(ifelse(scores > 0, 1, -1) == boot_data$y),
      cor = transfer_correlation(scores, boot_data$y)
    )
  })

  boot_metrics <- t(boot_metrics)
  adaptive_labels <- lapply(alpha_grid, function(alpha_level) {
    acc_choice <- if (mean(boot_metrics[, "acc"] > pooled_acc) >= 1 - alpha_level) "debiased" else "pooled"
    cor_choice <- if (mean(boot_metrics[, "cor"] > pooled_cor) >= 1 - alpha_level) "debiased" else "pooled"
    list(alpha = alpha_level, accuracy_choice = acc_choice, correlation_choice = cor_choice)
  })

  list(
    pooled_accuracy = pooled_acc,
    pooled_correlation = pooled_cor,
    bootstrap_metrics = boot_metrics,
    choices = adaptive_labels
  )
}

# Fit adaptive transfer learning when informative sources are supplied.
fit_atl_known_sources <- function(
  target_data,
  source_data_list,
  adaptive = TRUE,
  alpha_grid = seq(0.05, 0.5, 0.05),
  n_boot = 50,
  q_grid = c(seq(0.01, 0.21, 0.05), 0.5, 1, 5),
  lambda_grid = seq(0.01, 0.22, 0.03),
  folds = 5
) {
  target_only <- fit_functional_dwd(
    target_data$x,
    target_data$y,
    q_grid = q_grid,
    lambda_grid = lambda_grid,
    folds = min(folds, nrow(target_data$x))
  )

  pooled_data <- list(
    x = do.call(rbind, c(list(target_data$x), lapply(source_data_list, function(data) data$x))),
    y = unlist(c(list(target_data$y), lapply(source_data_list, function(data) data$y)), use.names = FALSE)
  )
  pooled_fit <- fit_functional_dwd(
    pooled_data,
    q_grid = q_grid,
    lambda_grid = lambda_grid,
    folds = min(folds, nrow(pooled_data$x))
  )

  debias_update <- fit_functional_dwd(
    target_data$x,
    target_data$y,
    q_grid = q_grid,
    lambda_grid = lambda_grid,
    add_fit = pooled_fit,
    folds = min(folds, nrow(target_data$x))
  )
  debiased_fit <- combine_fits(pooled_fit, debias_update)

  adaptive_summary <- NULL
  if (adaptive) {
    adaptive_summary <- bootstrap_adaptive_transfer(
      target_data = target_data,
      pooled_fit = pooled_fit,
      alpha_grid = alpha_grid,
      n_boot = n_boot,
      q_grid = q_grid,
      lambda_grid = lambda_grid,
      folds = min(folds, nrow(target_data$x))
    )
  }

  list(
    target_only = target_only,
    pooled = pooled_fit,
    debiased = debiased_fit,
    tsf1 = pooled_fit,
    tsf2 = debiased_fit,
    adaptive = adaptive_summary
  )
}

# Rank candidate source datasets by their transferability to the target task.
rank_sources_by_transferability <- function(
  target_data,
  source_data_list,
  metric = c("correlation", "accuracy"),
  q_grid = c(seq(0.01, 0.21, 0.05), 0.5, 1, 5),
  lambda_grid = seq(0.01, 0.22, 0.03),
  folds = 5
) {
  metric <- match.arg(metric)
  scores <- numeric(length(source_data_list))
  fits <- vector("list", length(source_data_list))
  for (i in seq_along(source_data_list)) {
    fits[[i]] <- fit_functional_dwd(
      source_data_list[[i]]$x,
      source_data_list[[i]]$y,
      q_grid = q_grid,
      lambda_grid = lambda_grid,
      folds = min(folds, nrow(source_data_list[[i]]$x))
    )
    values <- decision_values(fits[[i]], target_data$x)
    if (metric == "accuracy") {
      scores[i] <- 2 * mean(ifelse(values > 0, 1, -1) == target_data$y) - 1
    } else {
      scores[i] <- transfer_correlation(values, target_data$y)
    }
  }

  order_index <- order(scores, decreasing = TRUE)
  list(
    order = order_index,
    scores = scores,
    ordered_scores = scores[order_index],
    ordered_sources = source_data_list[order_index],
    source_fits = fits
  )
}

# Fit a ranked transfer path when the informative source set is unknown.
fit_ranked_transfer_path <- function(
  target_data,
  source_data_list,
  metric = c("correlation", "accuracy"),
  ranking_target_data = NULL,
  adaptive = TRUE,
  alpha_grid = seq(0.05, 0.5, 0.05),
  n_boot = 50,
  q_grid = c(seq(0.01, 0.21, 0.05), 0.5, 1, 5),
  lambda_grid = seq(0.01, 0.22, 0.03),
  folds = 5
) {
  metric <- match.arg(metric)
  if (is.null(ranking_target_data)) {
    ranking_target_data <- target_data
  }
  ranking <- rank_sources_by_transferability(
    target_data = ranking_target_data,
    source_data_list = source_data_list,
    metric = metric,
    q_grid = q_grid,
    lambda_grid = lambda_grid,
    folds = folds
  )

  path_fits <- lapply(seq_along(ranking$ordered_sources), function(prefix_size) {
    fit_atl_known_sources(
      target_data = target_data,
      source_data_list = ranking$ordered_sources[seq_len(prefix_size)],
      adaptive = adaptive,
      alpha_grid = alpha_grid,
      n_boot = n_boot,
      q_grid = q_grid,
      lambda_grid = lambda_grid,
      folds = folds
    )
  })

  list(ranking = ranking, path_fits = path_fits, metric = metric)
}

# Keep sources with non-negative transferability and fit Algorithm 1 on them.
fit_truncated_transfer <- function(
  target_data,
  source_data_list,
  metric = c("correlation", "accuracy"),
  adaptive = TRUE,
  alpha_grid = seq(0.05, 0.5, 0.05),
  n_boot = 50,
  q_grid = c(seq(0.01, 0.21, 0.05), 0.5, 1, 5),
  lambda_grid = seq(0.01, 0.22, 0.03),
  folds = 5
) {
  metric <- match.arg(metric)
  ranking <- rank_sources_by_transferability(
    target_data = target_data,
    source_data_list = source_data_list,
    metric = metric,
    q_grid = q_grid,
    lambda_grid = lambda_grid,
    folds = folds
  )

  keep <- which(ranking$ordered_scores >= 0)
  selected_sources <- ranking$ordered_sources[keep]
  fit <- fit_atl_known_sources(
    target_data = target_data,
    source_data_list = selected_sources,
    adaptive = adaptive,
    alpha_grid = alpha_grid,
    n_boot = n_boot,
    q_grid = q_grid,
    lambda_grid = lambda_grid,
    folds = folds
  )

  list(ranking = ranking, selected_sources = selected_sources, fit = fit)
}

# Average ranked-path classifiers with the DWD-loss weight update.
average_ranked_transfer_fits <- function(
  ranked_path,
  cv_decision_matrices,
  cv_labels,
  mode = c("debiased", "pooled", "adaptive"),
  include_target_only = TRUE,
  q_value = 1,
  max_iter = 100,
  tol = 1e-5
) {
  mode <- match.arg(mode)
  candidate_fits <- lapply(ranked_path$path_fits, function(fit_object) {
    if (mode == "adaptive") {
      if (is.null(fit_object$adaptive)) {
        fit_object$tsf2
      } else {
        choice <- fit_object$adaptive$choices[[1]]$correlation_choice
        if (choice == "debiased") fit_object$tsf2 else fit_object$tsf1
      }
    } else {
      fit_object[[mode]]
    }
  })
  if (include_target_only && length(ranked_path$path_fits) > 0) {
    candidate_fits <- c(list(target_only = ranked_path$path_fits[[1]]$target_only), candidate_fits)
  }
  candidate_names <- names(candidate_fits)
  if (is.null(candidate_names)) {
    candidate_names <- rep("", length(candidate_fits))
  }
  unnamed <- candidate_names == ""
  candidate_names[unnamed] <- paste0("ranked_", seq_len(sum(unnamed)))
  names(candidate_fits) <- candidate_names

  cv_errors <- sapply(seq_along(candidate_fits), function(candidate_index) {
    mean(unlist(Map(function(decision_matrix, y) {
      ifelse(decision_matrix[, candidate_index] > 0, 1, -1) != y
    }, cv_decision_matrices, cv_labels)))
  })
  cross_product <- Reduce(`+`, lapply(cv_decision_matrices, function(decision_matrix) {
    t(decision_matrix) %*% decision_matrix
  }))
  weights <- rep(1 / length(candidate_fits), length(candidate_fits))
  for (iter in seq_len(max_iter)) {
    gradient <- Reduce(`+`, Map(function(decision_matrix, y) {
      margin <- as.vector(y * (decision_matrix %*% weights))
      loss_gradient <- rep(-1, length(margin))
      active <- which(margin > q_value / (q_value + 1))
      if (length(active) > 0) {
        loss_gradient[active] <- -(q_value / margin[active] / (q_value + 1))^(q_value + 1)
      }
      as.vector(t(decision_matrix) %*% (y * loss_gradient))
    }, cv_decision_matrices, cv_labels))
    system_matrix <- rbind(
      cbind(cross_product * (q_value + 1)^2 / q_value, rep(-1, length(candidate_fits))),
      c(rep(1, length(candidate_fits)), 0)
    )
    update <- as.vector(MASS::ginv(system_matrix) %*% c(-gradient, 0))
    weights_new <- weights + update[-length(update)]
    if (length(which(weights_new <= 0)) > 0) {
      weights_new[which(weights_new <= 0)] <- 0
    }
    weights_new <- weights_new / sum(weights_new)
    if (mean((weights_new - weights)^2) <= tol) {
      weights <- weights_new
      break
    }
    weights <- weights_new
  }

  alpha <- sum(weights * sapply(candidate_fits, function(fit) fit$alpha))
  beta <- Reduce(`+`, Map(function(fit, weight) fit$beta * weight, candidate_fits, weights))

  averaged_fit <- structure(
    list(
      alpha = alpha,
      beta = as.numeric(beta),
      q = NA_real_,
      lambda = NA_real_,
      cv_score = NA_real_,
      grid_size = candidate_fits[[1]]$grid_size
    ),
    class = "functional_dwd"
  )

  list(
    fit = averaged_fit,
    weights = stats::setNames(weights, candidate_names),
    cv_errors = stats::setNames(cv_errors, candidate_names),
    candidates = candidate_names
  )
}

# Fit ranked-path model averaging using internal cross-validation folds.
average_transfer_fits <- function(
  target_data,
  source_data_list,
  metric = c("correlation", "accuracy"),
  mode = c("debiased", "pooled", "adaptive"),
  adaptive = FALSE,
  alpha_grid = seq(0.05, 0.5, 0.05),
  n_boot = 50,
  q_grid = c(seq(0.01, 0.21, 0.05), 0.5, 1, 5),
  lambda_grid = seq(0.01, 0.22, 0.03),
  folds = 5,
  include_target_only = TRUE,
  q_value = 1,
  max_iter = 100,
  tol = 1e-5
) {
  metric <- match.arg(metric)
  mode <- match.arg(mode)

  # Following the simulation code, rank sources on one target half and estimate
  # model-averaging weights with cross-validation on the other half.
  split_point <- floor(nrow(target_data$x) / 2)
  averaging_data <- list(
    x = target_data$x[seq_len(split_point), , drop = FALSE],
    y = target_data$y[seq_len(split_point)]
  )
  ranking_data <- list(
    x = target_data$x[-seq_len(split_point), , drop = FALSE],
    y = target_data$y[-seq_len(split_point)]
  )

  transfer_path <- fit_ranked_transfer_path(
    target_data = target_data,
    source_data_list = source_data_list,
    metric = metric,
    ranking_target_data = ranking_data,
    adaptive = adaptive,
    alpha_grid = alpha_grid,
    n_boot = n_boot,
    q_grid = q_grid,
    lambda_grid = lambda_grid,
    folds = folds
  )
  ordered_sources <- transfer_path$ranking$ordered_sources
  n_folds <- min(folds, nrow(averaging_data$x))
  target_folds <- lapply(seq_len(n_folds), function(fold_index) {
    start <- floor((fold_index - 1) * nrow(averaging_data$x) / n_folds) + 1
    end <- floor(fold_index * nrow(averaging_data$x) / n_folds)
    start:end
  })
  source_folds <- lapply(ordered_sources, function(source_data) {
    lapply(seq_len(n_folds), function(fold_index) {
      start <- floor((fold_index - 1) * nrow(source_data$x) / n_folds) + 1
      end <- floor(fold_index * nrow(source_data$x) / n_folds)
      start:end
    })
  })

  cv_results <- lapply(seq_len(n_folds), function(fold_index) {
    target_valid_index <- target_folds[[fold_index]]
    target_train_index <- setdiff(seq_len(nrow(averaging_data$x)), target_valid_index)
    target_train <- list(
      x = averaging_data$x[target_train_index, , drop = FALSE],
      y = averaging_data$y[target_train_index]
    )
    target_valid <- list(
      x = averaging_data$x[target_valid_index, , drop = FALSE],
      y = averaging_data$y[target_valid_index]
    )
    source_train <- lapply(seq_along(ordered_sources), function(source_index) {
      source_data <- ordered_sources[[source_index]]
      source_valid_index <- source_folds[[source_index]][[fold_index]]
      source_train_index <- setdiff(seq_len(nrow(source_data$x)), source_valid_index)
      list(
        x = source_data$x[source_train_index, , drop = FALSE],
        y = source_data$y[source_train_index]
      )
    })

    fold_path <- lapply(seq_along(source_train), function(prefix_size) {
      fit_atl_known_sources(
        target_data = target_train,
        source_data_list = source_train[seq_len(prefix_size)],
        adaptive = adaptive,
        alpha_grid = alpha_grid,
        n_boot = n_boot,
        q_grid = q_grid,
        lambda_grid = lambda_grid,
        folds = folds
      )
    })
    fold_fits <- lapply(fold_path, function(fit_object) {
      if (mode == "adaptive") {
        if (is.null(fit_object$adaptive)) {
          fit_object$tsf2
        } else {
          choice <- fit_object$adaptive$choices[[1]]$correlation_choice
          if (choice == "debiased") fit_object$tsf2 else fit_object$tsf1
        }
      } else {
        fit_object[[mode]]
      }
    })
    if (include_target_only && length(fold_path) > 0) {
      fold_fits <- c(list(target_only = fold_path[[1]]$target_only), fold_fits)
    }
    list(
      decisions = do.call(cbind, lapply(fold_fits, decision_values, x = target_valid$x)),
      y = target_valid$y
    )
  })

  averaged <- average_ranked_transfer_fits(
    ranked_path = transfer_path,
    cv_decision_matrices = lapply(cv_results, function(result) result$decisions),
    cv_labels = lapply(cv_results, function(result) result$y),
    mode = mode,
    include_target_only = include_target_only,
    q_value = q_value,
    max_iter = max_iter,
    tol = tol
  )
  averaged$transfer_path <- transfer_path
  averaged$final_fit <- transfer_path$path_fits[[length(transfer_path$path_fits)]]
  averaged
}

# Generate synthetic functional classification data for examples.
generate_synthetic_functional_data <- function(
  n,
  sclass = 0,
  h = 0,
  beta = NULL,
  signal_strength = 1,
  source_shift = 0,
  alpha = 0.1,
  grid_size = 50,
  basis_size = 50,
  seed = NULL,
  return_beta = FALSE
) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  zeta_phi <- matrix(unlist(lapply(seq_len(grid_size), function(i) {
    c(1, sqrt(2) * cos((1:(basis_size - 1)) * pi * i / grid_size))
  })), ncol = grid_size)
  x <- matrix(runif(n * basis_size, -sqrt(3), sqrt(3)), ncol = basis_size) %*%
    (zeta_phi * ((-1)^((1:basis_size) + 1) * (1:basis_size)^(-1)))

  if (!is.null(beta)) {
    beta_used <- as.vector(beta)
  } else if (sclass == 0) {
    beta_used <- as.vector((4 * (-1)^((1:basis_size) + 1) * (1:basis_size)^(-2)) %*% zeta_phi)
  } else if (sclass == 1) {
    beta_used <- as.vector(
      ((4 * (-1)^((1:basis_size) + 1) + runif(basis_size, -1, 1) * h / pi) *
        (1:basis_size)^(-2)) %*% zeta_phi
    )
  } else {
    rho <- exp(-15 / grid_size)
    rho1 <- sqrt(1 - rho^2)
    z <- rnorm(1 + grid_size)
    beta_used <- c(z[1], rep(0, grid_size))
    for (i in seq_len(grid_size)) {
      beta_used[i + 1] <- rho * beta_used[i] + rho1 * z[i + 1]
    }
    beta_used <- beta_used[-1]
  }

  fx <- as.vector(alpha + source_shift + x %*% beta_used * signal_strength / grid_size)
  y <- stats::rbinom(n, 1, exp(fx) / (1 + exp(fx)))
  data <- list(x = x, y = ifelse(y == 1, 1, -1))
  if (return_beta) {
    data$beta <- beta_used
  }
  data
}

# Generate all target and source datasets for the README example with one seed.
generate_transfer_learning_sample_data <- function(
  n_target_train = 30,
  n_target_test = 500,
  n_source = 20,
  positive_h = c(7, 7),
  include_negative_sources = TRUE,
  n_random_sources = 1,
  grid_size = 50,
  basis_size = 50,
  signal_strength = 1,
  alpha = 0.1,
  seed = 2036
) {
  if (!is.null(seed)) {
    set.seed(seed)
  }

  target_train <- generate_synthetic_functional_data(
    n = n_target_train,
    sclass = 0,
    signal_strength = signal_strength,
    alpha = alpha,
    grid_size = grid_size,
    basis_size = basis_size
  )
  target_test <- generate_synthetic_functional_data(
    n = n_target_test,
    sclass = 0,
    signal_strength = signal_strength,
    alpha = alpha,
    grid_size = grid_size,
    basis_size = basis_size
  )

  positive_sources_with_beta <- lapply(positive_h, function(h_value) {
    generate_synthetic_functional_data(
      n = n_source,
      sclass = 1,
      h = h_value,
      signal_strength = signal_strength,
      alpha = alpha,
      grid_size = grid_size,
      basis_size = basis_size,
      return_beta = TRUE
    )
  })
  positive_betas <- lapply(positive_sources_with_beta, function(data) data$beta)
  positive_sources <- lapply(positive_sources_with_beta, function(data) data[c("x", "y")])
  negative_sources <- list()
  if (include_negative_sources) {
    negative_sources <- lapply(positive_betas, function(beta) {
      generate_synthetic_functional_data(
        n = n_source,
        beta = -beta,
        signal_strength = signal_strength,
        alpha = alpha,
        grid_size = grid_size,
        basis_size = basis_size
      )
    })
  }
  random_sources <- lapply(seq_len(n_random_sources), function(i) {
    generate_synthetic_functional_data(
      n = n_source,
      sclass = 2,
      signal_strength = signal_strength,
      alpha = alpha,
      grid_size = grid_size,
      basis_size = basis_size
    )
  })
  sources <- c(positive_sources, negative_sources, random_sources)

  source_info <- data.frame(
    source = paste0("source_", seq_along(sources)),
    type = c(
      rep("positive", length(positive_sources)),
      rep("negative", length(negative_sources)),
      rep("random", length(random_sources))
    ),
    h = c(positive_h, positive_h[seq_along(negative_sources)], rep(NA_real_, length(random_sources))),
    stringsAsFactors = FALSE
  )

  list(
    target_train = target_train,
    target_test = target_test,
    sources = sources,
    source_info = source_info,
    positive_source_indices = seq_along(positive_sources),
    negative_source_indices = seq_along(negative_sources) + length(positive_sources),
    random_source_indices = seq_along(random_sources) + length(positive_sources) + length(negative_sources)
  )
}
