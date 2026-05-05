## ----setup, message=FALSE, warning=FALSE--------------------------------------
library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(pROC)

dir.create("../outputs/exploratory/fiber_diagnostics", recursive = TRUE, showWarnings = FALSE)
dir.create("../outputs/exploratory/fiber_diagnostics/plots", recursive = TRUE, showWarnings = FALSE)

model_columns <- c(
  primary_openai = "primary_openai_decision",
  primary_anthropic = "primary_anthropic_decision",
  primary_gemini = "primary_gemini_decision",
  escalation_openai = "escalation_openai_decision",
  escalation_xai = "escalation_xai_decision",
  escalation_anthropic = "escalation_anthropic_decision",
  ensemble = "final_decision"
)

model_labels <- c(
  primary_openai = "Primary OpenAI",
  primary_anthropic = "Primary Anthropic",
  primary_gemini = "Primary Gemini",
  escalation_openai = "Escalation OpenAI",
  escalation_xai = "Escalation xAI",
  escalation_anthropic = "Escalation Anthropic",
  ensemble = "Final ensemble"
)

read_stage <- function(stage) {
  path <- sprintf("../outputs/paper_analysis/fiber_%s/merged_predictions.csv", stage)
  df <- read_csv(path, show_col_types = FALSE)
  gold <- if (stage == "l1") df$gold_l1_label else df$gold_final_include
  mutate(df, gold = as.integer(gold))
}

decision_to_binary <- function(x) {
  case_when(
    x == "include" ~ 1L,
    x == "exclude" ~ 0L,
    TRUE ~ NA_integer_
  )
}

compute_metrics <- function(truth, pred) {
  keep <- !is.na(truth) & !is.na(pred)
  truth <- truth[keep]
  pred <- pred[keep]
  tp <- sum(pred == 1 & truth == 1)
  fp <- sum(pred == 1 & truth == 0)
  tn <- sum(pred == 0 & truth == 0)
  fn <- sum(pred == 0 & truth == 1)
  tibble(
    n = length(truth),
    true_positive = tp,
    false_positive = fp,
    true_negative = tn,
    false_negative = fn,
    sensitivity = ifelse(tp + fn > 0, tp / (tp + fn), NA_real_),
    specificity = ifelse(tn + fp > 0, tn / (tn + fp), NA_real_),
    balanced_accuracy = mean(c(
      ifelse(tp + fn > 0, tp / (tp + fn), NA_real_),
      ifelse(tn + fp > 0, tn / (tn + fp), NA_real_)
    ), na.rm = TRUE),
    precision = ifelse(tp + fp > 0, tp / (tp + fp), NA_real_)
  )
}

make_bins <- function(prob, truth, bins) {
  probs <- pmin(pmax(prob, 0), 1)
  cuts <- unique(quantile(probs, probs = seq(0, 1, length.out = bins + 1), na.rm = TRUE, type = 8))
  if (length(cuts) < 3) {
    cuts <- seq(0, 1, length.out = min(3, bins + 1))
  }
  bucket <- cut(probs, breaks = cuts, include.lowest = TRUE, ordered_result = TRUE)
  tibble(prob = probs, truth = truth, bucket = bucket) %>%
    group_by(bucket) %>%
    summarise(
      n = n(),
      mean_pred = mean(prob),
      obs_rate = mean(truth),
      .groups = "drop"
    ) %>%
    filter(n > 0)
}

calibration_summary <- function(prob, truth, bins) {
  bin_frame <- make_bins(prob, truth, bins)
  brier <- mean((prob - truth)^2)
  ece <- sum((bin_frame$n / sum(bin_frame$n)) * abs(bin_frame$obs_rate - bin_frame$mean_pred))

  g <- nrow(bin_frame)
  obs <- bin_frame$obs_rate * bin_frame$n
  exp <- bin_frame$mean_pred * bin_frame$n
  valid <- exp > 0 & exp < bin_frame$n
  hl_stat <- NA_real_
  hl_df <- NA_integer_
  hl_p <- NA_real_
  if (sum(valid) >= 3) {
    hl_stat <- sum(
      ((obs[valid] - exp[valid])^2 / exp[valid]) +
        (((bin_frame$n[valid] - obs[valid]) - (bin_frame$n[valid] - exp[valid]))^2 /
           (bin_frame$n[valid] - exp[valid]))
    )
    hl_df <- sum(valid) - 2
    hl_p <- 1 - pchisq(hl_stat, df = hl_df)
  }

  list(
    bins = bin_frame,
    summary = tibble(
      brier_score = brier,
      ece = ece,
      hosmer_lemeshow_stat = hl_stat,
      hosmer_lemeshow_df = hl_df,
      hosmer_lemeshow_p = hl_p
    )
  )
}

cohen_kappa_pair <- function(a, b) {
  keep <- !is.na(a) & !is.na(b)
  a <- a[keep]
  b <- b[keep]
  n <- length(a)
  if (n == 0) {
    return(tibble(n = 0, po = NA_real_, pe = NA_real_, kappa = NA_real_))
  }
  po <- mean(a == b)
  pa <- prop.table(table(factor(a, levels = c(0, 1))))
  pb <- prop.table(table(factor(b, levels = c(0, 1))))
  pe <- sum(pa * pb)
  kappa <- ifelse(pe < 1, (po - pe) / (1 - pe), NA_real_)
  tibble(n = n, po = po, pe = pe, kappa = kappa)
}

fleiss_kappa_binary <- function(mat) {
  mat <- as.matrix(mat)
  mat <- mat[complete.cases(mat), , drop = FALSE]
  if (nrow(mat) == 0) {
    return(tibble(n = 0, raters = ncol(mat), fleiss_kappa = NA_real_))
  }
  counts_include <- rowSums(mat == 1)
  counts_exclude <- rowSums(mat == 0)
  n_raters <- ncol(mat)
  p_j <- c(sum(counts_exclude), sum(counts_include)) / (nrow(mat) * n_raters)
  p_i <- ((counts_exclude^2 + counts_include^2) - n_raters) / (n_raters * (n_raters - 1))
  p_bar <- mean(p_i)
  p_e <- sum(p_j^2)
  kappa <- ifelse(p_e < 1, (p_bar - p_e) / (1 - p_e), NA_real_)
  tibble(n = nrow(mat), raters = n_raters, fleiss_kappa = kappa)
}

plot_reliability <- function(bin_frame, stage_label) {
  ggplot(bin_frame, aes(x = mean_pred, y = obs_rate, size = n)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
    geom_point(color = "#1b6ca8", alpha = 0.9) +
    geom_line(color = "#1b6ca8") +
    scale_x_continuous(limits = c(0, 1)) +
    scale_y_continuous(limits = c(0, 1)) +
    labs(
      title = paste(stage_label, "reliability diagram"),
      x = "Mean predicted include probability",
      y = "Observed include rate",
      size = "Bin n"
    ) +
    theme_minimal(base_size = 12)
}

plot_kappa_heatmap <- function(kappa_frame, stage_label) {
  ggplot(kappa_frame, aes(x = model_a, y = model_b, fill = kappa)) +
    geom_tile(color = "white") +
    geom_text(aes(label = sprintf("%.2f", kappa)), size = 3) +
    scale_fill_gradient2(low = "#b2182b", mid = "white", high = "#2166ac", midpoint = 0.5, na.value = "gray90") +
    labs(title = paste(stage_label, "pairwise Cohen's kappa"), x = NULL, y = NULL) +
    theme_minimal(base_size = 11) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}


## -----------------------------------------------------------------------------
l1 <- read_stage("l1")
l2 <- read_stage("l2")


## -----------------------------------------------------------------------------
run_calibration <- function(df, stage, bins) {
  out <- calibration_summary(df$mean_include_probability, df$gold, bins = bins)
  roc_auc <- as.numeric(pROC::auc(df$gold, df$mean_include_probability))
  summary <- out$summary %>%
    mutate(stage = stage, auc = roc_auc, n = nrow(df), bins = bins) %>%
    select(stage, n, bins, auc, everything())
  write_csv(summary, sprintf("../outputs/exploratory/fiber_diagnostics/%s_calibration_summary.csv", stage))
  write_csv(out$bins, sprintf("../outputs/exploratory/fiber_diagnostics/%s_reliability_bins.csv", stage))
  plot <- plot_reliability(out$bins, toupper(stage))
  ggsave(sprintf("../outputs/exploratory/fiber_diagnostics/plots/%s_reliability.png", stage), plot, width = 6, height = 4.5, dpi = 200)
  list(summary = summary, bins = out$bins)
}

cal_l1 <- run_calibration(l1, "l1", bins = 10)
cal_l2 <- run_calibration(l2, "l2", bins = 5)

bind_rows(cal_l1$summary, cal_l2$summary)


## -----------------------------------------------------------------------------
run_agreement <- function(df, stage) {
  model_df <- tibble(
    primary_openai = decision_to_binary(df$primary_openai_decision),
    primary_anthropic = decision_to_binary(df$primary_anthropic_decision),
    primary_gemini = decision_to_binary(df$primary_gemini_decision),
    escalation_openai = decision_to_binary(df$escalation_openai_decision),
    escalation_xai = decision_to_binary(df$escalation_xai_decision),
    escalation_anthropic = decision_to_binary(df$escalation_anthropic_decision)
  )

  keys <- names(model_df)
  pairwise <- expand.grid(model_a = keys, model_b = keys, stringsAsFactors = FALSE) %>%
    rowwise() %>%
    do({
      stats <- cohen_kappa_pair(model_df[[.$model_a]], model_df[[.$model_b]])
      bind_cols(tibble(model_a = .$model_a, model_b = .$model_b), stats)
    }) %>%
    ungroup() %>%
    mutate(
      stage = stage,
      model_a = recode(model_a, !!!model_labels),
      model_b = recode(model_b, !!!model_labels)
    )

  fleiss_all6 <- fleiss_kappa_binary(model_df) %>% mutate(stage = stage, scope = "all six models on escalated subset")
  fleiss_primary3 <- fleiss_kappa_binary(model_df[, c("primary_openai", "primary_anthropic", "primary_gemini")]) %>%
    mutate(stage = stage, scope = "primary tier only on full benchmark")

  write_csv(pairwise, sprintf("../outputs/exploratory/fiber_diagnostics/%s_pairwise_kappa.csv", stage))
  write_csv(bind_rows(fleiss_all6, fleiss_primary3), sprintf("../outputs/exploratory/fiber_diagnostics/%s_fleiss_kappa.csv", stage))

  heatmap_plot <- plot_kappa_heatmap(pairwise, toupper(stage))
  ggsave(sprintf("../outputs/exploratory/fiber_diagnostics/plots/%s_pairwise_kappa_heatmap.png", stage), heatmap_plot, width = 7, height = 6, dpi = 200)

  list(pairwise = pairwise, fleiss = bind_rows(fleiss_all6, fleiss_primary3))
}

agree_l1 <- run_agreement(l1, "l1")
agree_l2 <- run_agreement(l2, "l2")

bind_rows(agree_l1$fleiss, agree_l2$fleiss)


## -----------------------------------------------------------------------------
run_model_performance <- function(df, stage) {
  out <- lapply(names(model_columns), function(key) {
    pred <- decision_to_binary(df[[model_columns[[key]]]])
    metrics <- compute_metrics(df$gold, pred)
    metrics %>%
      mutate(
        stage = stage,
        model_key = key,
        model_label = model_labels[[key]]
      ) %>%
      select(stage, model_key, model_label, everything())
  }) %>%
    bind_rows()

  write_csv(out, sprintf("../outputs/exploratory/fiber_diagnostics/%s_per_model_performance.csv", stage))
  out
}

perf_l1 <- run_model_performance(l1, "l1")
perf_l2 <- run_model_performance(l2, "l2")

bind_rows(perf_l1, perf_l2)


## -----------------------------------------------------------------------------
tibble(
  stage = c("l1", "l2"),
  full_benchmark_n = c(nrow(l1), nrow(l2)),
  escalated_n = c(sum(l1$escalated), sum(l2$escalated)),
  non_escalated_n = c(sum(!l1$escalated), sum(!l2$escalated))
) %>%
  write_csv("../outputs/exploratory/fiber_diagnostics/subset_denominators.csv")

tibble(
  stage = c("l1", "l2"),
  full_benchmark_n = c(nrow(l1), nrow(l2)),
  escalated_n = c(sum(l1$escalated), sum(l2$escalated)),
  non_escalated_n = c(sum(!l1$escalated), sum(!l2$escalated))
)

