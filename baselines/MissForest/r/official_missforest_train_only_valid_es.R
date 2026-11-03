args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 11) {
  stop(paste(
    "Usage: Rscript official_missforest_train_only_valid_es.R",
    "<input_csv> <output_dir> <levels_csv> <cat_cols_or___NONE__>",
    "<train_rows> <max_iterations> <seed> <ntree> <num_threads>",
    "<verbose> <mice_method_unused>"
  ))
}

input_path <- args[[1]]
output_dir <- args[[2]]
levels_path <- args[[3]]
cat_arg <- args[[4]]
train_rows <- as.integer(args[[5]])
max_iterations <- as.integer(args[[6]])
seed <- as.integer(args[[7]])
ntree <- as.integer(args[[8]])
num_threads <- as.integer(args[[9]])
verbose_flag <- as.integer(args[[10]]) > 0

if (!requireNamespace("missForest", quietly = TRUE)) {
  stop("The official R package 'missForest' is required.")
}
if (!requireNamespace("ranger", quietly = TRUE)) {
  stop("The R package 'ranger', the default backend of missForest 1.6.1, is required.")
}

df <- read.csv(
  input_path,
  stringsAsFactors = FALSE,
  na.strings = c("", "NA", "NaN", "nan", "<NA>"),
  check.names = FALSE
)
if (train_rows < 1 || train_rows >= nrow(df)) {
  stop("train_rows must identify a non-empty training prefix and a non-empty target suffix.")
}
if (max_iterations < 1 || ntree < 1 || num_threads < 1) {
  stop("max_iterations, ntree, and num_threads must be positive.")
}

cat_cols <- character(0)
if (!identical(cat_arg, "__NONE__") && nchar(cat_arg) > 0) {
  cat_cols <- strsplit(cat_arg, ",", fixed = TRUE)[[1]]
}
levels_df <- read.csv(levels_path, stringsAsFactors = FALSE, check.names = FALSE)

for (column in setdiff(names(df), cat_cols)) {
  df[[column]] <- as.numeric(df[[column]])
}
for (column in cat_cols) {
  column_levels <- as.character(levels_df$level[levels_df$column == column])
  if (length(column_levels) == 0) {
    column_levels <- unique(na.omit(as.character(df[seq_len(train_rows), column])))
  }
  if (length(column_levels) == 0) {
    column_levels <- "__UNK__"
  }
  df[[column]] <- factor(as.character(df[[column]]), levels = unique(column_levels))
}

# missForest::missForest() imputes the supplied data frame transductively and
# returns no fitted forests/newdata transform. This adapter follows the update
# loop and ranger defaults in missForest 1.6.1, while changing only row scope:
# initialization and forest fitting use training rows; target rows are predicted.
train_index <- seq_len(train_rows)
n <- nrow(df)
p <- ncol(df)
train_mask <- seq_len(n) <= train_rows
missing_location <- is.na(df)
variable_type <- vapply(
  df,
  function(column) {
    if (is.numeric(column)) return("numeric")
    if (is.factor(column)) return("factor")
    stop("All columns must be numeric or factor.")
  },
  character(1)
)

mode_from_train <- function(values, fallback) {
  observed <- as.character(values[!is.na(values)])
  if (length(observed) == 0) return(fallback)
  counts <- table(observed)
  winners <- names(counts)[counts == max(counts)]
  sample(winners, 1)
}

set.seed(seed)
ximp <- df
for (column_index in seq_len(p)) {
  train_values <- df[train_index, column_index]
  missing_rows <- is.na(df[[column_index]])
  if (variable_type[[column_index]] == "numeric") {
    observed <- train_values[!is.na(train_values)]
    fill_value <- if (length(observed) > 0) mean(observed) else 0
    ximp[[column_index]][missing_rows] <- fill_value
  } else {
    fallback <- levels(df[[column_index]])[[1]]
    fill_value <- mode_from_train(train_values, fallback)
    initialized <- as.character(ximp[[column_index]])
    initialized[missing_rows] <- fill_value
    ximp[[column_index]] <- factor(initialized, levels = levels(df[[column_index]]))
  }
}

train_missing_count <- colSums(missing_location[train_index, , drop = FALSE])
visit_order <- order(train_missing_count)
mtry <- max(1L, floor(sqrt(p)))

fit_and_predict_column <- function(column_index, current_data) {
  observed_train <- train_mask & !missing_location[, column_index]
  missing_rows <- missing_location[, column_index]
  response <- current_data[observed_train, column_index]
  predictors_train <- current_data[observed_train, -column_index, drop = FALSE]
  predictors_missing <- current_data[missing_rows, -column_index, drop = FALSE]

  if (ncol(predictors_train) == 0 || sum(observed_train) < 2) {
    if (variable_type[[column_index]] == "numeric") {
      fill <- mean(as.numeric(response), na.rm = TRUE)
      if (!is.finite(fill)) fill <- 0
      return(rep(fill, sum(missing_rows)))
    }
    fallback <- levels(df[[column_index]])[[1]]
    fill <- mode_from_train(response, fallback)
    return(factor(rep(fill, sum(missing_rows)), levels = levels(df[[column_index]])))
  }

  effective_mtry <- min(mtry, ncol(predictors_train))
  if (variable_type[[column_index]] == "numeric") {
    forest <- ranger::ranger(
      x = predictors_train,
      y = as.numeric(response),
      num.trees = ntree,
      mtry = effective_mtry,
      replace = TRUE,
      sample.fraction = 1,
      min.bucket = 5,
      write.forest = TRUE,
      oob.error = TRUE,
      num.threads = num_threads,
      verbose = FALSE
    )
    if (sum(missing_rows) == 0) return(numeric(0))
    return(predict(forest, data = predictors_missing)$predictions)
  }

  response <- droplevels(factor(response))
  if (nlevels(response) < 2) {
    fill <- as.character(response[[1]])
    return(factor(rep(fill, sum(missing_rows)), levels = levels(df[[column_index]])))
  }
  forest <- ranger::ranger(
    x = predictors_train,
    y = response,
    num.trees = ntree,
    mtry = effective_mtry,
    replace = TRUE,
    sample.fraction = 1,
    min.bucket = 1,
    write.forest = TRUE,
    oob.error = TRUE,
    respect.unordered.factors = "order",
    num.threads = num_threads,
    verbose = FALSE
  )
  if (sum(missing_rows) == 0) {
    return(factor(character(0), levels = levels(df[[column_index]])))
  }
  predicted <- as.character(predict(forest, data = predictors_missing)$predictions)
  factor(predicted, levels = levels(df[[column_index]]))
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
metadata <- data.frame(
  key = c(
    "package", "package_version", "algorithm_source", "backend",
    "backend_version", "fit_rows", "target_rows_prediction_only", "seed",
    "ntree", "num_threads"
  ),
  value = c(
    "missForest", as.character(utils::packageVersion("missForest")),
    "missForest_1.6.1_update_loop_train_only_adapter", "ranger",
    as.character(utils::packageVersion("ranger")), as.character(train_rows),
    "TRUE", as.character(seed), as.character(ntree), as.character(num_threads)
  ),
  stringsAsFactors = FALSE
)
write.csv(metadata, file.path(output_dir, "runtime_metadata.csv"), row.names = FALSE)

for (iteration in seq_len(max_iterations)) {
  iteration_start <- proc.time()[[3]]
  previous <- ximp
  for (column_index in visit_order) {
    predictions <- fit_and_predict_column(column_index, ximp)
    missing_rows <- missing_location[, column_index]
    if (!any(missing_rows)) next
    if (variable_type[[column_index]] == "numeric") {
      ximp[[column_index]][missing_rows] <- as.numeric(predictions)
    } else {
      values <- as.character(ximp[[column_index]])
      predicted_values <- as.character(predictions)
      fallback <- levels(df[[column_index]])[[1]]
      predicted_values[is.na(predicted_values)] <- fallback
      values[missing_rows] <- predicted_values
      ximp[[column_index]] <- factor(values, levels = levels(df[[column_index]]))
    }
  }

  numeric_columns <- which(variable_type == "numeric")
  factor_columns <- which(variable_type == "factor")
  numeric_change <- NA_real_
  factor_change <- NA_real_
  if (length(numeric_columns) > 0) {
    numerator <- sum((as.matrix(ximp[, numeric_columns, drop = FALSE]) -
      as.matrix(previous[, numeric_columns, drop = FALSE]))^2)
    denominator <- sum(as.matrix(ximp[, numeric_columns, drop = FALSE])^2)
    numeric_change <- if (denominator > 0) numerator / denominator else 0
  }
  if (length(factor_columns) > 0) {
    changed <- sum(
      as.character(as.matrix(ximp[, factor_columns, drop = FALSE])) !=
        as.character(as.matrix(previous[, factor_columns, drop = FALSE]))
    )
    factor_change <- changed / (n * length(factor_columns))
  }

  output_path <- file.path(output_dir, sprintf("iteration_%03d.csv", iteration))
  completed <- ximp
  for (column in cat_cols) completed[[column]] <- as.character(completed[[column]])
  write.csv(completed, output_path, row.names = FALSE, na = "")
  cat(sprintf(
    "[r-iteration] method=missForest-official-derived iteration=%d/%d numeric_change=%s factor_change=%s sec=%.2f output=%s\n",
    iteration,
    max_iterations,
    format(numeric_change, digits = 6),
    format(factor_change, digits = 6),
    proc.time()[[3]] - iteration_start,
    output_path
  ))
  flush.console()
  if (verbose_flag) {
    cat(sprintf("[r-detail] visit_order=%s\n", paste(visit_order, collapse = ",")))
  }
}
