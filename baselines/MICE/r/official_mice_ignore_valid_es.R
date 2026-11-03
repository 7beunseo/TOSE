args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 11) {
  stop(paste(
    "Usage: Rscript official_mice_ignore_valid_es.R",
    "<input_csv> <output_dir> <levels_csv> <cat_cols_or___NONE__>",
    "<train_rows> <max_iterations> <seed> <rf_estimators_unused>",
    "<num_threads_unused> <verbose> <categorical_method>"
  ))
}

input_path <- args[[1]]
output_dir <- args[[2]]
levels_path <- args[[3]]
cat_arg <- args[[4]]
train_rows <- as.integer(args[[5]])
max_iterations <- as.integer(args[[6]])
seed <- as.integer(args[[7]])
verbose_flag <- as.integer(args[[10]]) > 0
categorical_method <- tolower(args[[11]])

if (!requireNamespace("mice", quietly = TRUE)) {
  stop("The official R package 'mice' is required.")
}
if (!(categorical_method %in% c("default", "cart"))) {
  stop("categorical_method must be 'default' or 'cart'.")
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
if (max_iterations < 1) {
  stop("max_iterations must be positive.")
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

train_data <- df[seq_len(train_rows), , drop = FALSE]
target_data <- df[(train_rows + 1):nrow(df), , drop = FALSE]

# A one-level factor cannot be used in an R model matrix. Impute train-constant
# columns deterministically from the training prefix, then exclude them from
# the chained equations. This uses no information from the target suffix.
constant_columns <- character(0)
for (column in names(train_data)) {
  train_values <- train_data[[column]]
  observed_values <- train_values[!is.na(train_values)]
  unique_observed <- unique(as.character(observed_values))
  if (length(unique_observed) <= 1) {
    if (length(unique_observed) == 1) {
      fill_value <- unique_observed[[1]]
    } else if (is.factor(train_values) && length(levels(train_values)) > 0) {
      fill_value <- levels(train_values)[[1]]
    } else {
      fill_value <- "0"
    }

    if (is.factor(train_values)) {
      train_filled <- as.character(train_data[[column]])
      target_filled <- as.character(target_data[[column]])
      train_filled[is.na(train_filled)] <- fill_value
      target_filled[is.na(target_filled)] <- fill_value
      train_data[[column]] <- factor(train_filled, levels = levels(train_values))
      target_data[[column]] <- factor(target_filled, levels = levels(train_values))
    } else {
      numeric_fill <- as.numeric(fill_value)
      train_data[[column]][is.na(train_data[[column]])] <- numeric_fill
      target_data[[column]][is.na(target_data[[column]])] <- numeric_fill
    }
    constant_columns <- c(constant_columns, column)
  }
}

# Fit the chain on training rows only. New rows are later passed through the
# official mice.mids(newdata=...) path, which marks them ignored for fitting.
data_init <- train_data
for (column in names(train_data)) {
  train_values <- train_data[[column]]
  if (is.numeric(train_data[[column]])) {
    observed <- train_values[!is.na(train_values)]
    fill_value <- if (length(observed) > 0) mean(observed) else 0
    data_init[[column]][is.na(data_init[[column]])] <- fill_value
  } else {
    observed <- as.character(train_values[!is.na(train_values)])
    if (length(observed) > 0) {
      counts <- sort(table(observed), decreasing = TRUE)
      fill_value <- names(counts)[[1]]
    } else {
      fill_value <- levels(train_data[[column]])[[1]]
    }
    initialized <- as.character(data_init[[column]])
    initialized[is.na(initialized)] <- fill_value
    data_init[[column]] <- factor(initialized, levels = levels(train_data[[column]]))
  }
}

method <- mice::make.method(train_data)
if (categorical_method == "cart") {
  for (column in cat_cols) {
    if (column %in% names(method) && any(is.na(train_data[[column]]))) {
      method[[column]] <- "cart"
    }
  }
}

predictor_matrix <- mice::make.predictorMatrix(train_data)
diag(predictor_matrix) <- 0
if (length(constant_columns) > 0) {
  predictor_matrix[constant_columns, ] <- 0
  predictor_matrix[, constant_columns] <- 0
  method[constant_columns] <- ""
}

# Some predictors vary globally but collapse to one level among rows where a
# particular target is observed. Remove only those target-predictor edges.
removed_predictor_edges <- 0
for (target_column in names(train_data)) {
  target_observed <- !is.na(train_data[[target_column]])
  if (!any(target_observed)) {
    next
  }
  for (predictor_column in setdiff(names(train_data), target_column)) {
    predictor_values <- data_init[[predictor_column]][target_observed]
    unique_predictor_values <- unique(as.character(predictor_values[!is.na(predictor_values)]))
    if (length(unique_predictor_values) < 2 && predictor_matrix[target_column, predictor_column] != 0) {
      predictor_matrix[target_column, predictor_column] <- 0
      removed_predictor_edges <- removed_predictor_edges + 1
    }
  }
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
metadata <- data.frame(
  key = c(
    "package", "package_version", "entry_point", "newdata_entry_point",
    "fit_rows", "target_rows_ignored", "categorical_method", "seed",
    "train_constant_columns", "removed_single_level_predictor_edges"
  ),
  value = c(
    "mice", as.character(utils::packageVersion("mice")), "mice::mice",
    "mice::mice.mids(newdata=...)", as.character(train_rows), "TRUE",
    categorical_method, as.character(seed),
    if (length(constant_columns) > 0) paste(constant_columns, collapse = ",") else "__NONE__",
    as.character(removed_predictor_edges)
  ),
  stringsAsFactors = FALSE
)
write.csv(metadata, file.path(output_dir, "runtime_metadata.csv"), row.names = FALSE)

set.seed(seed)
imputed <- mice::mice(
  train_data,
  m = 1,
  maxit = 1,
  seed = seed,
  method = method,
  predictorMatrix = predictor_matrix,
  data.init = data_init,
  printFlag = verbose_flag,
  remove.collinear = FALSE,
  remove.constant = FALSE,
  nnet.MaxNWts = 100000
)

cat(sprintf(
  "[r-predictor-guard] train_constant_columns=%s removed_single_level_edges=%d\n",
  if (length(constant_columns) > 0) paste(constant_columns, collapse = ",") else "__NONE__",
  removed_predictor_edges
))
flush.console()

for (iteration in seq_len(max_iterations)) {
  if (iteration > 1) {
    imputed <- mice::mice.mids(
      imputed,
      maxit = 1,
      printFlag = verbose_flag,
      nnet.MaxNWts = 100000
    )
  }
  target_imputed <- mice::mice.mids(
    imputed,
    newdata = target_data,
    maxit = 1,
    printFlag = verbose_flag,
    nnet.MaxNWts = 100000
  )
  completed_train <- mice::complete(imputed, 1)
  completed_target <- mice::complete(target_imputed, 1)
  completed <- rbind(completed_train, completed_target)
  output_path <- file.path(output_dir, sprintf("iteration_%03d.csv", iteration))
  write.csv(completed, output_path, row.names = FALSE, na = "")
  cat(sprintf("[r-iteration] method=MICE-official-R iteration=%d/%d output=%s\n", iteration, max_iterations, output_path))
  flush.console()
}
