if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes", repos = "https://cloud.r-project.org")
}

remotes::install_version("mice", version = "3.19.0", repos = "https://cloud.r-project.org")
remotes::install_version("missForest", version = "1.6.1", repos = "https://cloud.r-project.org")
install.packages("ranger", repos = "https://cloud.r-project.org")
