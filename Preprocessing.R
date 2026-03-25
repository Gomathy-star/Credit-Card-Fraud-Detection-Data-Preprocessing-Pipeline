#***********************************************************
# CREDIT CARD FRAUD DATA PREPROCESSING PIPELINE
#***********************************************************
# This script performs:
# 1. DATA CLEANING
#    - removes duplicate rows
#    - standardizes column names
#    - cleans text values
#    - creates a clean numeric copy of transaction amount
#    - removes rows without transaction time/date
#    - handles missing values
#
# 2. DATA TRANSFORMATION
#    - extracts useful time-based features
#    - handles outliers in cleaned amount column
#    - encodes selected categorical variables
#
# 3. FINAL OUTPUT PREPARATION
#    - keeps important columns readable
#    - saves a clean and sensible CSV file
#***********************************************************

#***********************************************************
# LOAD REQUIRED LIBRARIES
#***********************************************************
library(tidyverse)
library(janitor)
library(lubridate)
library(fastDummies)

#***********************************************************
# STEP 1: LOAD DATASET
#***********************************************************
# Reads the CSV file selected by the user
df <- read.csv(file.choose(), stringsAsFactors = FALSE)

#***********************************************************
#***********************DATA CLEANING***********************
#***********************************************************

#***********************************************************
# CLEANING 1: REMOVE DUPLICATES
#***********************************************************
# Removes repeated rows to avoid duplicate transaction records
df <- df %>% distinct()

#***********************************************************
# CLEANING 2: STANDARDIZE COLUMN NAMES
#***********************************************************
# Converts column names to lowercase and replaces spaces with underscores
df <- clean_names(df)

#***********************************************************
# CLEANING 3: CLEAN TEXT COLUMNS
#***********************************************************
# Removes brackets () [] {} and trims extra spaces from text columns
char_cols <- names(df)[sapply(df, is.character)]

df[char_cols] <- lapply(df[char_cols], function(x) {
  x <- gsub("\\[|\\]|\\(|\\)|\\{|\\}", "", x)
  trimws(x)
})

#***********************************************************
# CLEANING 4: DETECT IMPORTANT COLUMNS
#***********************************************************
# Detect likely amount and time columns from common possible names
amount_candidates <- c("transaction_amount", "amount", "txn_amount", "price", "value")
amount_col <- amount_candidates[amount_candidates %in% names(df)][1]

time_candidates <- c("transaction_time", "transaction_date", "date", "timestamp", "time")
time_col <- time_candidates[time_candidates %in% names(df)][1]

cat("Detected amount column:", amount_col, "\n")
cat("Detected time column:", time_col, "\n")


#***********************************************************
# CLEANING 5: CREATE CLEAN NUMERIC COPY OF AMOUNT
#***********************************************************
# Keeps original amount column unchanged
# Creates a separate numeric column for analysis/modeling
if (!is.na(amount_col)) {
  df$transaction_amount_clean <- gsub("₹|INR|Rs\\.|rs\\.|,", "", df[[amount_col]])
  df$transaction_amount_clean <- as.numeric(df$transaction_amount_clean)
}

#***********************************************************
# CLEANING 6: REMOVE ROWS WITHOUT TRANSACTION TIME
#***********************************************************
# Rows without transaction time/date are not useful for time-based analysis
# so they are removed before feature extraction
if (!is.na(time_col)) {
  before_rows <- nrow(df)
  
  df <- df %>%
    filter(!is.na(.data[[time_col]]) & .data[[time_col]] != "")
  
  after_rows <- nrow(df)
  
  cat("Rows removed due to missing transaction time/date:", before_rows - after_rows, "\n")
}

#***********************************************************
# CLEANING 7: HANDLE MISSING VALUES
#***********************************************************
# Numeric NA values -> replaced with MEDIAN of the column
# Categorical NA values -> replaced with MODE of the column

# Numeric NA imputation
num_cols <- names(df)[sapply(df, is.numeric)]

for (col in num_cols) {
  if (any(is.na(df[[col]]))) {
    med <- median(df[[col]], na.rm = TRUE)
    df[[col]][is.na(df[[col]])] <- med
    cat("Filled numeric NA in:", col, "using median =", med, "\n")
  }
}

# Categorical NA imputation
cat_cols <- names(df)[sapply(df, is.character)]

for (col in cat_cols) {
  if (any(is.na(df[[col]]))) {
    mode_val <- names(sort(table(df[[col]]), decreasing = TRUE))[1]
    df[[col]][is.na(df[[col]])] <- mode_val
    cat("Filled categorical NA in:", col, "using mode =", mode_val, "\n")
  }
}

#***********************************************************
#********************DATA TRANSFORMATION********************
#***********************************************************

#***********************************************************
# TRANSFORMATION 1: EXTRACT TIME FEATURES
#***********************************************************
# Dataset date format = MM/DD/YYYY HH:MM
# Example: 11/24/2023 22:39
if (!is.na(time_col)) {
  
  # Force MM/DD/YYYY HH:MM format
  parsed_time <- mdy_hm(df[[time_col]], tz = "Asia/Kolkata")
  
  # If some rows include seconds, fill them using mdy_hms where needed
  bad_rows <- is.na(parsed_time) & !is.na(df[[time_col]]) & df[[time_col]] != ""
  if (any(bad_rows)) {
    parsed_time[bad_rows] <- mdy_hms(df[[time_col]][bad_rows], tz = "Asia/Kolkata")
  }
  
  df$transaction_year   <- year(parsed_time)
  df$transaction_month  <- month(parsed_time)
  df$transaction_day    <- day(parsed_time)
  df$transaction_hour   <- hour(parsed_time)
  df$transaction_minute <- minute(parsed_time)
  df$transaction_second <- second(parsed_time)
}

#***********************************************************
# TRANSFORMATION 2: HANDLE OUTLIERS IN CLEAN AMOUNT COLUMN
#***********************************************************
# Uses IQR capping on the cleaned numeric amount column only
# Original amount column remains unchanged
if ("transaction_amount_clean" %in% names(df)) {
  
  Q1 <- quantile(df$transaction_amount_clean, 0.25, na.rm = TRUE)
  Q3 <- quantile(df$transaction_amount_clean, 0.75, na.rm = TRUE)
  IQRv <- Q3 - Q1
  
  lower <- Q1 - 1.5 * IQRv
  upper <- Q3 + 1.5 * IQRv
  
  df$transaction_amount_clean <- pmin(pmax(df$transaction_amount_clean, lower), upper)
}

#***********************************************************
# TRANSFORMATION 3: ENCODE SELECTED CATEGORICAL VARIABLES
#***********************************************************
# One-hot encoding is applied only to selected text columns
# These important columns are NOT encoded:
# transaction_id, merchant_id, customer_id, customer_age,
# location, purchase_category, is_fraud, original amount, original time
exclude_encode <- c("transaction_id",
                    "merchant_id",
                    "customer_id",
                    "customer_age",
                    "location",
                    "purchase_category",
                    "is_fraud",
                    amount_col,
                    time_col)

cat_cols <- names(df)[sapply(df, is.character)]
encode_cols <- setdiff(cat_cols, exclude_encode)

if (length(encode_cols) > 0) {
  df <- dummy_cols(df,
                   select_columns = encode_cols,
                   remove_selected_columns = TRUE,
                   remove_first_dummy = TRUE)
}

#***********************************************************
#*****************FINAL DATASET ORGANIZATION****************
# #***********************************************************


#***********************************************************
# ORGANIZE COLUMNS FOR READABLE OUTPUT
#***********************************************************
# Places important original columns first
front_cols <- c("transaction_id",
                "merchant_id",
                "customer_id",
                "customer_age",
                amount_col,
                "transaction_amount_clean",
                "location",
                "purchase_category",
                "is_fraud",
                time_col,
                "transaction_year",
                "transaction_month",
                "transaction_day",
                "transaction_hour",
                "transaction_minute",
                "transaction_second")

front_cols <- front_cols[front_cols %in% names(df)]
other_cols <- setdiff(names(df), front_cols)

df <- df[, c(front_cols, other_cols)]

#***********************************************************
#*********************** FINAL CHECK ***********************
#***********************************************************
# Check if any NA values still remain
na_counts <- colSums(is.na(df))

if (sum(na_counts) == 0) {
  cat("\nAll missing values handled successfully.\n")
} else {
  cat("\nRemaining missing values:\n")
  print(na_counts[na_counts > 0])
}

# Display final size of dataset
cat("\nFinal dataset shape:", nrow(df), "rows x", ncol(df), "columns\n")


# Save the cleaned and transformed dataset
write.csv(df, "cleaned_transformed_fraud_final.csv", row.names = FALSE)

cat("File saved successfully as: cleaned_transformed_fraud_final.csv\n")
