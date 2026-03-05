# =========================================================
# PREPROCESSING PIPELINE (RStudio)
# - Cleans columns/text
# - Cleans amount to numeric (INR)
# - Extracts ONLY useful time parts (Y/M/D/H/M/S)
# - Replaces NA: numeric -> MEDIAN, categorical -> MODE
# - Outlier capping (IQR) on amount
# - One-hot encodes categorical columns EXCEPT:
#     location, purchase_category, is_fraud
# - Scaling (z-score) on numeric columns EXCEPT is_fraud
# - Saves final CSV
# =========================================================

library(tidyverse)
library(janitor)
library(lubridate)
library(caret)
library(fastDummies)

# -------------------------
# 1) LOAD DATA
# -------------------------
df <- read.csv(file.choose(), stringsAsFactors = FALSE)

# -------------------------
# 2) DATA CLEANING
# -------------------------

# Remove duplicates
df <- df %>% distinct()

# Clean column names
df <- clean_names(df)

# Clean text columns (remove brackets + trim)
char_cols <- names(df)[sapply(df, is.character)]
df[char_cols] <- lapply(df[char_cols], function(x){
  x <- gsub("\\[|\\]|\\(|\\)|\\{|\\}", "", x)
  trimws(x)
})

# Clean currency/amount column (if present)
amount_candidates <- c("amount","transaction_amount","txn_amount","price","value")
amount_col <- amount_candidates[amount_candidates %in% names(df)][1]

if(!is.na(amount_col)){
  df[[amount_col]] <- gsub("₹|INR|Rs\\.|rs\\.|,", "", df[[amount_col]])
  df[[amount_col]] <- as.numeric(df[[amount_col]])
  
  names(df)[names(df) == amount_col] <- paste0(amount_col, "_inr")
  amount_col <- paste0(amount_col, "_inr")
}

# -------------------------
# 3) DATA TRANSFORMATION
# -------------------------

# Split combined columns if any (optional)
combo_candidates <- c("ram_rom","memory","device_memory")
combo_col <- combo_candidates[combo_candidates %in% names(df)][1]

if(!is.na(combo_col)){
  df <- df %>%
    separate(.data[[combo_col]], into = c("part1","part2"), sep = "/", fill = "right")
}

# Extract ONLY useful time features from transaction_time/date/timestamp
time_candidates <- c("transaction_time","transaction_date","date","timestamp","time")
time_col <- time_candidates[time_candidates %in% names(df)][1]

if(!is.na(time_col)){
  df[[time_col]] <- parse_date_time(df[[time_col]],
                                    orders = c("mdy HMS","dmy HMS","ymd HMS","mdy HM","dmy HM","ymd HM","ymd"),
                                    tz = "Asia/Kolkata")
  
  df <- df %>%
    mutate(
      txn_year   = year(.data[[time_col]]),
      txn_month  = month(.data[[time_col]]),
      txn_day    = day(.data[[time_col]]),
      txn_hour   = hour(.data[[time_col]]),
      txn_minute = minute(.data[[time_col]]),
      txn_second = second(.data[[time_col]])
    )
}

# -------------------------
# 4) MISSING VALUE IMPUTATION
# -------------------------

# Numeric NA -> median
num_cols <- names(df)[sapply(df, is.numeric)]
for(col in num_cols){
  if(any(is.na(df[[col]]))){
    med <- median(df[[col]], na.rm = TRUE)
    df[[col]][is.na(df[[col]])] <- med
  }
}

# Categorical NA -> mode
cat_cols <- names(df)[sapply(df, is.character)]
for(col in cat_cols){
  if(any(is.na(df[[col]]))){
    mode_val <- names(sort(table(df[[col]]), decreasing = TRUE))[1]
    df[[col]][is.na(df[[col]])] <- mode_val
  }
}

# -------------------------
# 5) OUTLIER HANDLING (IQR CAPPING on amount)
# -------------------------
if(!is.na(amount_col) && amount_col %in% names(df)){
  Q1 <- quantile(df[[amount_col]], 0.25, na.rm = TRUE)
  Q3 <- quantile(df[[amount_col]], 0.75, na.rm = TRUE)
  IQRv <- Q3 - Q1
  lower <- Q1 - 1.5 * IQRv
  upper <- Q3 + 1.5 * IQRv
  df[[amount_col]] <- pmin(pmax(df[[amount_col]], lower), upper)
}

# -------------------------
# 6) ENCODING (EXCEPT location, purchase_category, is_fraud)
# -------------------------
exclude_cols <- c("location", "purchase_category", "is_fraud")
cat_cols <- names(df)[sapply(df, is.character)]
encode_cols <- setdiff(cat_cols, exclude_cols)

if(length(encode_cols) > 0){
  df <- dummy_cols(df,
                   select_columns = encode_cols,
                   remove_selected_columns = TRUE,
                   remove_first_dummy = TRUE)
}

# -------------------------
# 7) FEATURE SCALING (Exclude is_fraud)
# -------------------------
target_col <- if ("is_fraud" %in% names(df)) "is_fraud" else NA

num_cols <- names(df)[sapply(df, is.numeric)]
num_cols_scale <- setdiff(num_cols, target_col)

if(length(num_cols_scale) > 0){
  pre_obj <- preProcess(df[, num_cols_scale, drop = FALSE], method = c("center","scale"))
  df[, num_cols_scale] <- predict(pre_obj, df[, num_cols_scale, drop = FALSE])
}

# -------------------------
# 8) FINAL NA CHECK + SAVE
# -------------------------
na_left <- colSums(is.na(df))
na_left <- na_left[na_left > 0]

cat("\nNA remaining (should be none):\n")
if(length(na_left) == 0) cat("0 NA remaining \n") else print(na_left)

write.csv(df, "cleaned_transformed_fraud_final.csv", row.names = FALSE)
cat("\nSaved: cleaned_transformed_fraud_final.csv\n")
cat("Final dataset shape:", nrow(df), "rows x", ncol(df), "columns\n")