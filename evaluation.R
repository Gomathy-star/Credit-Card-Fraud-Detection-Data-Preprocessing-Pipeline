# ============================================
# FRAUD DETECTION - MODEL EVALUATION SCRIPT
# ============================================

# 1. Load required libraries
library(randomForest)
library(caret)
library(pROC)        # For ROC curve & AUC
library(ggplot2)     # For plots

# ============================================
# STEP 1: Load Model & Data
# ============================================

load("fraud_model.RData")  # loads rf_model

data <- read.csv("cleaned_transformed_fraud_final.csv")

data <- data[, !(names(data) %in% c("transaction_id", "customer_id", "transaction_time"))]
data <- data[, !grepl("fraud_type", names(data))]
data$transaction_amount_clean <- NULL
data$location          <- as.factor(data$location)
data$purchase_category <- as.factor(data$purchase_category)
data$is_fraudulent     <- as.factor(data$is_fraudulent)

# Recreate the same test split (same seed as model.R)
set.seed(123)
trainIndex <- sample(1:nrow(data), 0.7 * nrow(data))
test <- data[-trainIndex, ]

# ============================================
# STEP 2: Predictions
# ============================================

# Get probabilities
rf_prob <- predict(rf_model, test, type = "prob")

# Apply threshold of 0.3 (same as model.R)
rf_pred <- as.factor(ifelse(rf_prob[, 2] > 0.3, 1, 0))

# ============================================
# STEP 3: Confusion Matrix
# ============================================

cat("\n========== CONFUSION MATRIX ==========\n")
conf_matrix <- confusionMatrix(rf_pred, test$is_fraudulent, positive = "1")
print(conf_matrix)

# Extract key values
TP <- conf_matrix$table[2, 2]
TN <- conf_matrix$table[1, 1]
FP <- conf_matrix$table[2, 1]
FN <- conf_matrix$table[1, 2]

cat("\n--- Manual breakdown ---\n")
cat("True Positives  (Fraud correctly caught)  :", TP, "\n")
cat("True Negatives  (Legit correctly passed)  :", TN, "\n")
cat("False Positives (Legit flagged as fraud)  :", FP, "\n")
cat("False Negatives (Fraud missed!)           :", FN, "\n")

# ============================================
# STEP 4: Key Metrics
# ============================================

accuracy    <- conf_matrix$overall["Accuracy"]
precision   <- conf_matrix$byClass["Precision"]
recall      <- conf_matrix$byClass["Sensitivity"]   # = Recall
specificity <- conf_matrix$byClass["Specificity"]
f1_score    <- conf_matrix$byClass["F1"]

cat("\n========== KEY METRICS ==========\n")
cat(sprintf("Accuracy    : %.4f  (%.2f%%)\n", accuracy,    accuracy    * 100))
cat(sprintf("Precision   : %.4f  (%.2f%%)\n", precision,   precision   * 100))
cat(sprintf("Recall      : %.4f  (%.2f%%)\n", recall,      recall      * 100))
cat(sprintf("Specificity : %.4f  (%.2f%%)\n", specificity, specificity * 100))
cat(sprintf("F1 Score    : %.4f\n",            f1_score))

# ============================================
# STEP 5: ROC Curve & AUC Score
# ============================================

cat("\n========== AUC SCORE ==========\n")
roc_obj <- roc(as.numeric(as.character(test$is_fraudulent)),
               rf_prob[, 2])
auc_val <- auc(roc_obj)
cat(sprintf("AUC (Area Under ROC Curve): %.4f\n", auc_val))

# Interpretation
if (auc_val >= 0.9) {
  cat("Interpretation: EXCELLENT model!\n")
} else if (auc_val >= 0.8) {
  cat("Interpretation: GOOD model\n")
} else if (auc_val >= 0.7) {
  cat("Interpretation: FAIR model\n")
} else {
  cat("Interpretation: Needs improvement\n")
}

# Plot and save ROC curve
png("roc_curve.png", width = 700, height = 600)
plot(roc_obj,
     col  = "steelblue",
     lwd  = 2,
     main = paste0("ROC Curve - Fraud Detection\n(AUC = ", round(auc_val, 4), ")"),
     xlab = "False Positive Rate (1 - Specificity)",
     ylab = "True Positive Rate (Recall)")
abline(a = 0, b = 1, lty = 2, col = "gray")   # random baseline
legend("bottomright",
       legend = c(paste("Random Forest (AUC =", round(auc_val, 3), ")"), "Random Baseline"),
       col    = c("steelblue", "gray"),
       lwd    = c(2, 1),
       lty    = c(1, 2))
dev.off()
cat("ROC curve saved as: roc_curve.png\n")

# ============================================
# STEP 6: Confusion Matrix Heatmap
# ============================================

cm_df <- as.data.frame(conf_matrix$table)
colnames(cm_df) <- c("Predicted", "Actual", "Freq")

p <- ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 8, fontface = "bold", color = "white") +
  scale_fill_gradient(low = "#4575b4", high = "#d73027") +
  labs(title  = "Confusion Matrix Heatmap",
       x      = "Actual Class",
       y      = "Predicted Class",
       fill   = "Count") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

ggsave("confusion_matrix_heatmap.png", plot = p, width = 6, height = 5)
cat("Confusion matrix heatmap saved as: confusion_matrix_heatmap.png\n")

# ============================================
# STEP 7: Feature Importance (already in model.R, but cleaner version)
# ============================================

importance_df <- as.data.frame(importance(rf_model))
importance_df$Feature <- rownames(importance_df)
importance_df <- importance_df[order(-importance_df$MeanDecreaseGini), ]

cat("\n========== TOP 10 IMPORTANT FEATURES ==========\n")
print(head(importance_df[, c("Feature", "MeanDecreaseGini")], 10))

# ============================================
# STEP 8: Summary Report
# ============================================

cat("\n\n========== EVALUATION SUMMARY REPORT ==========\n")
cat("Model        : Random Forest (ntree=300, mtry=4, threshold=0.3)\n")
cat("Dataset      : Credit Card Fraud Detection\n")
cat("Test Samples :", nrow(test), "\n")
cat(sprintf("Fraud cases in test : %d (%.1f%%)\n",
            sum(test$is_fraudulent == 1),
            100 * mean(test$is_fraudulent == 1)))
cat("------------------------------------------------\n")
cat(sprintf("Accuracy    : %.2f%%\n", accuracy    * 100))
cat(sprintf("Precision   : %.2f%%\n", precision   * 100))
cat(sprintf("Recall      : %.2f%%\n", recall      * 100))
cat(sprintf("F1 Score    : %.4f\n",   f1_score))
cat(sprintf("AUC Score   : %.4f\n",   auc_val))
cat("------------------------------------------------\n")
cat("Files generated:\n")
cat("  - roc_curve.png\n")
cat("  - confusion_matrix_heatmap.png\n")
cat("  - feature_importance.png  (from model.R)\n")
cat("================================================\n")
