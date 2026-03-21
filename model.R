# ================================
# FRAUD DETECTION - RANDOM FOREST
# ================================

# 1. Load required libraries
library(randomForest)
library(caret)

# 2. Load dataset
data <- read.csv("cleaned_transformed_fraud_final.csv")

# 3. Initial preprocessing
# Remove unnecessary columns
data <- data[, !(names(data) %in% c("transaction_id", "customer_id", "transaction_time"))]

# Remove leakage columns (VERY IMPORTANT)
data <- data[, !grepl("fraud_type", names(data))]

# Remove duplicate/irrelevant column
data$transaction_amount_clean <- NULL

# Convert categorical variables
data$location <- as.factor(data$location)
data$purchase_category <- as.factor(data$purchase_category)
data$is_fraudulent <- as.factor(data$is_fraudulent)

# 4. Train-test split
set.seed(123)
trainIndex <- sample(1:nrow(data), 0.7 * nrow(data))

train <- data[trainIndex, ]
test <- data[-trainIndex, ]

# 5. Handle class imbalance (manual oversampling)
fraud <- train[train$is_fraudulent == 1, ]
non_fraud <- train[train$is_fraudulent == 0, ]

fraud_upsampled <- fraud[sample(1:nrow(fraud), nrow(non_fraud), replace = TRUE), ]

train_balanced <- rbind(non_fraud, fraud_upsampled)
train_balanced <- train_balanced[sample(nrow(train_balanced)), ]

# 6. Train Random Forest model
rf_model <- randomForest(
  is_fraudulent ~ ., 
  data = train_balanced,
  ntree = 300,
  mtry = 4,
  importance = TRUE
)

# 7. Prediction with probability
rf_prob <- predict(rf_model, test, type = "prob")

# Adjust threshold (important for fraud detection)
rf_pred <- ifelse(rf_prob[,2] > 0.3, 1, 0)
rf_pred <- as.factor(rf_pred)

# 8. Evaluation
conf_matrix <- confusionMatrix(rf_pred, test$is_fraudulent, positive = "1")
print(conf_matrix)

# 9. Feature importance plot
png("feature_importance.png")
varImpPlot(rf_model)
dev.off()

# 10. Save model
save(rf_model, file = "fraud_model.RData")

# 11. Save predictions (optional)
results <- data.frame(
  Actual = test$is_fraudulent,
  Predicted = rf_pred
)

write.csv(results, "predictions.csv", row.names = FALSE)

# ================================
# END OF SCRIPT
# ================================