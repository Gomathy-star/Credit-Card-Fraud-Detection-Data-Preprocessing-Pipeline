%%R
library(dplyr)

fraud_rate_by_state <- fact_transactions %>%
  left_join(dim_location, by="location_key") %>%
  group_by(state) %>%
  summarise(
    transactions = sum(txn_count),
    fraud_cases = sum(is_fraud),
    fraud_rate = fraud_cases / transactions
  ) %>%
  arrange(desc(fraud_rate))

print(fraud_rate_by_state)

fraud_by_payment <- fact_transactions %>%
  left_join(dim_payment, by="payment_key") %>%
  group_by(payment_method) %>%
  summarise(
    transactions = sum(txn_count),
    fraud_cases = sum(is_fraud),
    fraud_rate = fraud_cases / transactions
  ) %>%
  arrange(desc(fraud_rate))

print(fraud_by_payment)

fraud_by_merchant <- fact_transactions %>%
  left_join(dim_merchant, by="merchant_key") %>%
  group_by(merchant_category) %>%
  summarise(
    transactions = sum(txn_count),
    fraud_cases = sum(is_fraud),
    fraud_rate = fraud_cases / transactions
  ) %>%
  arrange(desc(fraud_rate))

print(fraud_by_merchant)

fraud_by_hour <- fact_transactions %>%
  left_join(dim_time, by="time_key") %>%
  group_by(hour) %>%
  summarise(
    transactions = sum(txn_count),
    fraud_cases = sum(is_fraud),
    fraud_rate = fraud_cases / transactions
  ) %>%
  arrange(desc(fraud_rate))

print(fraud_by_hour)