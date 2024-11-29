# Load necessary libraries
library(ggplot2)
library(pROC)


# Load the loan data csv file
loan_data <- read.csv("loan_data.csv")

# Select relevant columns
data <- loan_data[, c("Age", 
                      "AnnualIncome", 
                      "MaritalStatus", 
                      "CreditScore",
                      "TotalDebtToIncomeRatio",
                      "SavingsAccountBalance",
                      "PreviousLoanDefaults", 
                      "PaymentHistory",
                      "LoanApproved")]


# Convert necessary variables to factors or numeric types
data$MaritalStatus <- as.factor(data$MaritalStatus)
data$LoanApproved <- as.factor(data$LoanApproved)

# Fit a probit model
probit_model <- glm(LoanApproved ~ Age + AnnualIncome + MaritalStatus + CreditScore +
                      PreviousLoanDefaults + PaymentHistory + SavingsAccountBalance + 
                      TotalDebtToIncomeRatio, 
                    family = binomial(link = "probit"), data = data)

# Summary of the probit model
summary(probit_model)

# Predicted probabilities
data$PredictedProbability <- predict(probit_model, type = "response")

# Plot predicted probabilities
ggplot(data, aes(x = PredictedProbability, fill = LoanApproved)) +
  geom_histogram(binwidth = 0.05, alpha = 0.7, position = "identity") +
  labs(title = "Predicted Probabilities of Loan Approval", x = "Predicted Probability", y = "Count") +
  theme_minimal()

# Coefficient plot
coef_plot <- data.frame(Term = names(coef(probit_model)),
                        Estimate = coef(probit_model),
                        SE = summary(probit_model)$coefficients[, "Std. Error"])

ggplot(coef_plot, aes(x = Term, y = Estimate)) +
  geom_point() +
  geom_errorbar(aes(ymin = Estimate - SE, ymax = Estimate + SE), width = 0.2) +
  labs(title = "Coefficient Estimates with Standard Errors", x = "Variables", y = "Coefficient Estimate") +
  theme_minimal() +
  coord_flip()


# Scatter plot: Predicted probability vs. actual loan approval
ggplot(data, aes(x = PredictedProbability, color = LoanApproved)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot of Predicted Probabilities",
       x = "Predicted Probability",
       y = "Density") +
  theme_minimal() +
  scale_color_manual(values = c("red", "blue"), labels = c("Rejected", "Approved")) +
  theme(legend.title = element_blank())

# Histogram: Count of predicted probabilities for approved vs. rejected
ggplot(data, aes(x = PredictedProbability, fill = LoanApproved)) +
  geom_histogram(binwidth = 0.05, position = "dodge", alpha = 0.7) +
  labs(title = "Approved vs. Rejected by Predicted Probability",
       x = "Predicted Probability",
       y = "Count") +
  theme_minimal() +
  scale_fill_manual(values = c("red", "blue"), labels = c("Rejected", "Approved")) +
  theme(legend.title = element_blank())




# Predict probabilities
data$PredictedProbability <- predict(probit_model, type = "response")

# Classify predictions based on a threshold of 0.5
data$PredictedClass <- ifelse(data$PredictedProbability > 0.5, 1, 0)

# Convert actual loan approvals to numeric for comparison
data$ActualClass <- as.numeric(as.character(data$LoanApproved))

# Confusion Matrix
confusion_matrix <- table(Predicted = data$PredictedClass, Actual = data$ActualClass)
print(confusion_matrix)

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Model Accuracy:", round(accuracy, 4)))

# Plot ROC curve and calculate AUC (optional)
roc_curve <- roc(data$ActualClass, data$PredictedProbability)
plot(roc_curve, main = "ROC Curve", col = "blue")
auc_value <- auc(roc_curve)
print(paste("AUC:", round(auc_value, 4)))




# Function to normalize continuous variables
normalize <- function(x, mean_val, sd_val) {
  (x - mean_val) / sd_val
}

loan_approval_predictor <- function(probit_model, data) {
  # Prompt user for input
  cat("Enter the following details to check loan approval:\n")
  
  age <- as.numeric(readline("Age (years): "))
  income <- as.numeric(readline("Annual Income: "))
  marital_status <- readline("Marital Status (Single/Married/Divorced): ")
  credit_score <- as.numeric(readline("Credit Score: "))
  previous_defaults <- as.numeric(readline("Previous Defaults (0 or 1): "))
  repayment_history <- as.numeric(readline("Repayment History Score (1 to 10): "))
  savings <- as.numeric(readline("Savings Account Balance: "))
  debts <- as.numeric(readline("Debt-to-Income Ratio (0 to 1): "))
  
  # Encode MaritalStatus as a factor with the same levels as in training data
  marital_status_factor <- factor(marital_status, levels = c("Single", "Married", "Divorced"))
  
  # Normalize numeric inputs using the dataset's mean and sd
  normalize <- function(x, mean_val, sd_val) {
    (x - mean_val) / sd_val
  }
  income_norm <- normalize(income, mean(data$AnnualIncome), sd(data$AnnualIncome))
  credit_score_norm <- normalize(credit_score, mean(data$CreditScore), sd(data$CreditScore))
  
  # Create a data frame for prediction
  new_data <- data.frame(
    Age = age,
    AnnualIncome = income_norm,
    MaritalStatus = marital_status_factor,
    CreditScore = credit_score_norm,
    PreviousLoanDefaults = previous_defaults,
    PaymentHistory = repayment_history,
    SavingsAccountBalance = savings,
    TotalDebtToIncomeRatio = debts
  )
  
  # Ensure factor levels in new_data match training data
  new_data$MaritalStatus <- factor(new_data$MaritalStatus, levels = levels(data$MaritalStatus))
  
  # Predict loan approval
  predicted_probability <- predict(probit_model, new_data, type = "response")
  result <- ifelse(predicted_probability > 0.5, "Approved", "Rejected")
  
  cat("\nLoan Application Result:\n")
  cat("Predicted Probability of Approval:", round(predicted_probability, 4), "\n")
  cat("Loan Status:", result, "\n")
}

# Run the function
loan_approval_predictor(probit_model, data)
