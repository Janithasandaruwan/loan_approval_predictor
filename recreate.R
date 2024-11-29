# Load necessary libraries
library(ggplot2)
library(pROC)

# Load and process data for probit-model creation
# Load the loan data csv file
loan_data <- read.csv("loan_data.csv")

# Handle missing values by replacing median values
loan_data$AnnualIncome[is.na(loan_data$AnnualIncome)] <- median(loan_data$AnnualIncome, na.rm = TRUE)
loan_data$LoanAmount[is.na(loan_data$LoanAmount)] <- median(loan_data$LoanAmount, na.rm = TRUE)
loan_data$LoanDuration[is.na(loan_data$LoanDuration)] <- median(loan_data$LoanDuration, na.rm = TRUE)

# Normalize continuous variables
normalize <- function(x) (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)

loan_data$AnnualIncome <- normalize(loan_data$AnnualIncome)
loan_data$CreditScore <- normalize(loan_data$CreditScore)
loan_data$LoanAmount <- normalize(loan_data$LoanAmount)
loan_data$LoanDuration <- normalize(loan_data$LoanDuration)

# Select relevant columns
data <- loan_data[, c("Age", 
                      "AnnualIncome", 
                      "MaritalStatus", 
                      "CreditScore", 
                      "TotalDebtToIncomeRatio",
                      "SavingsAccountBalance",
                      "PreviousLoanDefaults", 
                      "PaymentHistory", 
                      "LoanAmount", 
                      "LoanDuration", 
                      "LoanApproved")]

# Convert necessary variables to factors or numeric types
data$LoanApproved <- as.factor(data$LoanApproved)
data$PreviousLoanDefaults <- as.numeric(data$PreviousLoanDefaults)
data$MaritalStatus <- as.factor(data$MaritalStatus)

# Fit the probit-model with additional factors
probit_model <- glm(LoanApproved ~ Age + 
                      AnnualIncome + 
                      MaritalStatus + 
                      CreditScore +
                      PreviousLoanDefaults + 
                      PaymentHistory + 
                      SavingsAccountBalance +
                      TotalDebtToIncomeRatio + 
                      LoanAmount + 
                      LoanDuration, 
                      family = binomial(link = "probit"), data = data)

# Summary of the probit-model
summary(probit_model)

# Predicted probabilities
data$PredictedProbability <- predict(probit_model, type = "response")
################################################################################


# Create plots
# 01. Plot predicted probabilities
ggplot(data, 
       aes(x = PredictedProbability, fill = LoanApproved)) +
       geom_histogram(binwidth = 0.05, alpha = 0.7, position = "identity") +
       labs(title = "Predicted Probabilities of Loan Approval", x = "Predicted Probability", y = "Count") +
       theme_minimal()


# 02. Coefficient plot
coef_plot <- data.frame(Term = names(coef(probit_model)),
                        Estimate = coef(probit_model),
                        SE = summary(probit_model)$coefficients[, "Std. Error"])

ggplot(coef_plot, aes(x = Term, y = Estimate)) +
  geom_point() +
  geom_errorbar(aes(ymin = Estimate - SE, ymax = Estimate + SE), width = 0.2) +
  labs(title = "Coefficient Estimates with Standard Errors", x = "Variables", y = "Coefficient Estimate") +
  theme_minimal() +
  coord_flip()


# 03. Scatter plot: Predicted probability vs. actual loan approval
ggplot(data, aes(x = PredictedProbability, color = LoanApproved)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot of Predicted Probabilities",
       x = "Predicted Probability",
       y = "Density") +
  theme_minimal() +
  scale_color_manual(values = c("red", "blue"), labels = c("Rejected", "Approved")) +
  theme(legend.title = element_blank())

# 04. Histogram: Count of predicted probabilities for approved vs. rejected
ggplot(data, aes(x = PredictedProbability, fill = LoanApproved)) +
  geom_histogram(binwidth = 0.05, position = "dodge", alpha = 0.7) +
  labs(title = "Approved vs. Rejected Applications by Predicted Probability",
       x = "Predicted Probability",
       y = "Count") +
  theme_minimal() +
  scale_fill_manual(values = c("red", "blue"), labels = c("Rejected", "Approved")) +
  theme(legend.title = element_blank())

# Plot ROC curve
roc_curve <- roc(data$LoanApproved, data$PredictedProbability)
plot(roc_curve, main = "ROC Curve", col = "blue")
################################################################################


# Check the probit-model accuracy
# Classify predictions based on a threshold of 0.5
data$PredictedClass <- ifelse(data$PredictedProbability > 0.56, 1, 0)

# Convert actual loan approvals to numeric for comparison
data$ActualClass <- as.numeric(as.character(data$LoanApproved))

# Confusion Matrix
confusion_matrix <- table(Predicted = data$PredictedClass, Actual = data$ActualClass)
print(confusion_matrix)

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Model Accuracy:", round(accuracy, 4) * 100, "%"))
################################################################################


# Load approval predictor
loan_approval_predictor <- function(probit_model, data) {
  # Prompt user for input
  cat("Enter the following details to check loan approval:\n")
  
  age <- as.numeric(readline("Age (years): "))
  income <- as.numeric(readline("Annual Income (26032 SEK): "))
  marital_status <- readline("Marital Status (Single/Married/Divorced): ")
  credit_score <- as.numeric(readline("Credit Score (435): "))
  debts <- as.numeric(readline("Debt-to-Income Ratio (0 to 1 like 0.59228294): "))
  savings <- as.numeric(readline("Savings Account Balance (2158): "))
  previous_defaults <- as.numeric(readline("Previous Defaults (0 or 1): "))
  repayment_history <- as.numeric(readline("Repayment History Score (1 to 10): "))
  loan_amount <- as.numeric(readline("Loan Amount (16363 SEK): "))
  loan_duration <- as.numeric(readline("Loan Duration (24): "))
  
  # Encode Marital Status as a factor with the same levels as in training data
  marital_status_factor <- factor(marital_status, levels = c("Single", "Married", "Divorced"))
  
  # Normalize numeric inputs using the dataset's mean and sd
  normalize_data <- function(x, mean_val, sd_val) {
    (x - mean_val) / sd_val
  }
  # Normalize numeric inputs using the dataset's mean and sd
  income_norm <- normalize_data(income, mean(data$AnnualIncome), sd(data$AnnualIncome))
  credit_score_norm <- normalize_data(credit_score, mean(data$CreditScore), sd(data$CreditScore))
  loan_amount_norm <- normalize_data(loan_amount, mean(data$LoanAmount), sd(data$LoanAmount))
  loan_duration_norm <- normalize_data(loan_duration, mean(data$LoanDuration), sd(data$LoanDuration))

    # Create a data frame for prediction
  new_data <- data.frame(
    Age = age,
    AnnualIncome = income_norm,
    MaritalStatus = marital_status_factor,
    CreditScore = credit_score_norm,
    TotalDebtToIncomeRatio = debts,
    SavingsAccountBalance = savings,
    PreviousLoanDefaults = previous_defaults,
    PaymentHistory = repayment_history,
    LoanAmount = loan_amount,
    LoanDuration = loan_duration
  )
  # Ensure factor levels in new_data match training data
  new_data$MaritalStatus <- factor(new_data$MaritalStatus, levels = levels(data$MaritalStatus))
  
  # Predict loan approval
  predicted_probability <- predict(probit_model, new_data, type = "response")
  result <- ifelse(predicted_probability > 0.56, "Approved", "Rejected")
  
  cat("\nLoan Application Result:\n")
  cat("Predicted Probability of Approval:", predicted_probability, "\n")
  cat("Loan Status:", result, "\n")
}

# Run the function
loan_approval_predictor(probit_model, data)
