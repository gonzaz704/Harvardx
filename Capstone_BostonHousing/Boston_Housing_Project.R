# Loading/installing packages
if(!require(randomForest)) install.packages("randomForest")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(knitr)) install.packages("knitr")
if(!require(kableExtra)) install.packages("kableExtra")
if(!require(glmnet)) install.packages("glmnet")
if(!require(corrplot)) install.packages("corrplot")

# We first load the "Boston Housing" dataset into R. It could be loaded from the library or uncomment the url to load the dataset directly from the .csv file 
library(randomForest)
library(MASS)
data(Boston)
# Boston <- read.csv("https://raw.githubusercontent.com/gonzaz704/Edx/main/HousingData.csv")


# View the first few rows of the dataset
kable(head(Boston))


# Define a data frame with variable names and descriptions
vars <- c("crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "black", "lstat", "medv")
desc <- c("per capita crime rate by town", "proportion of residential land zoned for lots over 25,000 sq.ft.", "proportion of non-retail business acres per town", "Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)", "nitrogen oxides concentration (parts per 10 million)", "average number of rooms per dwelling", "proportion of owner-occupied units built prior to 1940", "weighted distances to five Boston employment centres", "index of accessibility to radial highways", "full-value property-tax rate per $10,000", "pupil-teacher ratio by town", "1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town", "lower status of the population (percent)", "median value of owner-occupied homes in $1000s")
var_desc_df <- data.frame(variable = vars, description = desc)

# Create a formatted table using kable and kableExtra
kable(var_desc_df, format = "latex", booktabs = TRUE) %>%
  kable_styling(latex_options = "striped")


# Splitting train/test set
set.seed(123)
train_index <- sample(nrow(Boston), floor(0.7*nrow(Boston)))
Boston_train <- Boston[train_index,]
Boston_test <- Boston[-train_index,]


# Handling Missing Values
colSums(is.na(Boston))

# Boxplot
boxplot(Boston, las = 2)

# Scaling the Data
Boston_scaled <- as.data.frame(scale(Boston_train))
summary(Boston_scaled)


# Scatter plot of the 'rm' and 'medv' variables in the 'Boston_scaled' dataset
ggplot(Boston_scaled, aes(x = rm, y = medv)) + 
  geom_point() + 
  ggtitle("Relationship between Number of Rooms and Median Value of 
          Owner-Occupied Homes in thousands of dollars")


# Correlation Matrix
corr <- cor(Boston_scaled[, -ncol(Boston_scaled)])
corrplot::corrplot(corr, method = "color", type = "upper", order = "hclust")

# compute correlation matrix between all variables
corr_matrix <- cor(Boston_scaled)

# select the column of correlations with "medv"
medv_correlations <- corr_matrix[, "medv"]

# view the correlation values
print(medv_correlations)


# Linear Regression Model
model_lm <- lm(medv ~ ., data = Boston_train)
summary(model_lm)
rmse_lm <- sqrt(mean((predict(model_lm, Boston_test) - Boston_test$medv)^2))


# Result Linear regression Model
results_table <- data.frame(Method = "Linear regression model", RMSE = rmse_lm)
kable(results_table, align = "c", col.names = c("Method", "RMSE"))


# Random Forest Model
model_rf <- randomForest(medv ~ ., data = Boston_train, ntree = 500, importance = TRUE)
print(model_rf)


# Variable Importance
varImpPlot(model_rf, type = 1, main = "Variable Importance in Random Forest Model")

# Result Random Forest Model
rmse_rf <- sqrt(mean((predict(model_rf, Boston_test) - Boston_test$medv)^2))
results_table <- data.frame(Method = "Random Forest model", RMSE = rmse_rf)
kable(results_table, align = "c", col.names = c("Method", "RMSE"))


# Fitting a Ridge Regression Model using cross-validation
model_ridge <- cv.glmnet(x = as.matrix(Boston_train[, -ncol(Boston_train)]), y = Boston_train$medv, alpha = 0, nfolds = 10, standardize = TRUE)

# Plotting the cross-validation results
plot(model_ridge, main = '')

# Showing optimal value of lambda (the tuning parameter that controls the amount of regularization)
model_ridge$lambda.min

# Showing the coefficients of the model at the optimal value of lambda
coef(model_ridge, s = model_ridge$lambda.min)


# Result Ridge Regression Model
rmse_ridge <- sqrt(mean((predict(model_ridge, as.matrix(Boston_test[, -ncol(Boston_test)])) - Boston_test$medv)^2))
results_table <- data.frame(Method = "Ridge Regression model", RMSE = rmse_ridge)
kable(results_table, align = "c", col.names = c("Method", "RMSE"))


# Build the linear regression model 
model_lm <- lm(medv ~ ., data = Boston_train)

# Identify the most important variables
important_vars <- c("rm", "lstat", "ptratio")

# Build the ridge regression model using only the important variables
X_train <- as.matrix(Boston_train[, important_vars])
y_train <- Boston_train$medv
X_test <- as.matrix(Boston_test[, important_vars])

# Use cross-validation to select the optimal regularization parameter
model_ridge2 <- cv.glmnet(X_train, y_train, alpha = 0, nfolds = 10, standardize = TRUE)

# Make predictions on the test data
y_pred <- predict(model_ridge2, s = model_ridge$lambda.min, newx = X_test)

# Calculate the RMSE
RMSE <- sqrt(mean((y_pred - Boston_test$medv)^2))

results_table <- data.frame(Method = "Feature-selected model", RMSE = RMSE)
kable(results_table, align = "c", col.names = c("Method", "RMSE"))


# RMSE results
Method <- c("Ridge regression model", "Feature-selected model", "Linear regression model", "Random forest model")
RMSE <- c("5.228", "5.092", "4.802", "3.343")

df <- data.frame(Method, RMSE)
knitr::kable(df, align = "c")



