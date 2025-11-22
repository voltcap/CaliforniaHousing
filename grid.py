import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

housing = fetch_california_housing()
labels = housing.data
prices = housing.target

labelsTrain, labelsTest, pricesTrain, pricesTest = train_test_split(
    labels, prices, test_size=0.2, random_state=42
)

scaler = StandardScaler()
trainScaled = scaler.fit_transform(labelsTrain)
testScaled = scaler.transform(labelsTest)

print("\n")
print("Hyperparameter Optimisation")

print("\n")
print("ridge regression grid search")

alphaRidge = np.logspace(-2, 2, 20)
paramGridRidge = {'alpha': alphaRidge}

gridSearchRidge = GridSearchCV(Ridge(random_state=42), paramGridRidge, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
gridSearchRidge.fit(trainScaled, pricesTrain)

print(f"Best alpha: {gridSearchRidge.best_params_['alpha']:.4f}")
print(f"Best CV mean squared error: {-gridSearchRidge.best_score_:.4f}")
print(f"Best CV R2: {gridSearchRidge.best_estimator_.score(trainScaled, pricesTrain):.4f}")

ridgeBestModel = gridSearchRidge.best_estimator_
ridgeTestPreds = ridgeBestModel.predict(testScaled)
ridgeTestMSE = mean_squared_error(pricesTest, ridgeTestPreds)
ridgeTestR2 = r2_score(pricesTest, ridgeTestPreds)

print(f"Test MSE: {ridgeTestMSE:.4f}")
print(f"Test R2: {ridgeTestR2:.4f}")

print("\n")
print("Lasso regression grid search")

alphaLasso = np.logspace(-3, 1, 20)
paramGridLasso = {'alpha': alphaLasso}

gridSearchLasso = GridSearchCV(Lasso(random_state=42, max_iter=10000), paramGridLasso, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
gridSearchLasso.fit(trainScaled, pricesTrain)

print(f"Best alpha: {gridSearchLasso.best_params_['alpha']:.4f}")
print(f"Best CV mean squared error: {-gridSearchLasso.best_score_:.4f}")
print(f"Best CV R2: {gridSearchLasso.best_estimator_.score(trainScaled, pricesTrain):.4f}")

lassoBestModel = gridSearchLasso.best_estimator_
lassoTestPreds = lassoBestModel.predict(testScaled)
lassoTestMSE = mean_squared_error(pricesTest, lassoTestPreds)
lassoTestR2 = r2_score(pricesTest, lassoTestPreds)
lassoZeroCoeffs = np.sum(lassoBestModel.coef_ == 0)

print(f"Test MSE: {lassoTestMSE:.4f}")
print(f"Test R2: {lassoTestR2:.4f}")
print(f"Zero coefficients: {lassoZeroCoeffs}")

print("\n")
print("ElasticNet grid search")

alphaElasticnet = np.logspace(-2, 1, 15)
l1RatioElasticnet = [0.1, 0.3, 0.5, 0.7, 0.9]

paramGridElasticnet = {
    'alpha': alphaElasticnet,
    'l1_ratio': l1RatioElasticnet
}

gridSearchElasticnet = GridSearchCV(ElasticNet(random_state=42, max_iter=10000), paramGridElasticnet, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
gridSearchElasticnet.fit(trainScaled, pricesTrain)

print(f"Best alpha: {gridSearchElasticnet.best_params_['alpha']:.4f}")
print(f"Best l1_ratio: {gridSearchElasticnet.best_params_['l1_ratio']:.4f}")
print(f"Best CV mean squared error: {-gridSearchElasticnet.best_score_:.4f}")
print(f"Best CV R2: {gridSearchElasticnet.best_estimator_.score(trainScaled, pricesTrain):.4f}")

elasticnetBestModel = gridSearchElasticnet.best_estimator_
elasticnetTestPreds = elasticnetBestModel.predict(testScaled)
elasticnetTestMSE = mean_squared_error(pricesTest, elasticnetTestPreds)
elasticnetTestR2 = r2_score(pricesTest, elasticnetTestPreds)
elasticnetZeroCoeffs = np.sum(elasticnetBestModel.coef_ == 0)

print(f"Test MSE: {elasticnetTestMSE:.4f}")
print(f"Test R2: {elasticnetTestR2:.4f}")
print(f"Zero coefficients: {elasticnetZeroCoeffs}")

print("\n")
print("results")
print(f"\nBest ridge mse = {-gridSearchRidge.best_score_:.4f}, Test MSE = {ridgeTestMSE:.4f}, Test R2 = {ridgeTestR2:.4f}")
print(f"Best lasso mse = {-gridSearchLasso.best_score_:.4f}, Test MSE = {lassoTestMSE:.4f}, Test R2 = {lassoTestR2:.4f}")
print(f"Best ElasticNet mse = {-gridSearchElasticnet.best_score_:.4f}, Test MSE = {elasticnetTestMSE:.4f}, Test R2 = {elasticnetTestR2:.4f}")

print("\n")
print("results")
print("./"*20)
print(f"GridSearchCV found the best hyperparameters for each model:")
print(f"Ridge regularisation strength = {gridSearchRidge.best_params_['alpha']:.4f}")
print(f"Lasso regularisation strength = {gridSearchLasso.best_params_['alpha']:.4f}")
print(f"ElasticNet balance = {gridSearchElasticnet.best_params_['alpha']:.4f}, l1_ratio = {gridSearchElasticnet.best_params_['l1_ratio']:.4f}")
print(f"Cross-validation helps models generalise well")