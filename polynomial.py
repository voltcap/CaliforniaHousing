import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

housing = fetch_california_housing()
labels = housing.data
prices = housing.target
feature_names = housing.feature_names

print("Initial dataset", labels.shape)
print(f"samples: {labels.shape[0]}, labels: {labels.shape[1]}\n")

medIncIdx = 0
houseAgeIdx = 1

labelsSubset = labels[:, [medIncIdx, houseAgeIdx]]

labelsTrain, labelsTest, pricesTrain, pricesTest = train_test_split(
    labelsSubset, prices, test_size=0.2, random_state=42
)

print("dataset shapes:")
print(f"Training shape: {labelsTrain.shape}")
print(f"Test shape: {labelsTest.shape}")
print(f"Training prices: {pricesTrain.shape}")
print(f"Test prices: {pricesTest.shape}")

scaler = StandardScaler()
trainScaled = scaler.fit_transform(labelsTrain)
testScaled = scaler.transform(labelsTest)

polynomialDegrees = [2, 3, 5]

ridgeResults = {
    'degree': [], 'trainMSE': [], 'testMSE': [], 'trainR2': [], 'testR2': [], 'numFeatures': []
}

lassoResults = {
    'degree': [], 'trainMSE': [], 'testMSE': [], 'trainR2': [], 'testR2': [], 'numFeatures': []
}

elasticnetResults = {
    'degree': [], 'trainMSE': [], 'testMSE': [], 'trainR2': [], 'testR2': [], 'numFeatures': []
}

print("\n")
print("Polynomial feature expansion and regularisation")

for degree in polynomialDegrees:
    print(f"\n Polynomial degree = {degree} ")
    
    polyTransform = PolynomialFeatures(degree=degree, include_bias=False)
    trainPoly = polyTransform.fit_transform(trainScaled)
    testPoly = polyTransform.transform(testScaled)
    
    numFeatures = trainPoly.shape[1]
    print(f"Number of polynomial features: {numFeatures}")
    
    ridgeModel = Ridge(alpha=0.1, random_state=42)
    ridgeModel.fit(trainPoly, pricesTrain)
    ridgeTrainPreds = ridgeModel.predict(trainPoly)
    ridgeTestPreds = ridgeModel.predict(testPoly)
    
    ridgeTrainMSE = mean_squared_error(pricesTrain, ridgeTrainPreds)
    ridgeTestMSE = mean_squared_error(pricesTest, ridgeTestPreds)
    ridgeTrainR2 = r2_score(pricesTrain, ridgeTrainPreds)
    ridgeTestR2 = r2_score(pricesTest, ridgeTestPreds)
    
    ridgeResults['degree'].append(degree)
    ridgeResults['trainMSE'].append(ridgeTrainMSE)
    ridgeResults['testMSE'].append(ridgeTestMSE)
    ridgeResults['trainR2'].append(ridgeTrainR2)
    ridgeResults['testR2'].append(ridgeTestR2)
    ridgeResults['numFeatures'].append(numFeatures)
    
    print(f"\nRidge (alpha=0.1)")
    print(f"  Train MSE: {ridgeTrainMSE:.4f}, Test MSE: {ridgeTestMSE:.4f}")
    print(f"  Train R2: {ridgeTrainR2:.4f}, Test R2: {ridgeTestR2:.4f}")
    print(f"  Bias variance: {ridgeTrainR2 - ridgeTestR2:.4f}")
    
    lassoModel = Lasso(alpha=0.001, random_state=42, max_iter=10000)
    lassoModel.fit(trainPoly, pricesTrain)
    lassoTrainPreds = lassoModel.predict(trainPoly)
    lassoTestPreds = lassoModel.predict(testPoly)
    
    lassoTrainMSE = mean_squared_error(pricesTrain, lassoTrainPreds)
    lassoTestMSE = mean_squared_error(pricesTest, lassoTestPreds)
    lassoTrainR2 = r2_score(pricesTrain, lassoTrainPreds)
    lassoTestR2 = r2_score(pricesTest, lassoTestPreds)
    lassoZeroCoeffs = np.sum(lassoModel.coef_ == 0)
    
    lassoResults['degree'].append(degree)
    lassoResults['trainMSE'].append(lassoTrainMSE)
    lassoResults['testMSE'].append(lassoTestMSE)
    lassoResults['trainR2'].append(lassoTrainR2)
    lassoResults['testR2'].append(lassoTestR2)
    lassoResults['numFeatures'].append(numFeatures)
    
    print(f"\nLasso (alpha=0.001)")
    print(f"  Train MSE: {lassoTrainMSE:.4f}, Test MSE: {lassoTestMSE:.4f}")
    print(f"  Train R2: {lassoTrainR2:.4f}, Test R2: {lassoTestR2:.4f}")
    print(f"  Bias variance: {lassoTrainR2 - lassoTestR2:.4f}")
    print(f"  Zero coefficients: {lassoZeroCoeffs} out of {numFeatures}")
    
    elasticnetModel = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000)
    elasticnetModel.fit(trainPoly, pricesTrain)
    elasticnetTrainPreds = elasticnetModel.predict(trainPoly)
    elasticnetTestPreds = elasticnetModel.predict(testPoly)
    
    elasticnetTrainMSE = mean_squared_error(pricesTrain, elasticnetTrainPreds)
    elasticnetTestMSE = mean_squared_error(pricesTest, elasticnetTestPreds)
    elasticnetTrainR2 = r2_score(pricesTrain, elasticnetTrainPreds)
    elasticnetTestR2 = r2_score(pricesTest, elasticnetTestPreds)
    elasticnetZeroCoeffs = np.sum(elasticnetModel.coef_ == 0)
    
    elasticnetResults['degree'].append(degree)
    elasticnetResults['trainMSE'].append(elasticnetTrainMSE)
    elasticnetResults['testMSE'].append(elasticnetTestMSE)
    elasticnetResults['trainR2'].append(elasticnetTrainR2)
    elasticnetResults['testR2'].append(elasticnetTestR2)
    elasticnetResults['numFeatures'].append(numFeatures)
    
    print(f"\nElasticNet (alpha=0.01, l1_ratio=0.5)")
    print(f"  Train MSE: {elasticnetTrainMSE:.4f}, Test MSE: {elasticnetTestMSE:.4f}")
    print(f"  Train R2: {elasticnetTrainR2:.4f}, Test R2: {elasticnetTestR2:.4f}")
    print(f"  Bias variance: {elasticnetTrainR2 - elasticnetTestR2:.4f}")
    print(f"  Zero coefficients: {elasticnetZeroCoeffs} out of {numFeatures}")

print("\n")

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle('results', fontsize=8, fontweight='bold')

axes[0, 0].plot(ridgeResults['degree'], ridgeResults['trainMSE'], marker='s', linewidth=2, markersize=8, label='Train MSE', color='olivedrab')
axes[0, 0].plot(ridgeResults['degree'], ridgeResults['testMSE'], marker='o', linewidth=2, markersize=8, label='Test MSE', color='royalblue')
axes[0, 0].set_xlabel('Polynomial degree')
axes[0, 0].set_ylabel('MSE')
axes[0, 0].set_title('Ridge: Train againt test mse')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks(polynomialDegrees)

axes[0, 1].plot(lassoResults['degree'], lassoResults['trainMSE'], marker='s', linewidth=2, markersize=8, label='Train MSE', color='hotpink')
axes[0, 1].plot(lassoResults['degree'], lassoResults['testMSE'], marker='o', linewidth=2, markersize=8, label='Test MSE', color='teal')
axes[0, 1].set_xlabel('Polynomial Degree')
axes[0, 1].set_ylabel('MSE')
axes[0, 1].set_title('Lasso: Train againt test mse')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xticks(polynomialDegrees)

axes[1, 0].plot(elasticnetResults['degree'], elasticnetResults['trainMSE'], marker='s', linewidth=2, markersize=8, label='Train MSE', color='saddlebrown')
axes[1, 0].plot(elasticnetResults['degree'], elasticnetResults['testMSE'], marker='o', linewidth=2, markersize=8, label='Test MSE', color='goldenrod')
axes[1, 0].set_xlabel('Polynomial Degree')
axes[1, 0].set_ylabel('MSE')
axes[1, 0].set_title('ElasticNet: Train againt test mse')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xticks(polynomialDegrees)

axes[1, 1].plot(ridgeResults['degree'], ridgeResults['testMSE'], marker='o', linewidth=2, markersize=8, label='Ridge', color='mediumorchid')
axes[1, 1].plot(lassoResults['degree'], lassoResults['testMSE'], marker='o', linewidth=2, markersize=8, label='Lasso', color='blueviolet')
axes[1, 1].plot(elasticnetResults['degree'], elasticnetResults['testMSE'], marker='o', linewidth=2, markersize=8, label='ElasticNet', color='mediumturquoise')
axes[1, 1].set_xlabel('Polynomial degree')
axes[1, 1].set_ylabel('Test MSE')
axes[1, 1].set_title('Model comparison: Test mse vs degree')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xticks(polynomialDegrees)

plt.tight_layout()
plt.show()
