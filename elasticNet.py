import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)

housing = fetch_california_housing()
labels = housing.data
prices = housing.target
feature_names = housing.feature_names

print("Initial dataset", labels.shape)
print(f"samples: {labels.shape[0]}, labels: {labels.shape[1]}\n")

labelsTrain, labelsTest, pricesTrain, pricesTest = train_test_split(
    labels, prices, test_size=0.2, random_state=42
)

print("dataset shapes:")
print(f"Training shape: {labelsTrain.shape}")
print(f"Test shape: {labelsTest.shape}")
print(f"Training prices: {pricesTrain.shape}")
print(f"Test prices: {pricesTest.shape}")

scaler = StandardScaler()
trainScaled = scaler.fit_transform(labelsTrain)
testScaled = scaler.transform(labelsTest)

print("\n")
print("verification for scaling")
print("\nScaled training statistics:")
print(f"Mean: {trainScaled.mean(axis=0)}")
print(f"Standard deviation: {trainScaled.std(axis=0)}")

print("\nScaled test statistics:")
print(f"Mean: {testScaled.mean(axis=0)}")
print(f"Standard deviation: {testScaled.std(axis=0)}")

alphaValues = [0.01, 0.1, 1]
l1RatioValues = [0.1, 0.5, 0.9]

elasticnetMetrics = {
    'alpha': [], 'l1_ratio': [], 'MAE': [], 'MSE': [], 'RMSE': [], 'R2': [], 'trainR2': [], 'zeroCoeffs': []
}

print("\n")
print("ElasticNet regression results")

for alpha in alphaValues:
    print(f"\n Alpha is {alpha} ")
    for l1Ratio in l1RatioValues:
        model = ElasticNet(alpha=alpha, l1_ratio=l1Ratio, random_state=42, max_iter=10000)
        model.fit(trainScaled, pricesTrain)
        
        testPreds = model.predict(testScaled)
        trainPreds = model.predict(trainScaled)
        
        mae = mean_absolute_error(pricesTest, testPreds)
        mse = mean_squared_error(pricesTest, testPreds)
        rmse = np.sqrt(mse)
        r2Test = r2_score(pricesTest, testPreds)
        r2Train = r2_score(pricesTrain, trainPreds)
        
        zeroCoeffs = np.sum(model.coef_ == 0)
        
        elasticnetMetrics['alpha'].append(alpha)
        elasticnetMetrics['l1_ratio'].append(l1Ratio)
        elasticnetMetrics['MAE'].append(mae)
        elasticnetMetrics['MSE'].append(mse)
        elasticnetMetrics['RMSE'].append(rmse)
        elasticnetMetrics['R2'].append(r2Test)
        elasticnetMetrics['trainR2'].append(r2Train)
        elasticnetMetrics['zeroCoeffs'].append(zeroCoeffs)
        
        print(f"\n  L1 Ratio = {l1Ratio}")
        print(f"    MSE:  {mse:.4f}")
        print(f"    MAE:  {mae:.4f}")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    Test R2:  {r2Test:.4f}")
        print(f"    Train R2: {r2Train:.4f}")
        print(f"    Bias variance is {r2Train - r2Test:.4f}")
        print(f"    Zero coefficients: {zeroCoeffs} ({zeroCoeffs / len(model.coef_) * 100:.1f}%)")

print("\n")

fig, axes = plt.subplots(2, 3, figsize=(13, 6))
fig.suptitle('results', fontsize=8, fontweight='bold')

for idx, alpha in enumerate(alphaValues):
    alphaIndices = [i for i, a in enumerate(elasticnetMetrics['alpha']) if a == alpha]
    l1RatiosForAlpha = [elasticnetMetrics['l1_ratio'][i] for i in alphaIndices]
    mseForAlpha = [elasticnetMetrics['MSE'][i] for i in alphaIndices]
    r2ForAlpha = [elasticnetMetrics['R2'][i] for i in alphaIndices]
    trainR2ForAlpha = [elasticnetMetrics['trainR2'][i] for i in alphaIndices]
    zeroCoeffsForAlpha = [elasticnetMetrics['zeroCoeffs'][i] for i in alphaIndices]
    
    axes[0, idx].plot(l1RatiosForAlpha, mseForAlpha, marker='o', linewidth=2, markersize=8, color='turquoise')
    axes[0, idx].set_xlabel('L1 Ratio')
    axes[0, idx].set_ylabel('Test MSE')
    axes[0, idx].set_title(f'Test MSE against L1 Ratio {alpha})')
    axes[0, idx].grid(True, alpha=0.3)
    
    axes[1, idx].plot(l1RatiosForAlpha, trainR2ForAlpha, marker='s', linewidth=2, markersize=8, label='Train R2', color='pink')
    axes[1, idx].plot(l1RatiosForAlpha, r2ForAlpha, marker='o', linewidth=2, markersize=8, label='Test R2', color='turquoise')
    axes[1, idx].set_xlabel('L1 Ratio')
    axes[1, idx].set_ylabel('R2')
    axes[1, idx].set_title(f'Bias variance {alpha})')
    axes[1, idx].legend()
    axes[1, idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n")
print("results")
print("./"*20)
print(f"L1 Ratio 0.1 so L2 regularisation, keeps most features")
print(f"L1 Ratio 0.5, balanced L1 and L2, moderate feature selection")
print(f"L1 Ratio 0.9, l1 regularisation, aggressive feature elimination")
print(f"\nAs alpha increases, all models show higher MSE and lower R2")
print(f"As l1 ratio increases, sparsity increases")
print(f"ElasticNet combines ridge stability with lasso feature selection enhancements")