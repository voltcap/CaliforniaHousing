from sklearn.linear_model import Ridge
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

alphaValues = [0.01, 0.1, 1, 10, 100]

resultsMetrics = {
    'alpha': [], 'MAE': [], 'MSE': [], 'RMSE': [], 'R2': [], 'trainR2': [], 'testR2': []
}

print("\n")
print("Ridge regression results")

for alpha in alphaValues:
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(trainScaled, pricesTrain)
    
    testPreds = model.predict(testScaled)
    trainPreds = model.predict(trainScaled)
    
    mae = mean_absolute_error(pricesTest, testPreds)
    mse = mean_squared_error(pricesTest, testPreds)
    rmse = np.sqrt(mse)
    r2Test = r2_score(pricesTest, testPreds)
    r2Train = r2_score(pricesTrain, trainPreds)
    
    resultsMetrics['alpha'].append(alpha)
    resultsMetrics['MAE'].append(mae)
    resultsMetrics['MSE'].append(mse)
    resultsMetrics['RMSE'].append(rmse)
    resultsMetrics['R2'].append(r2Test)
    resultsMetrics['trainR2'].append(r2Train)
    
    print(f"\nAlpha = {alpha}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Test R2:  {r2Test:.4f}")
    print(f"  Train R2: {r2Train:.4f}")
    print(f"  Bias variance is {r2Train - r2Test:.4f}")

print("\n")

fig, axes = plt.subplots(1, 2, figsize=(12, 3))
fig.suptitle('results', fontsize=8, fontweight='bold')

axes[0].plot(resultsMetrics['alpha'], resultsMetrics['MSE'], marker='o', linewidth=2, markersize=8, color='turquoise')
axes[0].set_xlabel('regularisation strength')
axes[0].set_ylabel('Test MSE')
axes[0].set_title('Test MSE against alpha')
axes[0].set_xscale('log')
axes[0].grid(True, alpha=0.3)

axes[1].plot(resultsMetrics['alpha'], resultsMetrics['trainR2'], marker='s', linewidth=2, markersize=8, label='Train R2', color='green')
axes[1].plot(resultsMetrics['alpha'], resultsMetrics['R2'], marker='o', linewidth=2, markersize=8, label='Test R2', color='turquoise')
axes[1].set_xlabel('regularisation strength')
axes[1].set_ylabel('R2')
axes[1].set_title('Bias variance')
axes[1].set_xscale('log')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n")
print("results")
print("./"*20)
print(f" if there's low alpha such as 0.01 to 0.1, there's low bias and high variance")
print(f"if the alpha value is 1, the regularisation is at an ideal range")
print(f"if the alpha's quite high, around 10 to 100, there's low variance and high bias")