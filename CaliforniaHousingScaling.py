import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

fig, axes = plt.subplots(2, len(feature_names), figsize=(20, 6)) 
fig.suptitle('results', fontsize=8, fontweight='bold')

for i, featureName in enumerate(feature_names):
    axBefore = axes[0, i]
    axAfter = axes[1, i]

    axBefore.hist(labelsTrain[:, i], bins=30, color='turquoise', edgecolor='purple', alpha=0.7)
    axBefore.set_title(f'{featureName}\nbefore the scaling', fontsize=13)
    axBefore.set_xlabel('Value')
    axBefore.set_ylabel('Frequency')

    axAfter.hist(trainScaled[:, i], bins=30, color='violet', edgecolor='black', alpha=0.7)
    axAfter.set_title(f'{featureName}\nafter scaling', fontsize=13)
    axAfter.set_xlabel('value')
    axAfter.set_ylabel('frequency')

plt.tight_layout()
plt.show()

print("\n")
print("final results")
print("./"*20)
print(f"dataset loaded: {labels.shape[0]} samples, {labels.shape[1]} labels")
print(f"dataset split: {labelsTrain.shape[0]} train, {labelsTest.shape[0]} test")