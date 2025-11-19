from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

svm_clf = SVC(kernel='linear', C=1.0) # soft margin C (0.1 to 100), hard margin C (large enough)
svm_clf.fit(X_train, y_train)

svm_accuracy = svm_clf.score(X_test, y_test)
print(f'SVM Accuracy: {svm_accuracy:.3f}')

svm_pred = svm_clf.predict([X_test[0]])

print(f"\nTrue Value: {y_test[0]}") # 1 - Benign (non-cancerous), 0 - Malignant (cancerous)
print(f"SVM Predicted: {'Benign' if svm_pred[0] == 1 else 'Malignant'}")
