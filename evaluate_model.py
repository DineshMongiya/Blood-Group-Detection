from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load test data
# Replace X_test and y_test with your actual test data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Load the model
from tensorflow.keras.models import load_model
model = load_model('./blood_detection/ml/model.keras')

# Predict test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(label_mapping.keys()), yticklabels=list(label_mapping.keys()))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification report
print(classification_report(y_test, y_pred_classes, target_names=list(label_mapping.keys())))
