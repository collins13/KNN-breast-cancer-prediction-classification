from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

breast_cancer_data = load_breast_cancer()

training_data, validation_data, training_labels,validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size=0.2,random_state=200)
accuracies = []
for k in range(1,201):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data,training_labels)
  accuracy = classifier.score(validation_data, validation_labels)
  accuracies.append(accuracy)
  print("k =", k, "accuracy =", accuracy)
  k_list = list(range(1,201))
plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")

plt.show()