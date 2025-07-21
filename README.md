# 🌸 Iris Species Classification

This project performs a comprehensive **classification of iris flower species** using multiple **machine learning algorithms**, **feature engineering**, **dimensionality reduction (PCA)**, and **ensemble techniques**. The best-performing model is exported for production use.

## Iris Species

| Iris Species                      |
| --------------------------------- |
| ![Iris Species](iris.jpg)         |


## 📁 Project Structure

- `Iris_Species_Classification.ipynb`: Jupyter Notebook with all steps
- `best_model_Iris_Species_Classification_KNN_2025_07_20.pkl`: Saved best model pipeline


## 🛠️ Requirements

Install the necessary libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm
```


## 📂 Dataset

We use the [Iris dataset](https://scikit-learn.org/1.4/auto_examples/datasets/plot_iris_dataset.html).

The Iris dataset contains:
- 150 flower samples
- 4 features:
    - Sepal length
    - Sepal width
    - Petal length
    - Petal width
- 3 target classes: `setosa`, `versicolor`, `virginica`

The dataset is automatically downloaded using `scikit-learn`:
```bash
from sklearn import datasets
iris = datasets.load_iris()
```


## 🧠 Model Used

A wide variety of regression models are trained and evaluated, including:
- **Linear Models**: LogisticRegression
- **Tree-Based Models**: DecisionTree, RandomForest, ExtraTrees, GradientBoosting
- **Boosting Models**: XGBoost, LightGBM, AdaBoost
- **Distance-Based Models**: KNeighborsClassifier
- **Kernel-Based Models**: SVC (Support Vector Classifier with RBF kernel)
- **Probabilistic Model**: GaussianNB (Naive Bayes)
- **Ensemble Strategy**: StackingClassifier, VotingClassifier


## 📊 Model Performance Metrics

Each model is evaluated using:
- **Accuracy**: Proportion of correct predictions over the total number of samples.
- **ROC AUC Score**: Measures the model's ability to distinguish between classes; supports multi-class (One-vs-Rest).
- **Confusion Matrix**: Visualizes true vs. predicted classifications across all classes.
- **Cross-Validation Accuracy**: Average accuracy across folds to assess generalization performance.


## 📘 Language

- Python