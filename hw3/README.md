## Getting Started

1. First, make sure you have Python 3.10 installed on your machine.

2. Install the required packages:

```bash
pip install numpy pandas
```

3. Download the `hw3.py` file which contains the assignment code.

4. Open the `hw3.py` file in your favorite code editor.

5. Read the comments and instructions in the code carefully.

6. Complete the missing code sections marked with `raise NotImplementedError`.

7. Save your changes and run the `hw3.py` script:

```bash
python hw3.py
```

8. Check your results and make sure they are correct.

## Dataset

The assignment involves two datasets:

1. Iris dataset: A famous dataset for classification tasks. It contains 150 samples of iris flowers with four features (sepal length, sepal width, petal length, and petal width) and three class labels (Setosa, Versicolor, and Virginica).

2. Boston Housing dataset: A dataset for regression tasks. It contains 506 samples of houses in Boston with 13 features (e.g., crime rate, average number of rooms, etc.) and a continuous target variable representing the median value of owner-occupied homes in $1000's.

## Algorithms

You will be implementing the following machine learning algorithms:

1. Linear Regression: A linear model for regression tasks.
2. Logistic Regression: A linear model for classification tasks.
3. Decision Tree: A tree-based model for classification or regression tasks.
4. Random Forest: An ensemble of decision trees for classification or regression tasks.

## Preprocessing Functions

You need to complete the following preprocessing functions:

1. `normalize(X: np.ndarray) -> np.ndarray`: Normalize features.
2. `encode_labels(y: np.ndarray) -> np.ndarray`: Encode labels to integers.

## Models

You need to complete the following model classes:

1. `LinearModel`: Implement the `fit`, `predict`, and `_compute_gradients` methods.
2. `DecisionTree`: Implement the `_build_tree`, `_find_best_split`, `_gini_index`, `_mse`, and `_traverse_tree` methods.
3. `RandomForest`: Implement the `__init__`, `fit`, and `predict` methods.

## Evaluation Metrics

You need to complete the following evaluation metric functions:

1. `accuracy(y_true, y_pred)`: Calculate the accuracy of the predictions.
2. `mean_squared_error(y_true, y_pred)`: Calculate the mean squared error of the predictions.

## Main Function

The `main()` function is provided for you to test your implementation. It loads the datasets, preprocesses the data, and trains and evaluates the models on both the Iris and Boston Housing datasets. After implementing the missing code sections, you should be able to run the `main()` function and see the results printed on the console.

## Tips for Debugging

- Use `print` to check the shape of your data. Shape mismatch is a common error.
- Use `ipdb` to debug your code:
  - `ipdb.set_trace()` to set breakpoints and check the values of your variables in interactive mode.
  - `python -m ipdb -c continue hw3.py` to run the entire script in debug mode. Once the script is paused, you can use `n` to step through the code line by line.

Good luck and happy coding!