from collections import Counter

import numpy as np
import pandas as pd

# set random seed
np.random.seed(0)

"""
Tips for debugging:
- Use `print` to check the shape of your data. Shape mismatch is a common error.
- Use `ipdb` to debug your code
    - `ipdb.set_trace()` to set breakpoints and check the values of your variables in interactive mode
    - `python -m ipdb -c continue hw3.py` to run the entire script in debug mode. Once the script is paused, you can use `n` to step through the code line by line.
"""


# 1. Load datasets
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    DO NOT MODIFY THIS FUNCTION.
    """
    # Load iris dataset
    iris = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
    )
    iris.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class",
    ]

    # Load Boston housing dataset
    boston = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    )

    return iris, boston


# 2. Preprocessing functions
def train_test_split(
    df: pd.DataFrame, target: str, test_size: float = 0.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Shuffle and split dataset into train and test sets
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    # Split target and features
    X_train = train.drop(target, axis=1).values
    y_train = train[target].values
    X_test = test.drop(target, axis=1).values
    y_test = test[target].values

    return X_train, X_test, y_train, y_test


def normalize(X: np.ndarray) -> np.ndarray:
    # Normalize features to [0, 1]
    # You can try other normalization methods, e.g., z-score, etc.
    # TODO: 1%

    # norm_X = (X - np.mean(X)) / np.std(X)
    norm_X = (X - np.min(X)) / (np.max(X) - np.min(X))
    return norm_X

encode_table = dict()
def encode_labels(y: np.ndarray) -> np.ndarray:
    """
    Encode labels to integers.
    """
    # TODO: 1%
    codes = np.zeros([len(y)])
    for i in range(len(y)):
        if y[i] not in encode_table:
            encode_table[y[i]] = len(encode_table)
        codes[i] = encode_table[y[i]]
    return codes


# 3. Models
class LinearModel:
    def __init__(
        self, learning_rate=0.5, iterations=2000, model_type="linear"
    ) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        # You can try different learning rate and iterations
        self.model_type = model_type

        assert model_type in [
            "linear",
            "logistic",
        ], "model_type must be either 'linear' or 'logistic'"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # ori_X = X
        X = np.insert(X, 0, 1, axis=1)
        # TODO: 2%
        n_classes = len(np.unique(y))
        n_features = X.shape[1]
        self.n_classes = n_classes
        self.n_features = n_features
        if self.model_type == "logistic":
            self.weights = np.zeros([self.n_classes, self.n_features])
            for i in range(self.iterations):
                self.weights -= self.learning_rate * self._compute_gradients(X, y)
        else:
            #closed form solutions (had introduced in class)
            self.w_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)), y)
            
            #optimazation with gradients descent
            self.weights = np.zeros([self.n_features])
            for i in range(self.iterations):
                self.weights -= self.learning_rate * self._compute_gradients(X, y)
                # print(mean_squared_error(self.predict(ori_X), y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.insert(X, 0, 1, axis=1)
        if self.model_type == "linear":
            # TODO: 2%
            h = np.zeros([X.shape[0]])
            for i in range(X.shape[0]):
                #gradients descent solution
                # h[i] = sum(np.multiply(self.weights, X[i]))
                #colsed form solution
                h[i] = sum(np.multiply(self.w_hat, X[i]))
            return h
        elif self.model_type == "logistic":
            # TODO: 2%
            h = np.matmul(self.weights, np.transpose(X))
            return np.argmax(h, axis=0)

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.model_type == "linear":
            # TODO: 3%
            #The gradients are referenced from the book titled "Learning from data: a short course"
            gradients = np.matmul(np.transpose(X), (np.subtract(np.matmul(X, self.weights), y))) / X.shape[0] * 2
            return gradients
        elif self.model_type == "logistic":
            # TODO: 3%
            #referenced from https://inst.eecs.berkeley.edu/~cs188/fa22/assets/notes/cs188-fa22-note21.pdf mutil-class logisitic regression
            #and https://medium.com/mlearning-ai/multiclass-logistic-regression-with-python-2ee861d5772a
            gradients = np.zeros([self.n_classes, self.n_features])
            for j in range(self.n_classes):
                for i in range(X.shape[0]):
                    weighted_sum = np.matmul(self.weights, np.transpose(X[i]))
                    soft_max = self._softmax(weighted_sum.reshape(1, len(weighted_sum)))
                    if y[i] == j:
                        gradients[j] -= ((1 - soft_max[0][j]) * X[i]) / X.shape[0]
                    else:
                        gradients[j] -= ((0 - soft_max[0][j]) * X[i]) / X.shape[0]
            return gradients

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)


class DecisionTree:
    def __init__(self, max_depth: int = 5, model_type: str = "classifier"):
        self.max_depth = max_depth
        self.model_type = model_type

        assert model_type in [
            "classifier",
            "regressor",
        ], "model_type must be either 'classifier' or 'regressor'"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.tree = self._build_tree(X, y, 0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        if depth >= self.max_depth or self._is_pure(y):
            return self._create_leaf(y)

        feature, threshold = self._find_best_split(X, y)
        # TODO: 4%
        node = dict()
        node['feature'], node['threshold'] = feature, threshold
        left_y, right_y, left_X, right_X = list(), list(), list(), list()
        for i in range(len(X)):
            if X[i][feature] <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
        node['left'] = self._build_tree(np.array(left_X), np.array(left_y), depth+1)
        node['right'] = self._build_tree(np.array(right_X), np.array(right_y), depth+1)
        return node
        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_child,
            "right": right_child,
        }

    def _is_pure(self, y: np.ndarray) -> bool:
        return len(set(y)) == 1

    def _create_leaf(self, y: np.ndarray):
        if self.model_type == "classifier":
            # TODO: 1%
            vals, count = np.unique(y, return_counts=True)
            return vals[np.argmax(count)]
        else:
            # TODO: 1%
            return np.sum(y) / len(y)

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        best_gini = float("inf")
        best_mse = float("inf")
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            sorted_indices = np.argsort(X[:, feature])
            for i in range(1, len(X)):
                if X[sorted_indices[i - 1], feature] != X[sorted_indices[i], feature]:
                    threshold = (
                        X[sorted_indices[i - 1], feature]
                        + X[sorted_indices[i], feature]
                    ) / 2
                    mask = X[:, feature] <= threshold
                    left_y, right_y = y[mask], y[~mask]

                    if self.model_type == "classifier":
                        gini = self._gini_index(left_y, right_y)
                        if gini < best_gini:
                            best_gini = gini
                            best_feature = feature
                            best_threshold = threshold
                    else:
                        mse = self._mse(left_y, right_y)
                        if mse < best_mse:
                            best_mse = mse
                            best_feature = feature
                            best_threshold = threshold

        return best_feature, best_threshold

    def _gini_index(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        # TODO: 4%
        #the definition of gini index isn't mentioned in class, so I referenced the following url https://ithelp.ithome.com.tw/articles/10276079
        len_left, len_right = len(left_y), len(right_y)
        val_left, count_left = np.unique(left_y, return_counts=True)
        val_right, count_right = np.unique(right_y, return_counts=True)
        gini_left, gini_right = 1, 1
        for c in count_left:
            gini_left -= (c / len_left) ** 2
        for c in count_right:
            gini_right -= (c / len_right) ** 2
        total = len_left + len_right
        return gini_left * (len_left) / total + gini_right * (len_right) / total


    def _mse(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        # TODO: 4%
        #referenced from https://jamleecute.web.app/regression-tree-%E8%BF%B4%E6%AD%B8%E6%A8%B9-bagging-bootstrap-aggrgation-r%E8%AA%9E%E8%A8%80/
        pred_left, pred_right = np.sum(left_y) / len(left_y), np.sum(right_y) / len(right_y)
        mse_left, mse_right = 0, 0
        for val in left_y:
            mse_left += (val - pred_left) ** 2
        for val in right_y:
            mse_right += (val - pred_right) ** 2
        return mse_left + mse_right

    def _traverse_tree(self, x: np.ndarray, node: dict):
        if isinstance(node, dict):
            feature, threshold = node["feature"], node["threshold"]
            if x[feature] <= threshold:
                return self._traverse_tree(x, node["left"])
            else:
                return self._traverse_tree(x, node["right"])
        else:
            return node


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        model_type: str = "classifier",
    ):
        # TODO: 1%
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model_type = model_type
        if model_type == "classifier":
            self.trees = [DecisionTree(max_depth=max_depth, model_type="classifier") for _ in range(n_estimators)]
        elif model_type == "regressor":
            self.trees = [DecisionTree(max_depth=max_depth, model_type="regressor") for _ in range(n_estimators)]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.model_type == "classifier":
            self.n_classes = len(set(y))
        total_data_num = X.shape[0]
        for tree in self.trees:
            # TODO: 2%
            bootstrap_indices = np.random.choice(np.arange(total_data_num), size=total_data_num*2//3, replace=False)
            sub_X = np.array([X[i] for i in range(total_data_num) if i in bootstrap_indices])
            sub_y = np.array([y[i] for i in range(total_data_num) if i in bootstrap_indices])
            tree.fit(sub_X, sub_y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: 2%
        ensemble_y = np.zeros([X.shape[0]])
        if self.model_type == "classifier":
            pred_y = np.zeros([X.shape[0], self.n_classes])
            for tree in self.trees:
                tree_pred = tree.predict(X)
                for i in range(X.shape[0]):
                    pred_y[i][int(tree_pred[i])] += 1
            ensemble_y = np.argmax(pred_y, axis=1)
        elif self.model_type == "regressor":
            pred_y = np.zeros([X.shape[0]])
            for tree in self.trees:
                pred_y += tree.predict(X)
            ensemble_y = pred_y / self.n_estimators
        return ensemble_y
    
# 4. Evaluation metrics
def accuracy(y_true, y_pred):
    # TODO: 1%
    count = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            count += 1
    return count / len(y_true)


def mean_squared_error(y_true, y_pred):
    # TODO: 1%
    lens, err = len(y_true), 0
    for i in range(lens):
        err += (y_true[i] - y_pred[i]) ** 2
    return err / lens


# 5. Main function
def main():
    iris, boston = load_data()

    # Iris dataset - Classification
    X_train, X_test, y_train, y_test = train_test_split(iris, "class")
    X_train, X_test = normalize(X_train), normalize(X_test)
    y_train, y_test = encode_labels(y_train), encode_labels(y_test)

    logistic_regression = LinearModel(model_type="logistic")
    logistic_regression.fit(X_train, y_train)
    y_pred = logistic_regression.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy(y_test, y_pred))

    decision_tree_classifier = DecisionTree(model_type="classifier")
    decision_tree_classifier.fit(X_train, y_train)
    y_pred = decision_tree_classifier.predict(X_test)
    print("Decision Tree Classifier Accuracy:", accuracy(y_test, y_pred))

    random_forest_classifier = RandomForest(model_type="classifier")
    random_forest_classifier.fit(X_train, y_train)
    y_pred = random_forest_classifier.predict(X_test)
    print("Random Forest Classifier Accuracy:", accuracy(y_test, y_pred))

    # Boston dataset - Regression
    X_train, X_test, y_train, y_test = train_test_split(boston, "medv")
    X_train, X_test = normalize(X_train), normalize(X_test)

    linear_regression = LinearModel(model_type="linear")
    linear_regression.fit(X_train, y_train)
    y_pred = linear_regression.predict(X_test)
    print("Linear Regression MSE:", mean_squared_error(y_test, y_pred))

    decision_tree_regressor = DecisionTree(model_type="regressor")
    decision_tree_regressor.fit(X_train, y_train)
    y_pred = decision_tree_regressor.predict(X_test)
    print("Decision Tree Regressor MSE:", mean_squared_error(y_test, y_pred))

    random_forest_regressor = RandomForest(model_type="regressor")
    random_forest_regressor.fit(X_train, y_train)
    y_pred = random_forest_regressor.predict(X_test)
    print("Random Forest Regressor MSE:", mean_squared_error(y_test, y_pred))


if __name__ == "__main__":
    main()
