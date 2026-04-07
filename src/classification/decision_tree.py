from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from configs.config import PLOTS_PATH


class DecisionTree():

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.dt = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42)

    def train_model(self, X_train, y_train):

        dt = self.dt.fit(X_train, y_train)

        preds = dt.predict(X_train)

        self.confusion_matrix(y_train, preds, 'train_confusion_matrix')

        accuracy = accuracy_score(y_train, preds)

        return accuracy

    def evaluate_model(self, X_test, y_test):
        test_preds = self.dt.predict(X_test)

        test_accuracy = accuracy_score(y_test, test_preds)

        self.confusion_matrix(y_test, test_preds, 'eval_confusion_matrix')

        return test_accuracy

    def make_predictions(self, X):

        return self.dt.predict(X)

    def confusion_matrix(self, actual, pred, filename):
        ConfusionMatrixDisplay.from_predictions(
            actual, pred, display_labels=["No Playoffs", "Playoffs"])

        file_path = PLOTS_PATH / 'classifications/decision-trees' / filename
        plt.savefig(file_path, bbox_inches="tight", dpi=300)
        # plt.show()

    def plot(self):
        plt.figure(figsize=(12, 8))
        tree.plot_tree(self.dt, filled=True, feature_names=None,
                       class_names=["No Playoffs", "Playoffs"])
        file_path = PLOTS_PATH / 'classifications/decision-trees/decision_tree.png'
        plt.savefig(file_path, bbox_inches="tight", dpi=300)
        # plt.show()
