from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from configs.config import PLOTS_PATH


class DecisionTree():

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.dt = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42)

    # ****FUNCTION TO TRAIN MODEL USING TRAINING DATA****
    def train_model(self, X_train, y_train):

        dt = self.dt.fit(X_train, y_train)

        preds = dt.predict(X_train)

        accuracy = accuracy_score(y_train, preds)

        precision = precision_score(y_train, preds)

        recall = recall_score(y_train, preds)

        f1 = f1_score(y_train, preds)

        self.confusion_matrix(y_train, preds, 'train_confusion_matrix.png')

        self.precision_recall_plot(
            y_train, preds, 'train_precision_recall_curve.png')

        return accuracy, precision, recall, f1

    # ****FUNCTION TO EVALUATE MODEL USING TEST DATA****
    def evaluate_model(self, X_test, y_test):
        test_preds = self.dt.predict(X_test)

        test_accuracy = accuracy_score(y_test, test_preds)

        test_precision = precision_score(y_test, test_preds)

        test_recall = recall_score(y_test, test_preds)

        test_f1 = f1_score(y_test, test_preds)

        self.confusion_matrix(y_test, test_preds, 'eval_confusion_matrix.png')

        self.precision_recall_plot(
            y_test, test_preds, 'eval_precision_recall_curve.png')

        return test_accuracy, test_precision, test_recall, test_f1

    # ****FUNCTION TO MAKE GENERAL PREDICTIONS****
    def make_predictions(self, X):

        return self.dt.predict(X)

    # ****FUNCTION TO CREATE AND SAVE CONFUSION MATRIX****
    def confusion_matrix(self, actual, pred, filename):

        file_path = PLOTS_PATH / 'classifications/decision_trees' / filename

        ConfusionMatrixDisplay.from_predictions(
            actual, pred, display_labels=["No Playoffs", "Playoffs"])

        plt.savefig(file_path, bbox_inches="tight", dpi=300)
        plt.close()

    # ****FUNCTION TO CREATE AND SAVE PRECISION-RECALL CURVE****
    def precision_recall_plot(self, actual, pred, filename):

        file_path = PLOTS_PATH / 'classifications/decision_trees' / filename

        precisions, recalls, thresholds = precision_recall_curve(actual, pred)

        plt.figure()
        plt.plot(thresholds, precisions[:-1],
                 "b--", label="Precision", linewidth=2)
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
        plt.vlines(0.5, 0, 1.0, "k", "dotted", label="threshold")
        plt.xlabel("Threshold")
        plt.grid()
        plt.legend(loc="center right")

        plt.savefig(file_path, bbox_inches="tight", dpi=300)
        plt.close()

    # ****FUNCTION TO CREATE AND SAVE PLOT OF DECISION TREE****
    def plot(self):

        file_path = PLOTS_PATH / 'classifications/decision_trees/decision_tree.png'

        plt.figure()
        tree.plot_tree(self.dt, filled=True, feature_names=None,
                       class_names=["No Playoffs", "Playoffs"])

        plt.savefig(file_path, bbox_inches="tight", dpi=300)
        plt.close()
