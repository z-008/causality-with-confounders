import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask import session
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    make_scorer,
)
from causal.preprocessing import generic_preprocessing as gprep


classification_Reports = []
confusion_Matrix = []
accuracies = []
target_Names = []


def classification_report_with_accuracy_score(y_true, y_pred):
    report = classification_report(
        y_true, y_pred, target_names=target_Names[0], output_dict=True
    )
    temp = pd.DataFrame(report).transpose()
    classification_Reports.append(
        [
            temp.to_html(
                classes=[
                    "table",
                    "table-bordered",
                    "table-striped",
                    "table-hover",
                    "thead-light",
                ]
            ).strip("\n")
        ]
    )
    matrix = confusion_matrix(y_true, y_pred)
    temp = pd.DataFrame(matrix).transpose()
    confusion_Matrix.append(
        [
            temp.to_html(
                classes=[
                    "table",
                    "table-bordered",
                    "table-striped",
                    "table-hover",
                    "thead-light",
                ]
            )
        ]
    )
    accuracies.append(round(100 * accuracy_score(y_true, y_pred), 2))
    return accuracy_score(y_true, y_pred)


def KNearestNeighbours(value, choice, scale_val, encode_val):
    classification_Reports.clear()
    confusion_Matrix.clear()
    accuracies.clear()
    target_Names.clear()

    df = gprep.read_dataset("causal/clean/clean.csv")

    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    target_Names.append(list(df.iloc[:, -1].unique()))
    le = LabelEncoder()
    sc = StandardScaler()

    # print(confusion_matrix(y_test,pred))
    # print(classification_report(y_test,pred))
    # print('Error Rate: {}'.format(np.mean(pred!=y_test))

    if choice == 1:
        size = value / 100
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=size, random_state=40
        )

        if scale_val == 1:
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

        if encode_val == 1:
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

        knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
        pred_vals = knn.predict(X_test)

        acc = accuracy_score(y_test, pred_vals)
        report = classification_report(
            y_test,
            pred_vals,
            target_names=list(df.iloc[:, -1].unique()),
            output_dict=True,
        )
        data = pd.DataFrame(report).transpose()

        matrix = confusion_matrix(y_test, pred_vals)
        matrix_data = pd.DataFrame(matrix).transpose()

        return [round(acc * 100, 2), data, matrix_data]

    elif choice == 2:
        k = value
        kfold = KFold(n_splits=k, random_state=7, shuffle=True)
        model = KNeighborsClassifier(n_neighbors=1)

        if scale_val == 1:
            X = sc.fit_transform(X)

        if encode_val == 1:
            y = le.fit_transform(y)

        predicted = cross_val_score(
            model,
            X,
            y,
            cv=kfold,
            scoring=make_scorer(classification_report_with_accuracy_score),
        )

        return [accuracies, classification_Reports, confusion_Matrix]

    elif choice == 0:

        if scale_val == 1:
            X = sc.fit_transform(X)

        if encode_val == 1:
            y = le.fit_transform(y)

        knn = KNeighborsClassifier(n_neighbors=1).fit(X, y)

        if session["ext"] == "csv":
            df = pd.read_csv("uploads/test.csv")
        elif session["ext"] == "json":
            df = pd.read_json("uploads/test.json")

        X_test = df.iloc[:, 1:-1]
        y = df.iloc[:, -1]

        if scale_val == 1:
            X_test = sc.transform(X_test)
        if encode_val == 1:
            y_test = le.transform(y_test)

        pred_vals = knn.predict(X_test)
        acc = accuracy_score(y_test, pred_vals)
        report = classification_report(
            y_test,
            pred_vals,
            target_names=list(df.iloc[:, -1].unique()),
            output_dict=True,
        )
        data = pd.DataFrame(report).transpose()

        matrix = confusion_matrix(y_test, pred_vals)
        matrix_data = pd.DataFrame(matrix).transpose()

        return [round(acc * 100, 2), data, matrix_data]
