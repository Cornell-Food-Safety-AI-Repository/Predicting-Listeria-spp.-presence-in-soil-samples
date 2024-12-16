import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Input

def main():
   
    parser = argparse.ArgumentParser(description="Run ML algorithms with specified parameters.")
    parser.add_argument('--file_path', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--algorithm', type=str, required=True, choices=['logistic_regression', 'neural_network', 'decision_tree', 'svm', 'knn', 'gbm'], help='Algorithm to use')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size fraction')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for train_test_split')

    # Neural Network specific parameters
    parser.add_argument('--nn_epochs', type=int, default=100, help='Number of epochs for Neural Network')
    parser.add_argument('--nn_batch_size', type=int, default=10, help='Batch size for Neural Network')
    parser.add_argument('--nn_layers', type=int, default=2, help='Number of hidden layers for Neural Network')
    parser.add_argument('--nn_neurons', type=int, default=64, help='Number of neurons per hidden layer for Neural Network')

    # Logistic Regression specific parameters
    parser.add_argument('--lr_C', type=float, default=1.0, help='Inverse of regularization strength for Logistic Regression')
    parser.add_argument('--lr_penalty', type=str, default='l2', choices=['l1', 'l2', 'elasticnet', 'none'], help='Penalty for Logistic Regression')

    # Decision Tree specific parameters
    parser.add_argument('--dt_max_depth', type=int, default=None, help='The maximum depth of the tree')
    parser.add_argument('--dt_min_samples_split', type=int, default=2, help='The minimum number of samples required to split an internal node')

    # SVM specific parameters
    parser.add_argument('--svm_C', type=float, default=1.0, help='Regularization parameter for SVM')
    parser.add_argument('--svm_kernel', type=str, default='rbf', help='Specifies the kernel type to be used in the SVM algorithm')

    # KNN specific parameters
    parser.add_argument('--knn_n_neighbors', type=int, default=5, help='Number of neighbors to use for kneighbors queries')
    parser.add_argument('--knn_metric', type=str, default='minkowski', help='The distance metric to use for the KNN algorithm')

    # GBM specific parameters
    parser.add_argument('--gbm_learning_rate', type=float, default=0.1, help='Learning rate for GBM')
    parser.add_argument('--gbm_n_estimators', type=int, default=100, help='The number of boosting stages to be run')

    args = parser.parse_args()

   
    df = pd.read_csv(args.file_path)

    
    df.iloc[:, -1] = (df.iloc[:, -1] > 0).astype(int)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

   
    if args.algorithm == 'logistic_regression':
        model = LogisticRegression(C=args.lr_C, penalty=args.lr_penalty)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif args.algorithm == 'neural_network':
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1],)))
        for _ in range(args.nn_layers):
            model.add(Dense(args.nn_neurons, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=args.nn_epochs, batch_size=args.nn_batch_size, verbose=1)
        y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    elif args.algorithm == 'decision_tree':
        model = DecisionTreeClassifier(max_depth=args.dt_max_depth, min_samples_split=args.dt_min_samples_split)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif args.algorithm == 'svm':
        model = SVC(C=args.svm_C, kernel=args.svm_kernel)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif args.algorithm == 'knn':
        model = KNeighborsClassifier(n_neighbors=args.knn_n_neighbors, metric=args.knn_metric)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif args.algorithm == 'gbm':
        model = GradientBoostingClassifier(learning_rate=args.gbm_learning_rate, n_estimators=args.gbm_n_estimators)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")

    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{conf_matrix}")

if __name__ == "__main__":
    main()
