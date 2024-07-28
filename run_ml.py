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
from keras.layers import Dense

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

    # Add additional parameters for other algorithms here as needed
    # For example, Logistic Regression regularization strength and penalty
    parser.add_argument('--lr_C', type=float, default=1.0, help='Inverse of regularization strength for Logistic Regression')
    parser.add_argument('--lr_penalty', type=str, default='l2', choices=['l1', 'l2', 'elasticnet', 'none'], help='Penalty for Logistic Regression')

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
        model.add(Dense(args.nn_neurons, activation='relu', input_dim=X_train.shape[1]))
        for _ in range(1, args.nn_layers):
            model.add(Dense(args.nn_neurons, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=args.nn_epochs, batch_size=args.nn_batch_size, verbose=1)
        y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    elif args.algorithm == 'decision_tree':
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif args.algorithm == 'svm':
        model = SVC()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif args.algorithm == 'knn':
        model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif args.algorithm == 'gbm':
        model = GradientBoostingClassifier()
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
