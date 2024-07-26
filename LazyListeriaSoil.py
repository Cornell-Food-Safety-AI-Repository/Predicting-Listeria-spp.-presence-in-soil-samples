import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lazypredict.Supervised import LazyClassifier, LazyRegressor

def preprocess_data(file_path):
    
    df = pd.read_csv(file_path)

    df.drop(df.columns[:4], axis=1, inplace=True)

 
    df.drop(df.columns[-3:], axis=1, inplace=True)

    df.replace('-', pd.NA, inplace=True)

    df.dropna(inplace=True)

    df.to_csv('ListeriaSoil_clean_updated.csv', index=False)

    return df

parser = argparse.ArgumentParser(description="Run ML algorithms with LazyPredict.")
parser.add_argument('--file_path', type=str, required=True, help='Path to the CSV file')
parser.add_argument('--test_size', type=float, default=0.2, help='Test size fraction')
parser.add_argument('--random_state', type=int, default=42, help='Random state for train_test_split')
args = parser.parse_args()

df = preprocess_data(args.file_path)

X = df.iloc[:, :-1] #delete
y = df.iloc[:, -1]   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

is_classification = y.nunique() <= 10  

# LazyPredict
if is_classification:
    clf = LazyClassifier(verbose=0, ignore_warnings=True, predictions=True)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
else:
    clf = LazyRegressor(verbose=0, ignore_warnings=True, predictions=True)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# output
print("Model Performance:")
print(models)
print("\nPredictions:")
print(predictions)

# save output and table
def save_results_as_images(models_df):
    models_df.reset_index(inplace=True)
    models_df.rename(columns={'index': 'Model'}, inplace=True)

    fig, ax = plt.subplots(figsize=(14, 10))  # fit to github
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=models_df.values, colLabels=models_df.columns, cellLoc='center', loc='center', colColours=['#f2f2f2'] * len(models_df.columns))

    table_path = 'model_performance_table.png'
    plt.savefig(table_path, dpi=300)

    fig, ax = plt.subplots(figsize=(14, 10))
    models_df.plot(kind='barh', x='Model', y='Accuracy', ax=ax, color='skyblue')
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Model')
    ax.set_title('Model Accuracy Comparison')

    bar_chart_path = 'model_performance_bar.png'
    plt.tight_layout()
    plt.savefig(bar_chart_path, dpi=300)

    return table_path, bar_chart_path

table_path, bar_chart_path = save_results_as_images(models)

print(f"Table and bar chart saved as '{table_path}' and '{bar_chart_path}'")
