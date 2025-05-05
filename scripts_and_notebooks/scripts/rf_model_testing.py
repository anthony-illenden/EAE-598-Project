import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from itertools import combinations
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report
from sklearn import metrics

zid = 1895595

df = pd.read_csv("final_all_events.csv")
df['vort_def'] =  df['rel_vort_1000'] / df['total_deformation_1000']
df['low_level_pv'] = (df['pv_925'] + df['pv_850']) / 2

train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=zid)
train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=zid)

variables_list = ['pv_300', 'pv_700', 'pv_850', 'pv_925', 'pv_1000',
       'wnd_300', 'wnd_500', 'wnd_850', 'z_250', 'z_500', 'z_850', 'z_925',
       'z_1000', 't_250', 't_500', 't_850', 't_925', 't_1000', 'q_850',
       'q_925', 'q_1000', 'ivt', 'ivt_grad', 'thickness_1000_500', 'qvec_div',
       'qvec_magn', 'abs_vort', 'thetae_850', 'thetae_925', 'thetae_1000',
       'fgen_700', 'fgen_850', 'fgen_925', 'fgen_1000', 'tadv_500', 'tadv_850',
       'tadv_925', 'tadv_1000', 'rel_vort_500', 'rel_vort_850', 'rel_vort_925',
       'rel_vort_1000', 'total_deformation_500', 'total_deformation_850',
       'total_deformation_925', 'total_deformation_1000',
       'shearing_deformation_500', 'shearing_deformation_850',
       'shearing_deformation_925', 'shearing_deformation_1000',
       'stretching_deformation_500', 'stretching_deformation_850',
       'stretching_deformation_925', 'stretching_deformation_1000',
       'thetae_grad_850', 'thetae_grad_925', 'thetae_grad_1000', 't_grad_850',
       't_grad_925', 't_grad_1000',
       'vort_def', 'low_level_pv']


def train_and_evaluate_rf(train_df, test_df, var1, var2, var3, label_col, random_state):
    """
    Train a Random Forest classifier and evaluate it.

    Parameters:
        train_df (pd.DataFrame): Training dataset.
        test_df (pd.DataFrame): Testing dataset.
        var1 (str): Name of the first variable.
        var2 (str): Name of the second variable.
        var3 (str): Name of the third variable.
        label_col (str): Name of the label column.
        random_state (int): Random state for reproducibility.

    Returns:
        RandomForestClassifier: Trained classifier.
        str: Classification report.

    """
    # Subset the training df to only keep the columns of the defined variables
    subset_train = train_df[[var1, var2, var3]]

    # Create a random forest classifier with the specified configuration
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state)

    # Train the random forest using the data from the variables as well as the labels
    rf_clf.fit(subset_train.values, train_df[label_col].values)

    # Subset the testing df to only keep the columns of the defined variables
    test_data = test_df[[var1, var2, var3]]

    # Use the random forest with the three variables' data for the predictions
    predicted = rf_clf.predict(test_data.values)

    # Get the expected labels
    expected = test_df[label_col].values

    # Generate the classification report
    classification_report = metrics.classification_report(expected, predicted)

    return rf_clf, classification_report

variable_triplets = list(combinations(variables_list, 3))
model_results = []

for var1, var2, var3 in variable_triplets:
    rf_clf, report = train_and_evaluate_rf(
        train_df=train_df,
        test_df=test_df,
        var1=var1,
        var2=var2,
        var3=var3,
        label_col="label",
        random_state=zid)

    predicted = rf_clf.predict(test_df[[var1, var2, var3]].values)
    expected = test_df["label"].values
    precision = precision_score(expected, predicted, average="weighted", zero_division=0)
    recall = recall_score(expected, predicted, average="weighted", zero_division=0)
    accuracy = accuracy_score(expected, predicted)

    model_results.append({
        "Var1": var1,
        "Var2": var2,
        "Var3": var3,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy})

results_df = pd.DataFrame(model_results)
results_df = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
top_50_df = results_df.head(50)
top_50_df.to_csv("top_50_variable_triplets.csv", index=False)
