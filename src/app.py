from utils import db_connect
engine = db_connect()

# Handle imports up-front
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import uniform, norm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Load the data from the URL
df=pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv")

# Separate features from labels
labels=df['Outcome']
features=df.drop('Outcome', axis=1)

# Split the data into training and testing features and labels
X_train, X_test, y_train, y_test=train_test_split(
    features,
    labels,
    test_size=0.2,
    random_state=315
)

# Define a reusable helper function for cross-validation here. We are going to
# be doing a lot of cross-validation, this allows us to reuse this code
# without having to copy-paste it over and over.

def cross_val(model, features: pd.DataFrame, labels: pd.Series) -> list[float]:
    '''Reusable helper function to run cross-validation on a model. Takes model,
    Pandas data frame of features and Pandas data series of labels. Returns 
    list of cross-validation fold accuracy scores as percents.'''

    # Define the cross-validation strategy
    cross_validation=StratifiedKFold(n_splits=7, shuffle=True, random_state=315)

    # Run the cross-validation, collecting the scores
    scores=cross_val_score(
        model,
        features,
        labels,
        cv=cross_validation,
        n_jobs=-1,
        scoring='accuracy'
    )

    # Print mean and standard deviation of the scores
    print(f'Cross-validation accuracy: {(scores.mean() * 100):.2f} +/- {(scores.std() * 100):.2f}%')

    # Return the scores
    return scores

# Instantiate a random forest classifier model
model=DecisionTreeClassifier(random_state=315)
fit_result = model.fit(X_train, y_train)

# Run the cross-validation
scores=cross_val(model, X_train, y_train)

# Visualizing the tree

fig = plt.figure(figsize = (15,15))

tree.plot_tree(model, feature_names = list(X_train.columns), filled = True)

plt.show()

X_train.hist(density=True, layout=(3,3))
plt.tight_layout()
plt.show()

y_pred = model.predict(X_test)

# Getting the accuracy of the model
accuracy = accuracy_score(y_test, y_pred) * 100

print(f"The model is {accuracy:.1f}% accurate")
