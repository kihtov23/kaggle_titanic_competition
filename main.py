import pandas as pd

import sklearn
from sklearn.linear_model import LogisticRegression


def apply_logistic_regressor():
    train = pd.read_csv('https://drive.google.com/uc?export=download&id=1u56FcuG2C_TgzPRM-J7BRdMgJUPJkjBB')
    test = pd.read_csv('https://drive.google.com/uc?export=download&id=12GTFLUmJQ8v9tW4AXcLEHBDjZNtobiuW')

    train_clean = train[['Pclass', 'Age', 'Survived']]
    train_clean = train_clean.fillna(0)
    # train_clean = train_clean.dropna()
    X_train = train_clean[['Pclass', 'Age']]
    Y_train = train_clean[['Survived']]

    test_clean = test[['PassengerId', 'Pclass', 'Age']]
    test_clean = test_clean.fillna(0)
    # test_clean = test_clean.dropna()
    X_test = test_clean[['Pclass', 'Age']]

    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict(X_test)

    submission = pd.DataFrame(data=Y_pred, index=test_clean.PassengerId, columns=['Survived'])
    submission.to_csv('titanic_submission')


if __name__ == '__main__':
    apply_logistic_regressor()
