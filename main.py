import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score

def get_predictions():
    train = pd.read_csv('https://drive.google.com/uc?export=download&id=1u56FcuG2C_TgzPRM-J7BRdMgJUPJkjBB')

    y = train.Survived
    x = train.drop('Survived', axis=1)

    x = pd.get_dummies(x)

    train_x, test_x, train_y, test_y = train_test_split(x, y)
    # print(train_x.shape)
    # print(train_y.shape)

    pipeline = make_pipeline(Imputer(), LogisticRegression())
    pipeline.fit(train_x, train_y)
    predictions = pipeline.predict(test_x)

    print(accuracy_score(test_y, predictions, normalize=True))


def get_submission():
    train = pd.read_csv('https://drive.google.com/uc?export=download&id=1u56FcuG2C_TgzPRM-J7BRdMgJUPJkjBB')

    train_y = train.Survived
    train_x = train.drop('Survived', axis=1)
    train_x = pd.get_dummies(train_x)

    test_x = pd.read_csv('https://drive.google.com/uc?export=download&id=12GTFLUmJQ8v9tW4AXcLEHBDjZNtobiuW')
    test_x = pd.get_dummies(test_x)

    train_x, test_x = train_x.align(test_x, join='left', axis=1)

    pipeline = make_pipeline(Imputer(), LogisticRegression())
    pipeline.fit(train_x, train_y)
    predictions = pipeline.predict(test_x)

    submission = pd.DataFrame(data=predictions, index=test_x.PassengerId, columns=['Survived'])
    submission.to_csv('titanic_submission')


if __name__ == '__main__':
    # get_predictions()
    get_submission()
