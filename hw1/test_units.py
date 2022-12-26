import pandas as pd

from main import model_class_list

test_params_lr = {'penalty': 'l2', 'C': 0.3}
test_params_rf = {'max_depth': 5, 'n_estimators': 100}
test_bad_params_lr = {'penalty': 'l3', 'C': 0.3}
test_bad_params_rf = {'max_depth': -1, 'n_estimators': 100}
test_bad_params = {'penalty': 'l3', 'n_estimators': 100}


def test_data(data_path='postgres/data.csv'):

    extension = data_path.split('.')[-1]
    assert extension == 'csv'

    data = pd.read_csv(data_path)

    assert data.columns.to_list() == ['col0', 'col1', 'target']

    # data_train = data.iloc[:, :2]
    # data_target = data['target']

    assert data['col0'].dtype == 'int64'
    assert data['col1'].dtype == 'int64'
    assert data['target'].dtype == 'int64'
    assert data.isna().sum().sum() == 0

    assert set(data['target'].unique()) == {0, 1}, 'not binary classification data'


def test_model_classes(list_of_classes=model_class_list):

    assert len(list_of_classes) > 0
