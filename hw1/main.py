from flask import Flask

import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from flask_restx import Api, Resource, reqparse, abort
import glob
import pickle
import ast

from sqlalchemy import create_engine

POSTGRES_HOST='service-db'
POSTGRES_DB='postgres_db'
POSTGRES_USER='postgres_db_user'
POSTGRES_PASSWORD='postgres_db_password'

# POSTGRES_HOST = os.environ['POSTGRES_HOST']
# POSTGRES_DB = os.environ['POSTGRES_DB']
# POSTGRES_USER = os.environ['POSTGRES_USER']
# POSTGRES_PASSWORD = os.environ['POSTGRES_PASSWORD']

POSTGRES_CONN_STRING = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:5432/{POSTGRES_DB}"

def get_data(conn=POSTGRES_CONN_STRING):
    engine_postgres = create_engine(conn)
    data = pd.read_sql_query(
        """
        SELECT
            "col0",
            "col1",
            "target"
        FROM public.dataset;
        """,
        engine_postgres
    )
    engine_postgres.dispose()
    return data


app = Flask(__name__)
api = Api(
    app,
    version="1.0",
    title="ML REST API",
    description="some SOTA decisions for my data that you can try",
    default="How to use",
    default_label="my models",
    contact="tg: @aweloomy",
    contact_email="smth"

)

df = pd.read_csv('data/train.csv')
X = df.iloc[:, :2]
y = df['target']
df_test = pd.read_csv('data/test.csv')
model_class_list = ['LogisticRegression', 'RandomForestClassifier']

def create_model_store(model_dir):
    files = glob.glob(f'./{model_dir}/*')
    model_store = {}
    for file in files:
        model_store[file.split('/')[-1][:-4]] = file
    return model_store

model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_store = create_model_store(model_dir)

req_put_args = reqparse.RequestParser()
req_put_args.add_argument(
    "model", type=str,
    location='args',
    help='name of model class is required',
    required=True
)
req_put_args.add_argument(
    "params",
    type=str,
    location='args',
    help='params of model',
    required=False
)


def abort_model_class_exist(model):
    if model not in model_class_list:
        abort(message='model class name is not valid')


def abort_model_in_models(model_name):
    if model_name not in os.listdir('models'):
        abort(message="model doesn't exist")


def model_params_is_right(model_name, params):
    try:
        eval(model_name)(**params)
    except:
        abort(message="parameters ate not suitable for the model class")


@api.route('/info', methods=['GET'],
           doc={'description': 'class about available models'})
class Info(Resource):

    def get(self):
        """get list of available classes"""
        # get_data()
        return model_class_list


@api.route('/models/<string:model_name>',
           methods=['GET', 'DELETE', 'PUT'])
class Models(Resource):

    @api.response(201, description='model created')
    @api.doc(params={
        'model': 'class of sklearn models, '
                 'it can be LogisticRegression or RandomForestClassifier',
        'params': 'relevant params dict for model'})

    def put(self, model_name):
        """train and save model"""

        request_data = req_put_args.parse_args()
        model = request_data.get('model')
        params = request_data.get('params')

        if params is None:
            params = {}
        else:
            params = ast.literal_eval(params)

        data = get_data()
        data_train = data.iloc[:, :2]
        data_target = data['target']
    #
    #     # abort_model_class_exist(model)
    #     # model_params_is_right(model, params)

        model_to_train = eval(model)(**params)
        model_to_train.fit(data_train, data_target)


        with open(os.path.join(model_dir, f'{model_name}.pkl'), 'wb') as f:
            pickle.dump(model_to_train, f)
        model_store[model_name] = os.path.join(model_dir, f'{model_name}.pkl')
        # joblib.dump(model_to_train, f'{model_dir}/{model_name}')
        # print('model saved (no)')
        return '', 201

    def get(self, model_name):
        """get predict on test data"""

        abort_model_in_models(model_name)

        with open(model_name, 'rb') as f:
            model_loaded = pickle.load(f)

        data = get_data()
        data_test = data.iloc[:, :2]
        pred = model_loaded.predict(data_test)
        pred_lst = pred.tolist()

        return pred_lst

    @api.response(204, description='model deleted')
    def delete(self, model_name):
        """delete model"""

        abort_model_in_models(model_name)

        try:
            model_path = model_store[model_name]
            os.remove(model_path)
            del model_store[model_name]
        except KeyError:
            raise KeyError('Model not found')

        os.remove(f'models/{model_name}')

        return '', 204


if __name__ == '__main__':
    app.run(debug=True,
            port=5000,
            host='0.0.0.0'
            )
