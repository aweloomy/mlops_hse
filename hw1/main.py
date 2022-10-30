from flask import Flask

import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
from flask_restx import Api, Resource, reqparse, abort


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

req_put_args = reqparse.RequestParser()
req_put_args.add_argument(
    "model", type=str,
    help='name of model class is required',
    required=True
)
req_put_args.add_argument(
    "params",
    type=dict,
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
        model = request_data['model']
        params = request_data['params']

        abort_model_class_exist(model)
        model_params_is_right(model, params)

        model_to_train = eval(model)(**params)
        model_to_train.fit(X, y)
        joblib.dump(model_to_train, f'models/{model_name}')

        return '', 201

    def get(self, model_name):
        """get predict on test data"""

        abort_model_in_models(model_name)
        model_loaded = joblib.load(f'models/{model_name}')

        df_for_pred = df_test.iloc[:3, :2]
        pred = model_loaded.predict(df_for_pred)
        pred_lst = pred.tolist()

        return pred_lst

    @api.response(204, description='model deleted')
    def delete(self, model_name):
        """delete model"""

        abort_model_in_models(model_name)
        os.remove(f'models/{model_name}')

        return '', 204


if __name__ == '__main__':
    app.run(debug=True, port=5000)
