# mlops_hse

## Как это юзать:

docker build . -t your_dockerhub_name/your_docker_image_name

docker-compose up

### host для проверки апи (можно из браузера через сваггер):

* http://127.0.0.1:5050

### получить список доступных классов для обучения:

* 127.0.0.1:5050/info

### создать модель выбранного класса и обучить:

* PUT 127.0.0.1:5050/models/your_model_name

### сделать предикт имеющейся моделью:

* GET 127.0.0.1:5050/models/your_model_name

### удалить имеющуюся модель:

* DELETE 127.0.0.1:5050/models/your_model_name
