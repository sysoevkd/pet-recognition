
# pet-recognition 

**Описание проекта**  
Проект представляет собой REST-сервис для распознавания домашних животных на изображениях.

**Структура проекта**
1. :file_folder: model training/ - содержит `oxford-iii-pets-classification.ipynb`, в котором находится код загрузки данных, дообучения модели и ее локального теста.
2. `Requirements.txt` - файл с зависимостями
3. `best_model.pth` - файл с весами обученной модели в PyTorch
4. `main.py` - основной исполняемый файл проекта, содержащий логику сервиса
5. `requests_api.py` - клиент для тестирования API
6. `supported_breeds.txt` - породы, на которых обучена модель; при использовании сервиса необходимо брать животных именно этих пород

**Локальный запуск сервиса**
1. Клонируйте репозиторий:
```
git clone https://github.com/kolyal/pet-recognition.git
cd pet-recognition
```
2. Установите зависимости:
`pip install -r Requirements.txt` (Рекомендуется создать виртуальное окружение с python 3.12.8 из-за несовместимости torch с более высокими версиями)
3. Запустите приложение: `uvicorn main:app --reload`

**Примеры запросов:**
После того как сервер запущен, можно выполнить `python requests_api.py` в терминале в папке проекта  
или через curl в PowerShell (вставить свою ссылку)  
```
curl -Method POST "http://127.0.0.1:8000/predict/json/" `
  -Headers @{"Content-Type"="application/json"} `
  -Body '{"image_url":"https://example.org/dog.jpg"}'
```
Также можно отправлять запросы через HTML-формы

**Стек:**
fastapi, pydantic, requests, torch, uvicorn, jupyter notebook, git, logging, postman

**Этапы работы:**  

Для проекта выбрал датасет **The Oxford-IIIT Pet Dataset**, На Kaggle дообучил модель **efficientnet_b4**, поменяв последний слой-классификатор (с GPU), замерил качество Accuracy (данные сбалансированны), протестировал, сохранил веса модели - код содержится в `oxford-iii-pets-classification.ipynb`  
Решил дополнительно реализовать HTML-интерфейс

В `main.py` прописал 4 endpoint:
1. `@app.get("/", response_class=HTMLResponse)` - главная страница
2. `@app.post("/predict/", response_class=HTMLResponse)` - предикт через форму по URL изображения
3. `@app.post("/predict/upload/", response_class=HTMLResponse)` - предикт через загрузку изображения с устройства
4. `@app.post("/predict/json/")` - предикт по URL изображения с возвратом результата в JSON-формате (для запросов curl, requests)

В каждом endpoint принимается изображение (через URL или с устройства) и обрабатывается 

В `requests_api.py` реализовал клиент для запросов по API, там же пример запроса через **requests**  
Также тестировл в postman, получал результаты применения к картинкам  

**Примечания:**  

Обрабатываются изображения формата JPEG, PNG, WEBP; при обращении к некоторым картинкам возможна ошибка Client Error: Forbidden for url (сервер может блокировать автоматические запросы), в таком случае лучше пользоваться загрузкой с устройства.

**Результат:**  
Обучил модель, которая с высокой точносью (Val Accuracy 90.92%) предсказывает породу домашнего животного на картинке, создал веб-сервис, встроил модель, захостил через Render - [Попробовать в браузере](https://pet-recognition-qpf3.onrender.com/) (может упасть + очень медленная обработка запросов, т.к. хостинг бесплатный).

