import base64
import logging
from io import BytesIO
import requests
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel
from torch import nn
from torchvision import models

# Модель pydantic для правильной обработки URL адреса
class ImageRequest(BaseModel):
    image_url: str

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # выбор девайса (для предиктов на усилителях)
MODEL_PATH = "best_model.pth"
CLASSES = 37 # количество меток класса для датасета Oxford-IIIT Pet Dataset
class_names = ['Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair',
                        'Chihuahua', 'Egyptian Mau', 'English Cocker Spaniel', 'English Setter', 'German Shorthaired', 'Great Pyrenees', 'Havanese', 'Japanese Chin',
                        'Keeshond', 'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'Newfoundland', 'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue',
                        'Saint Bernard', 'Samoyed', 'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx', 'Staffordshire Bull Terrier', 'Wheaten Terrier',
                        'Yorkshire Terrier'] # метки классов пришлось вытащить, чтобы не загружать датасет

def load_model():
    """
    Загружает предварительно обученную модель EfficientNet-B4 и дообучает последний слой.
    Модель загружается из файла `MODEL_PATH`, перемещается на указанное устройство
    и переводится в режим инференса.
    Отлавливает ошибки загрузки модели и логирует их.
    """
    try:
        model = models.efficientnet_b4(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, CLASSES) # дообучение efficientnet_b4 для моего датасета

        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()

        logger.info("Модель успешно загружена")
        return model
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {str(e)}")
        raise RuntimeError(f"Не удалось загрузить модель: {str(e)}")

def predict_image(model, image_tensor, class_names):
    """
    Применяет модель к изображению.

    :param model: Загруженная модель
    :param image_tensor: Тензор обработанного изображения
    :param class_names: Метки классов
    :return: Название предсказанного класса (породы), уверенность модели в предсказании
    """
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()
        confidence = torch.nn.functional.softmax(output, dim=1)[0][class_idx].item()
        return class_names[class_idx], confidence

def image_to_base64(img: Image.Image) -> str:
    """
    Конвертирует изображение PIL в строку base64 для передачи через API/веб.
    Функция сохраняет изображение в бинарном формате PNG в памяти, затем кодирует
    его в строку base64, которую можно использовать в HTML, JSON или API.

    :param img: Входное изображение в формате PIL.Image.
    :return: Строка base64, представляющая изображение в формате PNG.
    """
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


model = load_model() # загрузка модели

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]) # трансформер обрабатываемых изображений

@app.get("/", response_class=HTMLResponse)
async def main():
    """
    Главная страница сервиса классификации животных.

    Возвращает HTML-страницу с двумя формами для отправки изображений:
    1. Через публичный URL
    2. Через загрузку файла с устройства
    """
    return """
    <html>
        <head>
            <title>Предсказание породы животного</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f9f9f9;
                    color: #333;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }
                h2 {
                    margin-top: 40px;
                    color: #444;
                }
                form {
                    margin-top: 20px;
                    margin-bottom: 40px;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 10px;
                    background-color: #fff;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
                    width: 100%;
                    max-width: 500px;
                }
                input[type="text"], input[type="file"] {
                    width: 100%;
                    padding: 10px;
                    margin-top: 10px;
                    margin-bottom: 15px;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                }
                input[type="submit"] {
                    padding: 10px 20px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                }
                input[type="submit"]:hover {
                    background-color: #45a049;
                }
                @media (prefers-color-scheme: dark) {
                    body {
                        background-color: #121212;
                        color: #eee;
                    }
                    form {
                        background-color: #1e1e1e;
                        border: 1px solid #333;
                    }
                    input[type="text"], input[type="file"] {
                        background-color: #2c2c2c;
                        color: white;
                        border: 1px solid #444;
                    }
                    input[type="submit"] {
                        background-color: #388e3c;
                    }
                    input[type="submit"]:hover {
                        background-color: #2e7d32;
                    }
                }
            </style>
        </head>
        <body>
            <h1>Предсказание породы животного</h1>

            <form action="/predict/" method="post">
                <label for="image_url"><strong>Вставьте ссылку на изображение:</strong></label>
                <input type="text" name="image_url" id="image_url" placeholder="https://example.com/cat.jpg" required>
                <input type="submit" value="Предсказать по URL">
            </form>

            <form action="/predict/upload/" enctype="multipart/form-data" method="post">
                <label for="file"><strong>Загрузите изображение с компьютера:</strong></label>
                <input type="file" name="file" id="file" accept="image/*" required>
                <input type="submit" value="Предсказать по файлу">
            </form>
        </body>
    </html>
    """


@app.post("/predict/", response_class=HTMLResponse)
async def predict_from_form(image_url: str = Form(...)):
    """
    Обрабатывает изображение по URL и возвращает предсказание породы животного.

    Принимает URL изображения, загружает изображение, преобразует его
    в тензор, передает в модель и возвращает предсказание с уверенностью модели.
    Предназначен для отправки изображения через HTML макет.

    :param image_url: URL изображения
    :return: HTML-страница с предсказанием и уверенностью модели.
    """
    try:
        logger.info(f"Обработка изображения по URL: {image_url}")
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        predicted_class, confidence = predict_image(model, img_tensor, class_names)

        result = f"""
        <html>
            <head>
                <title>Результат предсказания</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: #f9f9f9;
                        color: #333;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                    }}
                    h2 {{
                        margin-top: 40px;
                        color: #444;
                    }}
                    .result {{
                        margin-top: 30px;
                        padding: 20px;
                        border: 1px solid #ddd;
                        border-radius: 10px;
                        background-color: #fff;
                        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
                        max-width: 500px;
                        width: 100%;
                        text-align: center;
                    }}
                    img {{
                        margin-top: 15px;
                        border-radius: 10px;
                        max-width: 100%;
                        height: auto;
                    }}
                    a {{
                        display: inline-block;
                        margin-top: 20px;
                        text-decoration: none;
                        padding: 10px 20px;
                        background-color: #4CAF50;
                        color: white;
                        border-radius: 5px;
                    }}
                    a:hover {{
                        background-color: #45a049;
                    }}
                    @media (prefers-color-scheme: dark) {{
                        body {{
                            background-color: #121212;
                            color: #eee;
                        }}
                        .result {{
                            background-color: #1e1e1e;
                            border: 1px solid #333;
                        }}
                        a {{
                            background-color: #388e3c;
                        }}
                        a:hover {{
                            background-color: #2e7d32;
                        }}
                    }}
                </style>
            </head>
            <body>
                <div class="result">
                    <h2>Результат предсказания</h2>
                    <p><strong>Животное на картинке:</strong> {predicted_class}</p>
                    <p><strong>Уверенность модели:</strong> {confidence:.2f}</p>
                    <img src="{image_url}" alt="Изображение животного">
                    <br>
                    <a href="/">← Назад</a>
                </div>
            </body>
        </html>
        """

        logger.info(f"Предсказание готово: {predicted_class} ({confidence:.2f})")
        return result

    except requests.exceptions.RequestException as e:
        logger.error(f"Неверный URL: {str(e)}")
        return HTMLResponse(content=f"<p>Ошибка: Не удалось загрузить изображение. {str(e)}</p><a href='/'>Назад</a>", status_code=400)
    except Exception as e:
        logger.error(f"Ошибка предсказания: {str(e)}")
        return HTMLResponse(content=f"<p>Ошибка сервера: {str(e)}</p><a href='/'>Назад</a>", status_code=500)


@app.post("/predict/upload/", response_class=HTMLResponse)
async def predict_from_upload(file: UploadFile = File(...)):
    """
    Обрабатывает загруженное изображение и возвращает HTML-страницу с результатом классификации.
    Принимает файл изображения через форму, определяет породу животного с помощью ML-модели
    и возвращает стилизованную HTML-страницу с результатом и превью изображения.

    :param file: Изображениz в формате JPEG, PNG, WEBP
    :return: Стилизованную HTML-страница с результатом и превью изображения.
    """
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        predicted_class, confidence = predict_image(model, img_tensor, class_names)

        result = f"""
        <html>
            <head>
                <title>Результат предсказания</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: #f9f9f9;
                        color: #333;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                    }}
                    h2 {{
                        margin-top: 40px;
                        color: #444;
                    }}
                    .result {{
                        margin-top: 30px;
                        padding: 20px;
                        border: 1px solid #ddd;
                        border-radius: 10px;
                        background-color: #fff;
                        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
                        max-width: 500px;
                        width: 100%;
                        text-align: center;
                    }}
                    img {{
                        margin-top: 15px;
                        border-radius: 10px;
                        max-width: 100%;
                        height: auto;
                    }}
                    a {{
                        display: inline-block;
                        margin-top: 20px;
                        text-decoration: none;
                        padding: 10px 20px;
                        background-color: #4CAF50;
                        color: white;
                        border-radius: 5px;
                    }}
                    a:hover {{
                        background-color: #45a049;
                    }}
                    @media (prefers-color-scheme: dark) {{
                        body {{
                            background-color: #121212;
                            color: #eee;
                        }}
                        .result {{
                            background-color: #1e1e1e;
                            border: 1px solid #333;
                        }}
                        a {{
                            background-color: #388e3c;
                        }}
                        a:hover {{
                            background-color: #2e7d32;
                        }}
                    }}
                </style>
            </head>
            <body>
                <div class="result">
                    <h2>Результат предсказания</h2>
                    <p><strong>Животное на картинке:</strong> {predicted_class}</p>
                    <p><strong>Уверенность модели:</strong> {confidence:.2f}</p>
                    <img src="data:image/png;base64,{image_to_base64(img)}" alt="Загруженное изображение">
                    <br>
                    <a href="/">← Назад</a>
                </div>
            </body>
        </html>
        """

        return result

    except Exception as e:
        logger.error(f"Ошибка предсказания по файлу: {str(e)}")
        return HTMLResponse(content=f"<p>Ошибка обработки файла: {str(e)}</p><a href='/'>Назад</a>", status_code=400)


@app.post("/predict/json/")
async def predict_from_url(request: ImageRequest):
    """
    Обрабатывает изображение по URL и возвращает предсказание породы животного.

    Принимает URL изображения в формате JSON, загружает изображение, преобразует его
    в тензор, передает в модель и возвращает предсказание с уверенностью модели.
    Предназначен для отправки запросов через requests, curl без HTML

    :param request: URL в формате JSON
    :return: Предсказание с уверенностью модели.
    """
    image_url = request.image_url
    try:
        logger.info(f"Обработка изображения по URL: {image_url}")
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        predicted_class, confidence = predict_image(model, img_tensor, class_names)

        result = {
            "Животное на картинке: ": predicted_class,
            "Уверенность модели: ": float(confidence)
        }

        logger.info(f"Предсказание готово: {result}")
        return result

    except requests.exceptions.RequestException as e:
        logger.error(f"Неверный URL: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Не удалось загрузить изображение: {str(e)}")
    except Exception as e:
        logger.error(f"Ошибка предсказания: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Не удалось сделать предсказание: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
