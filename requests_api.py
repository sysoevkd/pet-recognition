import requests
import json

API_URL = "http://localhost:8000/predict/json/"
IMAGE_URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcReYYQ7MAucpL8KFNgGYvclTaN1EfzpNOxBZby0qsdb-ur0z4JmEcmbHKcny4uNqpCCdiDM6aTLqdR3J8HfQhyiUA"

data = {
    "image_url": IMAGE_URL
}
headers = {
    "Content-Type": "application/json"
}

try:
    response = requests.post(
        url=API_URL,
        data=json.dumps(data),
        headers=headers
    )

    if response.status_code == 200:
        print("Успешный запрос")
        print("Результат предсказания:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    else:
        print(f"Ошибка. Статус код: {response.status_code}")
        print("Ответ сервера:", response.text)

except requests.exceptions.RequestException as e:
    print(f"Произошла ошибка при выполнении запроса: {str(e)}")
except json.JSONDecodeError:
    print("Ошибка декодирования JSON ответа")