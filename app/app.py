import streamlit as st
import requests
from PIL import Image
import io

url = "http://95.163.228.196:8000/ai/detect"

st.title("LazyLearn. Распознавание документов")

if (st.button("Информация о приложении")):
    st.info("Перед Вами web-приложение, предназначенное для распознавания документов и ключевой информации. Для работы необходимо загрузить изображение и немного подождать. Программа выдаст результат в течение некоторого времени.")

uploaded_image = st.file_uploader("Выберите изображение", type="jpg")
if uploaded_image is not None:
    img = Image.open(uploaded_image)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    payload = {"image": img_byte_arr}
    response = requests.post(url, files=payload)
    api_result = response.json()

    if api_result != "Не удалось распознать фотографию. Пожалуйста, проверьте качество изображения.":
        if api_result.get("series") == "":
            api_result['series'] = "не удалось распознать"
        if api_result.get("number") == "":
            api_result['number'] = "не удалось распознать"

        st.write(f'### Тип загруженного документа: {api_result.get("type")}')
        st.write(f'### Вероятность предсказания: {api_result.get("confidence")}')
        st.write(f'### Серия документа: {api_result.get("series")}')
        st.write(f'### Номер документа: {api_result.get("number")}')
        st.write(f'### Номер страницы: {api_result.get("page_number")}')
