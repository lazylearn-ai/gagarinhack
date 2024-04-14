"""
API для предсказания сканов документов
"""
from fastapi import APIRouter, FastAPI, File, UploadFile
from PIL import Image
from nn.detect_doc import predict_class_proba_page_box, crop_box
from nn.detect_text import predict_text, split_to_seria_and_number
ai_router = APIRouter()

@ai_router.post('/detect')
def detect(image: UploadFile = File(...)):
    im = Image.open(image.file)
    try:
        cls, proba, page, box = predict_class_proba_page_box(im)
        im = crop_box(im, box)
        text = predict_text(im)
        seria, number = split_to_seria_and_number(text, cls)
        return {
            "type": cls,
            "confidence": str(proba),
            "series": seria,
            "number": number,
            "page_number": page
        }
    except:
        return "Не удалось распознать фотографию. Пожалуйста, проверьте качество изображения."
