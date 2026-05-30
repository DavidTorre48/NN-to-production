# https://core.telegram.org/bots#
# https://pytorch.org/get-started/locally/

import os
import requests
import cv2
from PIL import Image
import numpy as np
import telebot
import traceback
import torch
from torchvision import transforms
import config
from handler import *


bot = telebot.TeleBot(config.TOKEN)
classes=['Cat', 'Dog']

def load_model():
        model = torch.jit.load('PetImages.pt')
        if model is not None:
            model.eval()
            print("Модель успешно загружена")
        else:
            print("X Не удалось загрузить модель")
        return model

model = load_model()

transform = transforms.Compose(
    [transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def get_photo(message):
    photo = message.photo[1].file_id
    file_info = bot.get_file(photo)
    file_content = bot.download_file(file_info.file_path)
    return file_content

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Привет! Пришли фото сюда, а нейронная сеть определит на фото кота или собаку.\nАвтор: @Dave_Score')

@bot.message_handler(content_types=['photo'])
def repeat_all_messages(message):
    try:
        file_content = get_photo(message)
        image = byte2image(file_content)
        image=transform(image)
        model.eval()
        image=torch.unsqueeze(image, 0)
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        bot.send_message(message.chat.id,text='Обнаружено: {}'.format(classes[int(preds)]))
    except Exception:
        traceback.print_exc()
        bot.send_message(message.chat.id, 'Упс, что-то пошло не так :( Обратитесь в службу поддержки!')

if __name__ == '__main__':
    import time
    while True:
        try:
            print("Запуск бота")
            bot.polling(none_stop=True, timeout=50)
        except Exception as e:
            print(e)
            print("Перезапуск")
