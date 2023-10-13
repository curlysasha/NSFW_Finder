import os
import cv2
import shutil
import time
from rich import print
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

config_path = "C:/NSFW_Finder/deploy.prototxt"
model_path = "C:/NSFW_Finder/resnet_50_1by2_nsfw.caffemodel"

# Определите директории входных и выходных изображений
input_directory = "C:/BOT GIT/TelegramSDGenerator v1.3 youcass/OLDoutputs"
output_directory = "C:/NSFW_Finder/nude"

# Функция для определения NSFW контента
def is_nsfw(image_path):
    net = cv2.dnn.readNet(model_path, config_path)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping invalid: {os.path.basename(image_path)}")
        return 0.0
    blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), (104, 117, 123))
    net.setInput(blob)
    predictions = net.forward()
    return predictions[0][1]

# Создайте выходную директорию, если она не существует
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Инициализируйте пустой DataFrame для хранения информации об изображениях
results = pd.DataFrame(columns=["Filename", "Accuracy"])

# Соберем список всех изображений в директории входных изображений
image_files = [filename for filename in os.listdir(input_directory) if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

# Функция для обработки одного изображения
def process_image(image_path):
    nsfw_prob = is_nsfw(image_path)
    if nsfw_prob > 0.82:
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}-{nsfw_prob:.2f}{ext}"
        output_path = os.path.join(output_directory, output_filename)
        shutil.copyfile(image_path, output_path)
        print(f"Copied: {output_filename}")
        return {"Filename": output_filename, "Accuracy": nsfw_prob}
    return None

# Создадим прогресс бар с использованием tqdm
with tqdm(total=len(image_files), position=0, unit="image", dynamic_ncols=True, bar_format="{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") as pbar:
    # Создаем пул потоков для параллельной обработки изображений
    with ThreadPoolExecutor(max_workers=8) as executor:
        for result in executor.map(process_image, [os.path.join(input_directory, filename) for filename in image_files]):
            if result is not None:
                results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)
            pbar.update(1)  # Обновляем индикацию прогресса

# Используем rich для логирования
print("\n[bold]NSFW detection and copying completed.[/bold]")

# Сохраняем результаты в Excel-файл
df = pd.DataFrame(results)
df.to_excel("NSFW_Results.xlsx", index=False, engine="openpyxl")
print("Results saved to NSFW_Results.xlsx")
