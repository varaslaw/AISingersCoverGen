import json
import os
import shutil
import urllib.request
import zipfile
from argparse import ArgumentParser
import time
import gradio as gr
import gdown
from mega import Mega
import requests

from main import song_cover_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Создаем директорию для загрузок, если она не существует
os.makedirs(os.path.join(BASE_DIR, 'uploads'), exist_ok=True)

mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')

mega = Mega()
m = mega.login()

def get_current_models(models_dir):
    models_list = os.listdir(models_dir)
    items_to_remove = ['hubert_base.pt', 'MODELS.txt', 'public_models.json', 'rmvpe.pt']
    return [item for item in models_list if item not in items_to_remove]

def update_models_list():
    models_l = get_current_models(rvc_models_dir)
    return gr.Dropdown.update(choices=models_l)

def load_public_models():
    models_table = []
    for model in public_models['voice_models']:
        if model['name'] not in voice_models:
            model_info = [model['name'], model['description'], model['credit'], model['url'], ', '.join(model['tags'])]
            models_table.append(model_info)

    tags = list(public_models['tags'].keys())
    return gr.DataFrame.update(value=models_table), gr.CheckboxGroup.update(choices=tags)

def extract_zip(extraction_folder, zip_name):
    os.makedirs(extraction_folder, exist_ok=True)
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(extraction_folder)
    os.remove(zip_name)

    index_filepath, model_filepath = None, None
    for root, dirs, files in os.walk(extraction_folder):
        for name in files:
            if name.endswith('.index') and os.stat(os.path.join(root, name)).st_size > 1024 * 100:
                index_filepath = os.path.join(root, name)

            if name.endswith('.pth') and os.stat(os.path.join(root, name)).st_size > 1024 * 1024 * 40:
                model_filepath = os.path.join(root, name)

    if not model_filepath:
        raise gr.Error(f'Не найден .pth файл модели в извлеченном zip. Проверьте {extraction_folder}.')

    os.rename(model_filepath, os.path.join(extraction_folder, os.path.basename(model_filepath)))
    if index_filepath:
        os.rename(index_filepath, os.path.join(extraction_folder, os.path.basename(index_filepath)))

    for filepath in os.listdir(extraction_folder):
        if os.path.isdir(os.path.join(extraction_folder, filepath)):
            shutil.rmtree(os.path.join(extraction_folder, filepath))

def convert_drive_url(url):
    if "drive.google.com" in url:
        file_id = url.split('/d/')[1].split('/')[0]
        direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        return direct_url
    return url

def download_yandex_disk_file(yandex_url, output):
    download_url = get_yandex_disk_download_url(yandex_url)
    if download_url:
        urllib.request.urlretrieve(download_url, output)
    else:
        raise gr.Error("Не удалось получить ссылку для загрузки с Яндекс Диска.")

def get_yandex_disk_download_url(public_url):
    api_endpoint = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
    response = requests.get(api_endpoint, params={'public_key': public_url})
    download_url = response.json().get('href')
    return download_url

def download_online_model(url, dir_name, progress=gr.Progress()):
    try:
        progress(0, desc=f'[~] Загрузка голосовой модели с именем {dir_name}...')
        zip_name = url.split('/')[-1]
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Каталог голосовой модели {dir_name} уже существует! Выберите другое имя для своей голосовой модели.')

        url = convert_drive_url(url)

        if 'drive.google.com' in url:
            gdown.download(url, zip_name, quiet=False)
        elif 'mega.nz' in url:
            m.download_url(url, dest_filename=zip_name)
        elif 'pixeldrain.com' in url:
            url = f'https://pixeldrain.com/api/file/{zip_name}'
            urllib.request.urlretrieve(url, zip_name)
        elif 'disk.yandex.ru' in url:
            download_yandex_disk_file(url, zip_name)
        else:
            urllib.request.urlretrieve(url, zip_name)

        progress(0.5, desc='[~] Извлечение zip...')
        extract_zip(extraction_folder, zip_name)
        return f'[+] Модель {dir_name} успешно загружена!'

    except Exception as e:
        raise gr.Error(str(e))

def upload_local_model(zip_path, dir_name, progress=gr.Progress()):
    try:
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Каталог голосовой модели {dir_name} уже существует! Выберите другое имя для своей голосовой модели.')

        zip_name = zip_path.name
        progress(0.5, desc='[~] Извлечение zip...')
        extract_zip(extraction_folder, zip_name)
        return f'[+] Модель {dir_name} успешно загружена!'

    except Exception as e:
        raise gr.Error(str(e))

def filter_models(tags, query):
    models_table = []

    if len(tags) == 0 and len(query) == 0:
        for model in public_models['voice_models']:
            models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])

    elif len(tags) > 0 and len(query) > 0:
        for model in public_models['voice_models']:
            if all(tag in model['tags'] for tag in tags):
                model_attributes = f"{model['name']} {model['description']} {model['credit']} {' '.join(model['tags'])}".lower()
                if query.lower() in model_attributes:
                    models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])

    elif len(tags) > 0:
        for model in public_models['voice_models']:
            if all(tag in model['tags'] for tag in tags):
                models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])

    else:
        for model in public_models['voice_models']:
            model_attributes = f"{model['name']} {model['description']} {model['credit']} {' '.join(model['tags'])}".lower()
            if query.lower() in model_attributes:
                models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])

    return gr.DataFrame.update(value=models_table)

def pub_dl_autofill(pub_models, event: gr.SelectData):
    return gr.Text.update(value=pub_models.loc[event.index[0], 'URL']), gr.Text.update(value=pub_models.loc[event.index[0], 'Model Name'])

def swap_visibility():
    return gr.update(visible=True), gr.update(visible=False), gr.update(value=''), gr.update(value=None)

def process_file_upload(file):
    return file.name, gr.update(value=file.name)

def process_record_upload(file):
    if file is None:
        raise gr.Error("Пожалуйста, сначала запишите аудио.")
    
    print(f"Временный путь файла: {file}")
    
    temp_path = file
    new_path = os.path.join(BASE_DIR, 'uploads', f"recorded_audio_{int(time.time())}.wav")
    
    print(f"Новый путь файла: {new_path}")
    
    shutil.copy(temp_path, new_path)
    
    print(f"Файл скопирован: {os.path.exists(new_path)}")
    
    return new_path, gr.update(value=new_path)

def show_hop_slider(pitch_detection_algo):
    if pitch_detection_algo == 'mangio-crepe':
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

if __name__ == '__main__':
    parser = ArgumentParser(description='Создание AI кавера песни в каталоге song_output/id.', add_help=True)
    parser.add_argument("--share", action="store_true", dest="share_enabled", default=False, help="Включить обмен")
    parser.add_argument("--listen", action="store_true", default=False, help="Сделать WebUI доступным из вашей локальной сети.")
    parser.add_argument('--listen-host', type=str, help='Хост, который будет использовать сервер.')
    parser.add_argument('--listen-port', type=int, help='Порт, который будет использовать сервер.')
    args = parser.parse_args()

    voice_models = get_current_models(rvc_models_dir)
    with open(os.path.join(rvc_models_dir, 'public_models.json'), encoding='utf8') as infile:
        public_models = json.load(infile)

    with gr.Blocks(title='🐳 AISINGERS') as app:

        gr.Label('🐳 AISINGERS | https://t.me/aisingers', show_label=False)

        with gr.Tab("Генерация"):

            with gr.Accordion('Основные опции'):
                with gr.Row():
                    with gr.Column():
                        rvc_model = gr.Dropdown(voice_models, label='Голосовые модели', info='Папка моделей "AICoverGen --> rvc_models". После добавления новых моделей в эту папку нажмите кнопку обновления.')
                        ref_btn = gr.Button('Обновить модели 🔁', variant='primary')

                    with gr.Column() as yt_link_col:
                        song_input = gr.Text(label='Ввод песни', info='Ссылка на песню на YouTube или полный путь к локальному файлу. Для загрузки файла нажмите кнопку ниже.')
                        show_file_upload_button = gr.Button('Загрузить файл')

                    with gr.Column(visible=False) as file_upload_col:
                        local_file = gr.File(label='Аудио файл')
                        song_input_file = gr.UploadButton('Загрузить 📂', file_types=['audio'], variant='primary')
                        show_yt_link_button = gr.Button('Вставить ссылку YouTube/Путь к локальному файлу')
                        song_input_file.upload(process_file_upload, inputs=[song_input_file], outputs=[local_file, song_input])

                    with gr.Column():
                        record_button = gr.Audio(label='Записать вокал', source="microphone", type="filepath")
                        upload_record_button = gr.Button('Загрузить запись')
                        upload_record_button.click(process_record_upload, inputs=[record_button], outputs=[local_file, song_input])

                    with gr.Column():
                        pitch = gr.Slider(-20, 20, value=0, step=1, label='Изменение высоты (только вокал)', info='Обычно используется 12 для преобразования мужского голоса в женский и -12 наоборот. (Октавы)')
                        pitch_all = gr.Slider(-12, 12, value=0, step=1, label='Общее изменение высоты', info='Изменяет высоту/тон вокала и инструментов вместе. Незначительное изменение ухудшает качество звука. (Полутона)')
                    show_file_upload_button.click(swap_visibility, outputs=[file_upload_col, yt_link_col, song_input, local_file])
                    show_yt_link_button.click(swap_visibility, outputs=[yt_link_col, file_upload_col, song_input, local_file])

            with gr.Accordion('Опции преобразования голоса', open=False):
                with gr.Row():
                    index_rate = gr.Slider(0, 1, value=0.5, label='Скорость индексации', info="Контролирует, насколько акцент AI-голоса сохраняется в вокале")
                    filter_radius = gr.Slider(0, 7, value=3, step=1, label='Радиус фильтра', info='Если >=3: применяется медианная фильтрация к извлеченным результатам высоты звука. Может уменьшить шумность')
                    rms_mix_rate = gr.Slider(0, 1, value=0.25, label='Скорость смешивания RMS', info="Контролирует, насколько оригинальная громкость вокала сохраняется (0) или устанавливается фиксированная громкость (1)")
                    protect = gr.Slider(0, 0.5, value=0.33, label='Скорость защиты', info='Защищает глухие согласные и звуки дыхания. Установите на 0.5, чтобы отключить.')
                    with gr.Column():
                        f0_method = gr.Dropdown(['rmvpe', 'mangio-crepe'], value='rmvpe', label='Алгоритм определения высоты звука', info='Лучший вариант rmvpe (ясность вокала), затем mangio-crepe (гладкость вокала)')
                        crepe_hop_length = gr.Slider(32, 320, value=128, step=1, visible=False, label='Длина хопа Crepe', info='Низкие значения приводят к более длительным конверсиям и высокому риску голосовых дефектов, но лучшей точности высоты звука.')
                        f0_method.change(show_hop_slider, inputs=f0_method, outputs=crepe_hop_length)
                keep_files = gr.Checkbox(label='Сохранять промежуточные файлы', info='Сохранять все аудиофайлы, созданные в каталоге song_output/id, например, изолированные вокалы/инструменталы. Оставьте отключенным для экономии места')

            with gr.Accordion('Опции микширования аудио', open=False):
                gr.Markdown('### Изменение громкости (децибелы)')
                with gr.Row():
                    main_gain = gr.Slider(-20, 20, value=0, step=1, label='Основной вокал')
                    backup_gain = gr.Slider(-20, 20, value=0, step=1, label='Резервный вокал')
                    inst_gain = gr.Slider(-20, 20, value=0, step=1, label='Музыка')

                gr.Markdown('### Контроль реверберации на AI вокале')
                with gr.Row():
                    reverb_rm_size = gr.Slider(0, 1, value=0.15, label='Размер комнаты', info='Чем больше комната, тем дольше время реверберации')
                    reverb_wet = gr.Slider(0, 1, value=0.2, label='Уровень влажности', info='Уровень AI вокала с реверберацией')
                    reverb_dry = gr.Slider(0, 1, value=0.8, label='Уровень сухости', info='Уровень AI вокала без реверберации')
                    reverb_damping = gr.Slider(0, 1, value=0.7, label='Уровень затухания', info='Поглощение высоких частот в реверберации')

                gr.Markdown('### Формат аудиовыхода')
                output_format = gr.Dropdown(['mp3', 'wav'], value='mp3', label='Тип выходного файла', info='mp3: малый размер файла, приличное качество. wav: большой размер файла, лучшее качество')

            with gr.Row():
                clear_btn = gr.ClearButton(value='Очистить', components=[song_input, rvc_model, keep_files, local_file])
                generate_btn = gr.Button("Сгенерировать", variant='primary')
                ai_cover = gr.Audio(label='AI кавер', show_share_button=False)

            ref_btn.click(update_models_list, None, outputs=rvc_model)
            is_webui = gr.Number(value=1, visible=False)
            generate_btn.click(song_cover_pipeline,
                               inputs=[song_input, rvc_model, pitch, keep_files, is_webui, main_gain, backup_gain,
                                       inst_gain, index_rate, filter_radius, rms_mix_rate, f0_method, crepe_hop_length,
                                       protect, pitch_all, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping,
                                       output_format],
                               outputs=[ai_cover])
            clear_btn.click(lambda: [0, 0, 0, 0, 0.5, 3, 0.25, 0.33, 'rmvpe', 128, 0, 0.15, 0.2, 0.8, 0.7, 'mp3', None],
                            outputs=[pitch, main_gain, backup_gain, inst_gain, index_rate, filter_radius, rms_mix_rate,
                                     protect, f0_method, crepe_hop_length, pitch_all, reverb_rm_size, reverb_wet,
                                     reverb_dry, reverb_damping, output_format, ai_cover])

        with gr.Tab('Скачать модель'):

            with gr.Tab('Ссылки HuggingFace/Pixeldrain/Google Drive/Mega/Яндекс Диск'):
                with gr.Row():
                    model_zip_link = gr.Text(label='Ссылка на загрузку модели', info='Должен быть zip файл, содержащий .pth файл модели и опциональный .index файл.')
                    model_name = gr.Text(label='Назовите свою модель', info='Дайте вашей новой модели уникальное имя, отличное от других голосовых моделей.')

                with gr.Row():
                    download_btn = gr.Button('Скачать 🌐', variant='primary', scale=19)
                    dl_output_message = gr.Text(label='Выходное сообщение', interactive=False, scale=20)

                download_btn.click(download_online_model, inputs=[model_zip_link, model_name], outputs=dl_output_message)

                gr.Markdown('## Примеры ввода')
                gr.Examples(
                    [
                        ['https://huggingface.co/phant0m4r/LiSA/resolve/main/LiSA.zip', 'Lisa'],
                        ['https://pixeldrain.com/u/3tJmABXA', 'Gura'],
                        ['https://huggingface.co/Kit-Lemonfoot/kitlemonfoot_rvc_models/resolve/main/AZKi%20(Hybrid).zip', 'Azki'],
                        ['https://drive.google.com/file/d/1btADHFCL-Xp40qquIrjTNR1QS66qnw_z/view?usp=sharing', 'Google Drive Model'],
                        ['https://mega.nz/file/abcd1234#key', 'Mega Model']
                    ],
                    [model_zip_link, model_name],
                    [],
                    download_online_model,
                )

            with gr.Tab('Из публичного индекса'):

                gr.Markdown('## Как использовать')
                gr.Markdown('- Нажмите "Инициализировать таблицу публичных моделей"')
                gr.Markdown('- Отфильтруйте модели по тегам или строке поиска')
                gr.Markdown('- Выберите строку для автоматического заполнения ссылки на загрузку и имени модели')
                gr.Markdown('- Нажмите "Скачать"')

                with gr.Row():
                    pub_zip_link = gr.Text(label='Ссылка на загрузку модели')
                    pub_model_name = gr.Text(label='Имя модели')

                with gr.Row():
                    download_pub_btn = gr.Button('Скачать 🌐', variant='primary', scale=19)
                    pub_dl_output_message = gr.Text(label='Выходное сообщение', interactive=False, scale=20)

                filter_tags = gr.CheckboxGroup(value=[], label='Показать голосовые модели с тегами', choices=[])
                search_query = gr.Text(label='Поиск')
                load_public_models_button = gr.Button(value='Инициализировать таблицу публичных моделей', variant='primary')

                public_models_table = gr.DataFrame(value=[], headers=['Имя модели', 'Описание', 'Кредит', 'URL', 'Теги'], label='Доступные публичные модели', interactive=False)
                public_models_table.select(pub_dl_autofill, inputs=[public_models_table], outputs=[pub_zip_link, pub_model_name])
                load_public_models_button.click(load_public_models, outputs=[public_models_table, filter_tags])
                search_query.change(filter_models, inputs=[filter_tags, search_query], outputs=public_models_table)
                filter_tags.change(filter_models, inputs=[filter_tags, search_query], outputs=public_models_table)
                download_pub_btn.click(download_online_model, inputs=[pub_zip_link, pub_model_name], outputs=pub_dl_output_message)

        with gr.Tab('Загрузить модель'):
            gr.Markdown('## Загрузить локально обученную модель RVC v2 и файл индекса')
            gr.Markdown('- Найдите файл модели (папка weights) и опциональный файл индекса (папка logs/[name])')
            gr.Markdown('- Сожмите файлы в zip архив')
            gr.Markdown('- Загрузите zip файл и дайте уникальное имя голосу')
            gr.Markdown('- Нажмите "Загрузить модель"')

            with gr.Row():
                with gr.Column():
                    zip_file = gr.File(label='Zip файл')

                local_model_name = gr.Text(label='Имя модели')

            with gr.Row():
                model_upload_button = gr.Button('Загрузить модель', variant='primary', scale=19)
                local_upload_output_message = gr.Text(label='Выходное сообщение', interactive=False, scale=20)
                model_upload_button.click(upload_local_model, inputs=[zip_file, local_model_name], outputs=local_upload_output_message)

        with gr.Tab('ИНФО'):
            gr.Markdown('## Информация об авторе')
            gr.Markdown('**🐣 ТЕЛЕГРАМ КАНАЛ:** https://t.me/aisingers')
            gr.Markdown('**👤 ЗАКАЗАТЬ МОДЕЛЬ НА ЗАКАЗ ТГ:** https://t.me/simbioz_2002')
            gr.Markdown('**🐣 YouTube Канал:** https://www.youtube.com/@DrawAvatarsTV')

    app.launch(
        share=args.share_enabled,
        enable_queue=True,
        server_name=None if not args.listen else (args.listen_host or '0.0.0.0'),
        server_port=args.listen_port or 7860,
    )
