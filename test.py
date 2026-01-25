import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from scripts.custom_dataset import CustomDataset
from model.model import MHAModel
import pandas as pd
import json
import time
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import precision_recall_fscore_support

# Определение платформы запуска
if sys.platform == 'linux':
    load_dotenv(dotenv_path=(Path('.')/'.env.linux'))
elif sys.platform == 'win32':
    load_dotenv(dotenv_path=(Path('.')/'.env.win'))
else:
    raise ValueError('Ваша операционная система не поддерживается!')

# Переопределяем параметры логгирования для вывода сообщений уровня info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Определение пути датасетов
DATASETS_FOLDER_PATH = os.getenv('DATASETS_FOLDER_PATH')
SYNTAGRUS_VERSION = os.getenv('SYNTAGRUS_VERSION', '2.16') # Допустимые занчения: 2.3; 2.16 | В версии 2.3 меньше тренировочных примеров, по сравнению с 2.16. Точность на тестовой выборке практически не меняется
SYNTAGRUS_PATH = os.getenv('SYNTAGRUS_PATH')
SYNTAGRUS_TEXTS_PATH = os.getenv('SYNTAGRUS_TEXTS_PATH')

TAIGA_PATH = os.getenv('TAIGA_PATH')
TAIGA_TEXTS_PATH = os.getenv('TAIGA_TEXTS_PATH')

MERGED_PATH = os.path.join(DATASETS_FOLDER_PATH, 'sintagrus_taiga_merged')
MERGED_TEXTS_PATH = os.path.join(DATASETS_FOLDER_PATH, 'sintagrus_taiga_merged.txt')

DATASET_TO_PREPARE = 'merged' # taiga, syntagrus of merged

# Определение датасета для обучения
if DATASET_TO_PREPARE == 'syntagrus':
    DATASET_PATH = SYNTAGRUS_PATH
elif DATASET_TO_PREPARE == 'taiga':
    DATASET_PATH = TAIGA_PATH
elif DATASET_TO_PREPARE == 'merged':
    DATASET_PATH = MERGED_PATH
else:
    raise ValueError('Неверный параметр используемого датасета')


SHUFFLE = True
DROP_LAST = False

BATCH_SIZE = 1

WORD_REPRESENTATION = 'tokens' # tokens; letters; both  Уровень представления слова (токены, буквы, токены + буквы)

MODEL_SAVE_FILEPATH = f'checkpoints/final_{WORD_REPRESENTATION}_model_params.pt'

RANDOM_STATE = 42

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def generate_batches(dataset:CustomDataset, batch_size:int, shuffle:bool=True, drop_last:bool=True, device='cpu'):
    '''Создает батчи из датасета и переносит данные на девайс'''
    dataloader = DataLoader(dataset, batch_size, shuffle, drop_last=drop_last)
    for data_dict in dataloader:
        out_data_dict = {}
        for name, _ in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict


def save_results_to_file(model, model_filepath:str, train_states:list=None, validation_states:list=None):
    '''Сохраняет параметры модели и метрики обучения в файлы'''
    torch.save(model, model_filepath)
    logging.info('Параметры модели сохранены')
    if train_states is not None:
        with open(f"data/{WORD_REPRESENTATION}_train_states.json", "w", encoding="utf-8") as file:
            json.dump(train_states, file, indent=4, ensure_ascii=False)
            logging.info('Метрики обучения сохранены')
        
        with open(f"checkpoints/{WORD_REPRESENTATION}_model_hyperparams.json", "w", encoding="utf-8") as file:
            json.dump(model.get_hyperparams(), file, indent=4, ensure_ascii=False)
            logging.info('Гиперпараметры сохранены')

    if validation_states is not None:
        with open(f"data/{WORD_REPRESENTATION}_validation_states.json", "w", encoding="utf-8") as file:
            json.dump(validation_states, file, indent=4, ensure_ascii=False)
            logging.info('Метрики валидации сохранены')


def normalize_sizes(predictions:dict[str:torch.tensor], targets:dict[str:torch.tensor], target_names:list[str]):
    for key in target_names:
        # Для predictions: [B, S, C] -> [B*S, C]
        if len(predictions[key].size()) == 3:
            predictions[key] = predictions[key].contiguous().view(-1, predictions[key].size(-1))
        
        # Для targets: [B, S] -> [B*S]
        if len(targets[key].size()) == 2:
            targets[key] = targets[key].contiguous().view(-1)
    
    return predictions, targets


# def compute_loss(predictions:dict[str:torch.tensor], targets:dict[str:list[int]], target_names:list[str], target_weights:dict[str:float], pad_idx:int=0):
def compute_loss(predictions:dict[str:torch.tensor], targets:dict[str:list[int]], target_names:list[str], pad_idx:int=0):
    predictions, targets = normalize_sizes(predictions, targets, target_names)
    losses = {}
    total_loss = 0
    for key in target_names:
        # target_weights[key]['classes_weights'] = target_weights[key]['classes_weights'].to(DEVICE)
        # losses[key] = torch.nn.functional.cross_entropy(predictions[key], targets[key], weight=target_weights[key]['classes_weights'], ignore_index=pad_idx)
        losses[key] = torch.nn.functional.cross_entropy(predictions[key], targets[key], ignore_index=pad_idx)
        # total_loss += losses[key] * target_weights[key]['grammem_weight']
        total_loss += losses[key]

    return total_loss, losses


def compute_metrics(predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], target_names: list[str], pad_idx: int = 0, average: str = 'macro') -> dict:
    """Вычисляет метрики precision, recall, f1-score для каждой целевой переменной"""

    predictions, targets = normalize_sizes(predictions, targets, target_names)
    metrics_dict = {}
    
    for key in target_names:
        # Получаем предсказанные индексы классов
        _, pred_indices = predictions[key].max(dim=-1)
        
        pred_np = pred_indices.to('cpu').numpy()
        target_np = targets[key].to('cpu').numpy()
        
        # Создаем маску для игнорирования pad_idx
        mask = target_np != pad_idx
        
        # Фильтруем паддинг
        pred_filtered = pred_np[mask]
        target_filtered = target_np[mask]
        
        # Вычисляем метрики
        precision, recall, f1, support = precision_recall_fscore_support(target_filtered, pred_filtered, average=average, zero_division=0)
        
        # Вычисляем accuracy (для полноты)
        accuracy = (pred_filtered == target_filtered).mean()
        
        metrics_dict[key] = {
            'accuracy': accuracy,
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)}
    
    return metrics_dict


logging.info('Загрука датасетов...')
# Для валидации будем использовать датасет синтагрус. Для обучения и синтагрус и тайга
# train_df = pd.read_parquet(os.path.join(DATASET_PATH, 'prepared_train.parquet'))
train_df = None
validation_df = pd.read_parquet(os.path.join(DATASET_PATH, 'prepared_dev.parquet'))
# validation_df = pd.read_parquet(os.path.join('/mnt/12A4CA9DA4CA8329/Files/Datasets/UD_Russian-SynTagRus-master_2.16', 'prepared_dev.parquet'))
test_df = pd.read_parquet(os.path.join(DATASET_PATH, 'prepared_test.parquet'))


logging.info('Чтение конфигурации словаря...')
# Конфигурация словарей для определения модели
with open('data/merged_vocabs_configuration.json', 'r', encoding='utf-8') as file:
    vocabs_config = json.load(file)

MAX_WORDS_COUNT = vocabs_config['MAX_WORDS_COUNT']
MAX_SUBTOKENS_COUNT = vocabs_config['MAX_SUBTOKENS_COUNT']
MAX_LETTERS_COUNT = vocabs_config['MAX_LETTERS_COUNT']
PAD_IDX = vocabs_config['PAD_IDX']


target_names = ['upos', 'head', 'deprel', 'Mood', 'NumType', 'VerbForm',
       'ExtPos', 'Reflex', 'Polarity', 'Typo', 'NameType', 'InflClass',
       'Person', 'Poss', 'Animacy', 'Degree', 'Foreign', 'Variant', 'Number',
       'Gender', 'NumForm', 'Aspect', 'Case', 'PronType', 'Tense', 'Abbr', 'Voice']
source_name = 'source_text'

    
model = torch.load(MODEL_SAVE_FILEPATH, weights_only=False)

logging.info('Инициализация датасета...')
dataset = CustomDataset(train_df, target_names, MAX_SUBTOKENS_COUNT, MAX_WORDS_COUNT,\
                        MAX_LETTERS_COUNT, valid_df=validation_df, test_df=test_df)

logging.info('Перемещение модели на device')
model = model.to(device=DEVICE)

logging.info('Переход к основному циклу обучения и валидации...')
# Основной цикл обучения и валидации

dataset.set_dataframe_split('validation')
batch_generator = generate_batches(dataset, BATCH_SIZE, SHUFFLE, DROP_LAST, DEVICE)
epoch_sum_valid_loss = 0.0
epoch_running_valid_loss = 0.0
mean_generation_time = 0.0
# total_generation_time = 0.0
valid_epoch_metrics = {key:{'accuracy' : 0.0, 'precision' : 0.0, 'recall' : 0.0, 'f1' : 0.0, 'mean_loss' : 0.0} for key in target_names}
model.eval()

valid_start_time = time.time()
with torch.no_grad():
    for batch_idx, batch_dict in enumerate(batch_generator):
        
        start_generation_time = time.time()
        if WORD_REPRESENTATION == 'tokens':
            predictions = model(tokens=batch_dict['input_ids'], letters=None)
        elif WORD_REPRESENTATION == 'letters':
            predictions = model(tokens=None, letters=batch_dict['letters'])
        else:
            predictions = model(tokens=batch_dict['input_ids'], letters=batch_dict['letters'])
        end_generation_time = time.time()

        # total_loss, valid_losses = compute_loss(predictions, batch_dict, target_names, target_weights, PAD_IDX)
        total_loss, valid_losses = compute_loss(predictions, batch_dict, target_names, PAD_IDX)

        # Средние потери, точность и время генерации
        epoch_running_valid_loss += (total_loss.item() - epoch_running_valid_loss) / (batch_idx + 1)
        mean_generation_time += ((end_generation_time - start_generation_time) - mean_generation_time) / (batch_idx + 1)
        epoch_sum_valid_loss += total_loss.item()

        cur_metrics = compute_metrics(predictions, batch_dict, target_names, PAD_IDX)
        for key in target_names:
            for metric, value in cur_metrics[key].items():
                valid_epoch_metrics[key][metric] += (value - valid_epoch_metrics[key][metric]) / (batch_idx + 1)
            valid_epoch_metrics[key]['mean_loss'] += (valid_losses[key].item() - valid_epoch_metrics[key]['mean_loss']) / (batch_idx + 1)
valid_end_time = time.time()


print('-'*40)
print(f'Validation: Средняя ошибка {epoch_running_valid_loss}')
for key in target_names:
    print('-'*20)
    print(f'Validation: Ошибка на признаке {key}: {valid_epoch_metrics[key]['mean_loss']}')
    print(f'Validation: Точность на признаке {key}: {valid_epoch_metrics[key]['accuracy']*100}%')
    print(f'Validation: precision на признаке {key}: {valid_epoch_metrics[key]['precision']*100}%')
    print(f'Validation: recall на признаке {key}: {valid_epoch_metrics[key]['recall']*100}%')
    print(f'Validation: f1-score на признаке {key}: {valid_epoch_metrics[key]['f1']*100}%')
print(f'Среднее время генерации при размере батча {BATCH_SIZE}: {mean_generation_time}')
print(f'Время выполнения всего цикла тестирования: {valid_end_time - valid_start_time}')