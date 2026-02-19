'''Скрипт для тестирования метрик модели на различных датасетах (sintagrus, taiga or merged)'''

import torch
from torch.utils.data import DataLoader
from scripts.custom_dataset import CustomDataset
import pandas as pd
import json
import time
import os
import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import precision_recall_fscore_support

load_dotenv(dotenv_path=(Path('.')/'.env'))

# Переопределяем параметры логгирования для вывода сообщений уровня info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Определение пути датасетов
DATASETS_FOLDER_PATH = os.getenv('DATASETS_FOLDER_PATH')

DATA_SAVE_FILEPATH = os.getenv('DATA_SAVE_FILEPATH')
EXPERIMENT_NAME=os.getenv('EXPERIMENT_NAME')

CHECKPOINTS_FILEPATH = os.path.join(DATA_SAVE_FILEPATH, EXPERIMENT_NAME, 'checkpoints')
DATA_INFO_FILEPATH = os.path.join(DATA_SAVE_FILEPATH, EXPERIMENT_NAME, 'data')
DATASETS_FOLDER_PATH = os.path.join(DATA_SAVE_FILEPATH, EXPERIMENT_NAME, 'dataset')

# Парсинг аргументов командной строки
parser = argparse.ArgumentParser(description='Тестирование модели на различных датасетах')
parser.add_argument(
    '--dataset',
    type=str,
    default='merged',
    choices=['taiga', 'syntagrus', 'merged'],
    help='Какой датасет подготовить: taiga, syntagrus или merged',
    required=True,
)
parser.add_argument(
    '--split',
    type=str,
    default='validation',
    choices=['validation', 'test'],
    help='Выбор сплита датасета для теста',
    required=True,
)
parser.add_argument(
    '--batch',
    type=int,
    default=64,
    help='Размера батча при тестировании.'
)
parser.add_argument(
    '--device',
    choices=['cpu', 'cuda'],
    default='cuda',
    help='Устройство для инференса модели.'
)
args = parser.parse_args()

DATASET_TO_PREPARE = args.dataset
BATCH_SIZE = args.batch
SPLIT = args.split
DEVICE = args.device
DEVICE = DEVICE if torch.cuda.is_available() else 'cpu'
logging.info(f'''Текущие параметры тестирования:
             DATASET_TO_PREPARE: {DATASET_TO_PREPARE}
             BATCH_SIZE: {BATCH_SIZE}
             SPLIT: {SPLIT}
             DEVICE: {DEVICE}''')

SHUFFLE = True
DROP_LAST = False

WORD_REPRESENTATION = 'tokens' # tokens; letters; both  Уровень представления слова (токены, буквы, токены + буквы)

MODEL_SAVE_FILEPATH = f'{CHECKPOINTS_FILEPATH}/final_{WORD_REPRESENTATION}_model_params.pt'

RANDOM_STATE = 42


def generate_batches(dataset:CustomDataset, batch_size:int, shuffle:bool=True, drop_last:bool=True, device='cpu'):
    '''Создает батчи из датасета и переносит данные на девайс'''
    dataloader = DataLoader(dataset, batch_size, shuffle, drop_last=drop_last)
    for data_dict in dataloader:
        out_data_dict = {}
        for name, _ in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict


def normalize_sizes(predictions:dict[str:torch.tensor], targets:dict[str:torch.tensor], target_names:list[str]):
    for key in target_names:
        # Для predictions: [B, S, C] -> [B*S, C]
        if len(predictions[key].size()) == 3:
            predictions[key] = predictions[key].contiguous().view(-1, predictions[key].size(-1))
        
        # Для targets: [B, S] -> [B*S]
        if len(targets[key].size()) == 2:
            targets[key] = targets[key].contiguous().view(-1)
    
    return predictions, targets


def compute_loss(predictions:dict[str:torch.tensor], targets:dict[str:list[int]], target_names:list[str], pad_idx:int=0):
    predictions, targets = normalize_sizes(predictions, targets, target_names)
    losses = {}
    total_loss = 0
    for key in target_names:
        losses[key] = torch.nn.functional.cross_entropy(predictions[key], targets[key], ignore_index=pad_idx)
        total_loss += losses[key]

    return total_loss, losses


def compute_metrics(predictions, targets, target_names, pad_idx=0, average='macro'):
    metrics_dict = {}
    for key in target_names:

        targets[key] = targets[key].reshape(BATCH_SIZE, -1)
        predictions[key] = predictions[key].reshape(*targets[key].size(), -1)

        # predictions[key]: [B, S, C]; targets[key]: [B, S]
        _, pred_indices = predictions[key].max(dim=-1)  # [B, S]
        
        # Маска значимых токенов
        mask = targets[key] != pad_idx  # [B, S]

        errors_per_sentence = ((pred_indices != targets[key]) & mask).sum(dim=1)  # [B]
        sentence_correct = errors_per_sentence == 0  # [B]
        sentence_accuracy = sentence_correct.float().mean().item()
        
        # Фильтруем паддинг
        pred_filtered = pred_indices[mask].cpu().numpy()
        target_filtered = targets[key][mask].cpu().numpy()

        precision, recall, f1, _ = precision_recall_fscore_support(
            target_filtered, pred_filtered, average=average, zero_division=0)
        
        # Общая точность (accuracy) по токенам
        # token_accuracy = (pred_indices[mask] == targets[key][mask]).float().mean().item()
        accuracy = (pred_filtered == target_filtered).mean()
        
        metrics_dict[key] = {
            'accuracy': accuracy,
            'sentence_accuracy': sentence_accuracy,
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }
    return metrics_dict


def save_results_to_file(validation_states:list=None):
    '''Сохраняет метрики в файл'''
    if validation_states is not None:
        with open(os.path.join(DATA_INFO_FILEPATH, f"{WORD_REPRESENTATION}_validation_states.json"), "w", encoding="utf-8") as file:
            json.dump(validation_states, file, indent=4, ensure_ascii=False)
            logging.info('Метрики валидации сохранены')

logging.info('Загрука датасетов...')
# Для валидации будем использовать датасет синтагрус. Для обучения и синтагрус и тайга
train_df = None
validation_df = pd.read_parquet(os.path.join(DATASETS_FOLDER_PATH, f'{DATASET_TO_PREPARE}_prepared_dev.parquet'))
test_df = pd.read_parquet(os.path.join(DATASETS_FOLDER_PATH, f'{DATASET_TO_PREPARE}_prepared_test.parquet'))


logging.info('Чтение конфигурации словаря...')
with open(f'{DATA_INFO_FILEPATH}/merged_vocabs_configuration.json', 'r', encoding='utf-8') as file:
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


model = torch.load(MODEL_SAVE_FILEPATH, weights_only=False, map_location=torch.device(DEVICE))

logging.info('Инициализация датасета...')
dataset = CustomDataset(train_df, target_names, MAX_SUBTOKENS_COUNT, MAX_WORDS_COUNT,\
                        MAX_LETTERS_COUNT, valid_df=validation_df, test_df=test_df)

logging.info('Перемещение модели на device')
model = model.to(device=DEVICE)

logging.info('Переход к валидации...')
# Основной цикл обучения и валидации

# С помощью этой функции выбирается набор данных для тестирования. Доступные значения train, validation, test
dataset.set_dataframe_split(SPLIT)
batch_generator = generate_batches(dataset, BATCH_SIZE, SHUFFLE, DROP_LAST, DEVICE)
epoch_sum_valid_loss = 0.0
epoch_running_valid_loss = 0.0
mean_generation_time = 0.0
valid_epoch_metrics = {key:{'accuracy' : 0.0, 'sentence_accuracy' : 0.0, 'precision' : 0.0, 
                            'recall' : 0.0, 'f1' : 0.0, 'mean_loss' : 0.0} for key in target_names}
validation_states = []
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

validation_states.append(valid_epoch_metrics)
validation_states[-1]['summed loss'] = epoch_sum_valid_loss
validation_states[-1]['execution_time'] = valid_end_time - valid_start_time

print('-'*40)
print(f'Validation: Средняя ошибка {epoch_running_valid_loss}')
for key in target_names:
    print('-'*20)
    print(f'Validation: Ошибка на признаке {key}: {valid_epoch_metrics[key]['mean_loss']}')
    print(f'Validation: Точность на признаке {key}: {valid_epoch_metrics[key]['accuracy']*100}%')
    print(f'Validation: Точность предложения на признаке {key}: {valid_epoch_metrics[key]['sentence_accuracy']*100}%')
    print(f'Validation: precision на признаке {key}: {valid_epoch_metrics[key]['precision']*100}%')
    print(f'Validation: recall на признаке {key}: {valid_epoch_metrics[key]['recall']*100}%')
    print(f'Validation: f1-score на признаке {key}: {valid_epoch_metrics[key]['f1']*100}%')
print(f'Среднее время генерации при размере батча {BATCH_SIZE}: {mean_generation_time}')
print(f'Время выполнения всего цикла тестирования: {valid_end_time - valid_start_time}')

save_results_to_file(validation_states)
logging.info('Готово!')