import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from scripts.custom_dataset import CustomDataset
from model.model import MHAModel
import pandas as pd
import json
import time
import os
import sys
import logging
import argparse
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

WORD_REPRESENTATION = 'tokens' # tokens; letters; both  Уровень представления слова (токены, буквы, токены + буквы)
WORDS_POS_ENCODING = 'learnable' # Допустимые значения: sin; learnable; None
WORD_SUBTOKENS_POS_ENCODING = 'rope' # Допустимые значения: learnable; rope; None
LETTERS_POS_ENCODING = 'learnable' # Допустимые значения: learnable; sin; None. Работоспособность при rope не проверялась

# Определение пути датасетов
DATASETS_FOLDER_PATH = os.getenv('DATASETS_FOLDER_PATH')
SYNTAGRUS_VERSION = os.getenv('SYNTAGRUS_VERSION', '2.16') # Допустимые занчения: 2.3; 2.16 | В версии 2.3 меньше тренировочных примеров, по сравнению с 2.16. Точность на тестовой выборке практически не меняется
DATA_SAVE_FILEPATH = os.getenv('DATA_SAVE_FILEPATH')

EXPERIMENT_NAME=os.getenv('EXPERIMENT_NAME')
CHECKPOINTS_FILEPATH = os.path.join(DATA_SAVE_FILEPATH, EXPERIMENT_NAME, 'checkpoints')
DATA_INFO_FILEPATH = os.path.join(DATA_SAVE_FILEPATH, EXPERIMENT_NAME, 'data')
DATASET_SAVE_FILEPATH = os.path.join(DATA_SAVE_FILEPATH, EXPERIMENT_NAME, 'dataset') # Путь для сохранения подготовленного датасета. Подготовленный датасет включает в себя столбцы с индексами входов и выходов


MODEL_SAVE_FILEPATH = os.path.join(CHECKPOINTS_FILEPATH, f'final_{WORD_REPRESENTATION}_model_params.pt')

Path.mkdir(Path(DATA_SAVE_FILEPATH, EXPERIMENT_NAME), exist_ok=True)
Path.mkdir(Path(DATA_SAVE_FILEPATH, EXPERIMENT_NAME, 'data'), exist_ok=True)
Path.mkdir(Path(DATA_SAVE_FILEPATH, EXPERIMENT_NAME, 'checkpoints'), exist_ok=True)
Path.mkdir(Path(DATASET_SAVE_FILEPATH), exist_ok=True)
logging.info('Пути для сохранения файлов созданы')

DATASET_TO_PREPARE = 'merged' # taiga, syntagrus or merged

# Парсинг аргумента командной строки
parser = argparse.ArgumentParser(description='Обучение модели морфологического классификатора')
parser.add_argument(
    '--dataset',
    type=str,
    default='merged',
    choices=['taiga', 'syntagrus', 'merged'],
    help='Выбор датасета для обучения: taiga, syntagrus или merged (слияние taiga и syntagrus)',
    required=True,
)
parser.add_argument(
    '--pretrained',
    action='store_true',
    help='Использование предобученной модели или обучение новой.',
)
parser.add_argument(
    '--batch',
    type=int,
    default=96,
    help='Выбор размера батча для обучения.'
)
parser.add_argument(
    '--device',
    choices=['cpu', 'cuda'],
    default='cuda',
    help='Устройство для инференса модели.'
)
parser.add_argument(
    '--checkpoint_epoch',
    type=int,
    default=2,
    help='Сохранение параметров модели и метрик обучения каждые checkpoint_epoch эпох'
)

# Параметры обучения модели
USE_CLASSES_WEIGHTS = False
CLASSES_WEIGHTS_SCALER = 12

SHUFFLE = True
DROP_LAST = True
EPOCHS = 35
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

INIT_WEIGHTS = True
BIAS = True
TOKENS_EMBEDDING_DIM = 512
LETTERS_EMBEDDING_DIM = 32 # Важно, если WORD_REPRESENTATION = 'both', то ATTENTION_DIM = (LETTERS_EMBEDDING_DIM * MAX_LETTERS_COUNT) + TOKENS_EMBEDDING_DIM
LETTERS_IN_WORD_ATTENTION_DIM = 128
MAIN_ATTENTION_DIM = 512
MAIN_NUM_HEADS = 8
MAIN_NUM_ENCODER_LAYERS = 4
MAIN_ENCODER_FC_HIDDEN_DIM = MAIN_ATTENTION_DIM*4 # Как в классическом трансформере

CLASSIFIER_FC_HIDDEN_DIM = MAIN_ATTENTION_DIM*4

ROPE_BASE = 10000

DROPOUT = 0.25
TEMPERATURE = 1
BATCH_FIRST = True

RANDOM_STATE = 42

args = parser.parse_args()
DATASET_TO_PREPARE = args.dataset
BATCH_SIZE = args.batch
USE_PRETRAINED = True if args.pretrained else False
CHECKPOINT_EPOCH = args.checkpoint_epoch
DEVICE = args.device
DEVICE = DEVICE if torch.cuda.is_available() else 'cpu'
logging.info(f'''Текущие параметры обработки датасета и конфигурация токенизатора:
             DATASET_TO_PREPARE: {DATASET_TO_PREPARE}
             BATCH_SIZE: {BATCH_SIZE}
             CHECKPOINT_EPOCH: {CHECKPOINT_EPOCH}
             DEVICE: {DEVICE}
             USE_PRETRAINED: {USE_PRETRAINED}''')

# print(torch.backends.cuda.flash_sdp_enabled())
# print(torch.backends.cuda.mem_efficient_sdp_enabled())
# print(torch.backends.cuda.math_sdp_enabled())


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
        with open(os.path.join(DATA_INFO_FILEPATH, f"{WORD_REPRESENTATION}_train_states.json"), "w", encoding="utf-8") as file:
            json.dump(train_states, file, indent=4, ensure_ascii=False)
            logging.info('Метрики обучения сохранены')
        
        with open(os.path.join(CHECKPOINTS_FILEPATH, f"{WORD_REPRESENTATION}_model_hyperparams.json"), "w", encoding="utf-8") as file:
            json.dump(model.get_hyperparams(), file, indent=4, ensure_ascii=False)
            logging.info('Гиперпараметры сохранены')

    if validation_states is not None:
        with open(os.path.join(DATA_INFO_FILEPATH, f"{WORD_REPRESENTATION}_validation_states.json"), "w", encoding="utf-8") as file:
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
train_df = pd.read_parquet(os.path.join(DATASET_SAVE_FILEPATH, f'{DATASET_TO_PREPARE}_prepared_train.parquet'))
# validation_df = pd.read_parquet(os.path.join(DATASET_SAVE_FILEPATH, f'{DATASET_TO_PREPARE}_prepared_dev.parquet'))
validation_df = pd.read_parquet(os.path.join(DATASET_SAVE_FILEPATH, f'syntagrus_prepared_dev.parquet')) # Для валидации используем только syntagrus
test_df = pd.read_parquet(os.path.join(DATASET_SAVE_FILEPATH, f'{DATASET_TO_PREPARE}_prepared_test.parquet'))


logging.info('Чтение конфигурации словаря...')
# Конфигурация словарей для определения модели
with open(f'{DATA_INFO_FILEPATH}/merged_vocabs_configuration.json', 'r', encoding='utf-8') as file:
    vocabs_config = json.load(file)

MAX_WORDS_COUNT = vocabs_config['MAX_WORDS_COUNT']
MAX_SUBTOKENS_COUNT = vocabs_config['MAX_SUBTOKENS_COUNT']
MAX_LETTERS_COUNT = vocabs_config['MAX_LETTERS_COUNT']
SOURCE_VOCAB_LEN = vocabs_config['SOURCE_VOCAB_LEN']
LETTERS_VOCAB_LEN = vocabs_config['LETTERS_VOCAB_LEN']
TRG_VOCABS_LEN = vocabs_config['TRG_VOCABS_LEN']
PAD_IDX = vocabs_config['PAD_IDX']


target_names = ['upos', 'head', 'deprel', 'Mood', 'NumType', 'VerbForm',
       'ExtPos', 'Reflex', 'Polarity', 'Typo', 'NameType', 'InflClass',
       'Person', 'Poss', 'Animacy', 'Degree', 'Foreign', 'Variant', 'Number',
       'Gender', 'NumForm', 'Aspect', 'Case', 'PronType', 'Tense', 'Abbr', 'Voice']
source_name = 'source_text'


if USE_PRETRAINED:
    logging.info('Загрузка предобученной модели и предыдущих метрик обучения...')
    with open(f"{DATA_INFO_FILEPATH}/{WORD_REPRESENTATION}_train_states.json", "r", encoding="utf-8") as file:
        train_states = json.load(file)
        training_epochs = int(train_states[-1]['training_epochs'])

    with open(f"{DATA_INFO_FILEPATH}/{WORD_REPRESENTATION}_validation_states.json", "r", encoding="utf-8") as file:
        validation_states = json.load(file)
    
    model = torch.load(MODEL_SAVE_FILEPATH, weights_only=False)
else:
    logging.info('Инициализация модели с нуля...')
    train_states = []
    validation_states = []
    training_epochs = 0
    model = MHAModel(MAX_WORDS_COUNT, MAX_SUBTOKENS_COUNT, MAX_LETTERS_COUNT, LETTERS_VOCAB_LEN, SOURCE_VOCAB_LEN, TOKENS_EMBEDDING_DIM, LETTERS_EMBEDDING_DIM,\
                     MAIN_ATTENTION_DIM, MAIN_NUM_HEADS, MAIN_NUM_ENCODER_LAYERS, CLASSIFIER_FC_HIDDEN_DIM, MAIN_ENCODER_FC_HIDDEN_DIM,\
                     TRG_VOCABS_LEN, WORDS_POS_ENCODING, WORD_SUBTOKENS_POS_ENCODING, LETTERS_POS_ENCODING, ROPE_BASE,\
                     LETTERS_IN_WORD_ATTENTION_DIM, DROPOUT, TEMPERATURE, BATCH_FIRST, WORD_REPRESENTATION, INIT_WEIGHTS, BIAS, PAD_IDX, DEVICE)

logging.info('Инициализация датасета...')
dataset = CustomDataset(train_df, target_names, MAX_SUBTOKENS_COUNT, MAX_WORDS_COUNT,\
                        MAX_LETTERS_COUNT, valid_df=validation_df, test_df=test_df)

logging.info('Перемещение модели на device...')
model = model.to(device=DEVICE)
optimizer = optim.AdamW(model.parameters(), LEARNING_RATE, weight_decay=WEIGHT_DECAY)

logging.info('Переход к основному циклу обучения и валидации...')
try:
    for epoch in range(1, EPOCHS+1):
        train_start_time = time.time()
        training_epochs += 1
        print('='*50)
        print(f'Epoch {training_epochs}')
        dataset.set_dataframe_split('train')
        batch_generator = generate_batches(dataset, BATCH_SIZE, SHUFFLE, DROP_LAST, DEVICE)
        epoch_sum_train_loss = 0.0
        epoch_running_train_loss = 0.0
        train_epoch_metrics = {key:{'accuracy' : 0.0, 'precision' : 0.0, 'recall' : 0.0, 'f1' : 0.0, 'mean_loss' : 0.0} for key in target_names}
        model.train()
        for batch_idx, batch_dict in enumerate(batch_generator):

            optimizer.zero_grad()
            
            if WORD_REPRESENTATION == 'tokens':
                predictions = model(tokens=batch_dict['input_ids'], letters=None)
            elif WORD_REPRESENTATION == 'letters':
                predictions = model(tokens=None, letters=batch_dict['letters'])
            else:
                predictions = model(tokens=batch_dict['input_ids'], letters=batch_dict['letters'])

            # total_loss, train_losses = compute_loss(predictions, batch_dict, target_names, target_weights, PAD_IDX)
            total_loss, train_losses = compute_loss(predictions, batch_dict, target_names, PAD_IDX)

            total_loss.backward()

            optimizer.step()

            # Метрики
            epoch_running_train_loss += (total_loss.item() - epoch_running_train_loss) / (batch_idx + 1)
            epoch_sum_train_loss += total_loss.item()

            cur_metrics = compute_metrics(predictions, batch_dict, target_names, PAD_IDX)
            for key in target_names:
                for metric, value in cur_metrics[key].items():
                    train_epoch_metrics[key][metric] += (value - train_epoch_metrics[key][metric]) / (batch_idx + 1)
                train_epoch_metrics[key]['mean_loss'] += (train_losses[key].item() - train_epoch_metrics[key]['mean_loss']) / (batch_idx + 1)
        train_end_time = time.time()

        dataset.set_dataframe_split('validation')
        batch_generator = generate_batches(dataset, BATCH_SIZE, SHUFFLE, DROP_LAST, DEVICE)
        epoch_sum_valid_loss = 0.0
        epoch_running_valid_loss = 0.0
        valid_epoch_metrics = {key:{'accuracy' : 0.0, 'precision' : 0.0, 'recall' : 0.0, 'f1' : 0.0, 'mean_loss' : 0.0} for key in target_names}
        model.eval()
        valid_start_time = time.time()

        with torch.no_grad():
            for batch_idx, batch_dict in enumerate(batch_generator):
                
                if WORD_REPRESENTATION == 'tokens':
                    predictions = model(tokens=batch_dict['input_ids'], letters=None)
                elif WORD_REPRESENTATION == 'letters':
                    predictions = model(tokens=None, letters=batch_dict['letters'])
                else:
                    predictions = model(tokens=batch_dict['input_ids'], letters=batch_dict['letters'])

                # total_loss, valid_losses = compute_loss(predictions, batch_dict, target_names, target_weights, PAD_IDX)
                total_loss, valid_losses = compute_loss(predictions, batch_dict, target_names, PAD_IDX)

                # Средние потери и точность
                epoch_running_valid_loss += (total_loss.item() - epoch_running_valid_loss) / (batch_idx + 1)
                epoch_sum_valid_loss += total_loss.item()

                cur_metrics = compute_metrics(predictions, batch_dict, target_names, PAD_IDX)
                for key in target_names:
                    for metric, value in cur_metrics[key].items():
                        valid_epoch_metrics[key][metric] += (value - valid_epoch_metrics[key][metric]) / (batch_idx + 1)
                    valid_epoch_metrics[key]['mean_loss'] += (valid_losses[key].item() - valid_epoch_metrics[key]['mean_loss']) / (batch_idx + 1)
        valid_end_time = time.time()

        train_states.append(train_epoch_metrics)
        train_states[-1]['summed loss'] = epoch_sum_train_loss
        train_states[-1]['training_epochs'] = training_epochs
        train_states[-1]['execution_time'] = train_end_time - train_start_time

        validation_states.append(valid_epoch_metrics)
        validation_states[-1]['summed loss'] = epoch_sum_valid_loss
        validation_states[-1]['training_epochs'] = training_epochs
        validation_states[-1]['execution_time'] = valid_end_time - valid_start_time
        
        print(f'Train: Средняя ошибка эпохи {epoch_running_train_loss}')
        for key in target_names:
            print('-'*20)
            print(f'Train: Ошибка на признаке {key}: {train_epoch_metrics[key]['mean_loss']}')
            print(f'Train: Точность на признаке {key}: {train_epoch_metrics[key]['accuracy']*100}%')
            print(f'Train: precision на признаке {key}: {train_epoch_metrics[key]['precision']*100}%')
            print(f'Train: recall на признаке {key}: {train_epoch_metrics[key]['recall']*100}%')
            print(f'Train: f1-score на признаке {key}: {train_epoch_metrics[key]['f1']*100}%')
        print(f'Время выполнения {train_end_time - train_start_time}')

        print('-'*40)
        print(f'Validation: Средняя ошибка эпохи {epoch_running_valid_loss}')
        for key in target_names:
            print('-'*20)
            print(f'Validation: Ошибка на признаке {key}: {valid_epoch_metrics[key]['mean_loss']}')
            print(f'Validation: Точность на признаке {key}: {valid_epoch_metrics[key]['accuracy']*100}%')
            print(f'Validation: precision на признаке {key}: {valid_epoch_metrics[key]['precision']*100}%')
            print(f'Validation: recall на признаке {key}: {valid_epoch_metrics[key]['recall']*100}%')
            print(f'Validation: f1-score на признаке {key}: {valid_epoch_metrics[key]['f1']*100}%')
        print(f'Время выполнения {valid_end_time - valid_start_time}')

        # Блок с сохранением результатов обучения и изменением learning rate
        if epoch % CHECKPOINT_EPOCH == 0:
            logging.info('Сохранение результатов обучения...')
            save_results_to_file(model, os.path.join(CHECKPOINTS_FILEPATH, f'iter_{epoch}_{WORD_REPRESENTATION}_model_params.pt'),\
                                 train_states, validation_states)
            torch.save(model, MODEL_SAVE_FILEPATH)
        # Реализация нелинейного расписания изменения learning rate. Такое расписание показывает наискорейшую сходимость согласно наблюдениям
        if epoch == 15:
            LEARNING_RATE = 5e-5
            logging.info(f'Изменение скорости обучения на эпохе {epoch}. Новая скорость обучения: {LEARNING_RATE}')
            optimizer = optim.AdamW(model.parameters(), 5e-5, weight_decay=WEIGHT_DECAY)
        if epoch == 25:
            LEARNING_RATE = 1e-5
            logging.info(f'Изменение скорости обучения на эпохе {epoch}. Новая скорость обучения: {LEARNING_RATE}')
            optimizer = optim.AdamW(model.parameters(), LEARNING_RATE, weight_decay=WEIGHT_DECAY)

except KeyboardInterrupt:
    print('Принудительная остановка')

save_results_to_file(model, MODEL_SAVE_FILEPATH, train_states, validation_states)