'''Скрипт для тестирования модели на одном входном предложении.
Декодированные предсказания модели храняться в словаре response, где ключом служит наименование грамматического атрибута, а значением - список предсказанных значений.
Доступные грамматические атрибуты (ключи словаря):
[upos, head, deprel, Mood, NumType, VerbForm, ExtPos, Reflex, Polarity, Typo, NameType, InflClass,
       Person, Poss, Animacy, Degree, Foreign, Variant, Number, Gender, NumForm, Aspect, Case, PronType, Tense, Abbr, Voice]
'''

import torch
from model.model import MHAModel
from scripts.tokenizer import BPETokenizer, SeparatorTokenizer
from scripts.vectorizer import Vectorizer
from scripts.vocabulary import Vocabulary
import json
import os
import sys
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

default_target_names = ['upos', 'head', 'deprel', 'Mood', 'NumType', 'VerbForm',
       'ExtPos', 'Reflex', 'Polarity', 'Typo', 'NameType', 'InflClass',
       'Person', 'Poss', 'Animacy', 'Degree', 'Foreign', 'Variant', 'Number',
       'Gender', 'NumForm', 'Aspect', 'Case', 'PronType', 'Tense', 'Abbr', 'Voice']

load_dotenv(dotenv_path=(Path('.')/'.env'))

DATA_SAVE_FILEPATH = os.getenv('DATA_SAVE_FILEPATH')
EXPERIMENT_NAME=os.getenv('EXPERIMENT_NAME')
CHECKPOINTS_FILEPATH = os.path.join(DATA_SAVE_FILEPATH, EXPERIMENT_NAME, 'checkpoints')
DATA_INFO_FILEPATH = os.path.join(DATA_SAVE_FILEPATH, EXPERIMENT_NAME, 'data') # Место сохранения метрик валидации и обучения
MODEL_CHECKPOINT = os.path.join(CHECKPOINTS_FILEPATH, f'final_tokens_model_params.pt')

parser = argparse.ArgumentParser(description='Тестирование модели морфологического классификатора')
parser.add_argument(
    'user_query',
    type=str,
    help='Предложение для обработки'
)
parser.add_argument(
    '--morpheme',
    type=str,
    help='Строка с морфемами для вывода на экран, отделенные пробелом. Если не указано, то будет производиться анализ по всем морфемам',
    default='upos head deprel Mood NumType VerbForm ExtPos Reflex Polarity Typo NameType InflClass Person Poss Animacy Degree Foreign Variant Number Gender NumForm Aspect Case PronType Tense Abbr Voice',
)
parser.add_argument(
    '--device',
    choices=['cpu', 'cuda'],
    default='cuda',
    help='Устройство для инференса модели.'
)

args = parser.parse_args()
user_query = args.user_query
target_names = (args.morpheme).split()
device = args.device
device = device if torch.cuda.is_available() else 'cpu'

if len((set(default_target_names) | set(target_names))) != len(default_target_names):
    raise ValueError('Передана морфема, предсказание которой невозможно!')

# Чтение конфигурации словарей
with open(f'{DATA_INFO_FILEPATH}/merged_vocabs_configuration.json', 'r', encoding='utf-8') as file:
    vocabs_config = json.load(file)

# Чтение словарей из файла
source_vocab = Vocabulary.from_json(f'{DATA_INFO_FILEPATH}/merged_source_vocab.json')
with open(f'{DATA_INFO_FILEPATH}/merged_target_vocabs.json', 'r', encoding='utf-8') as file:
    target_vocabs_dict = json.load(file)
target_vocabs = {target_name: Vocabulary.from_serializable(target_vocabs_dict[target_name]) for target_name in target_vocabs_dict.keys()}

# Создание векторизатора
vectorizer = Vectorizer(source_vocab, None, None, 'tokens')

# Создание модели и токенизатора
model = MHAModel.from_pretrained(MODEL_CHECKPOINT).to(device)
tokenizer = BPETokenizer.from_pretrained(f'{CHECKPOINTS_FILEPATH}/tokenizer.json')

start_time = time.perf_counter()
# Токенизация и кодирование предложения
response = dict()
input_ids = list()
# Разбиение предложения на отдельные слова / знаки пунктуации. Правила разбиения максимально похожи на те, что используются в обучающем датасете
separated = SeparatorTokenizer.tokenize(user_query)
useful_len = len(separated) # полезная длина предложения (без учета паддинга)

# Токенизация и получение индексов токенов в том формате, который ожидает метод vectorize_tokens
for word in separated:
    tokenized = tokenizer.encode(word).tokens
    input_ids.append(tokenized)
input_ids = vectorizer.vectorize_tokens(vocabs_config['MAX_SUBTOKENS_COUNT'], vocabs_config['MAX_WORDS_COUNT'], input_ids)

# Преобразование в тензор и перенос на устройство
vectorized = torch.tensor(input_ids).unsqueeze(0).to(device)
model = model.to(device)
with torch.no_grad():
    predictions = model(tokens=vectorized, letters=None, apply_softmax=True, temperature=1)

# Получение предсказаний и декодирование индексов в токены
for key, value in predictions.items():
    _, pred_indices = predictions[key].max(dim=-1)
    response[key] = target_vocabs[key].get_tokens(pred_indices.squeeze(0)[:useful_len].tolist())

end_time = time.perf_counter()

# Вывод на экран
print(f'Время генерации на устройсте {device} = {end_time-start_time}')
print(f'Длина текущего предложения: {useful_len}')
print(f'Исходное предложение: {separated}')
for target_name in target_names:
    print(f'Грамматический атрибут {target_name}')
    for idx, mark in enumerate(response[target_name]):
        print(f'"{separated[idx]}" {mark}', end=' '*5)
    print('\n')