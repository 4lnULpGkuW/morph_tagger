from scripts.tokenizer import BPETokenizer
from scripts.vectorizer import Vectorizer
from scripts.vocabulary import Vocabulary
import pandas as pd
import numpy as np
import json
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Переопределяем параметры логгирования для вывода сообщений уровня info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

load_dotenv(dotenv_path=(Path('.')/'.env'))

# Пути для сохранения распакованных из формата conllu датасетов, не поддверженных модификации текущим скриптом
DATASETS_FOLDER_PATH = os.getenv('DATASETS_FOLDER_PATH')
SYNTAGRUS_VERSION = os.getenv('SYNTAGRUS_VERSION', '2.16') # Допустимые занчения: 2.3; 2.16 | В версии 2.3 меньше тренировочных примеров, по сравнению с 2.16. Точность на тестовой выборке практически не меняется
SYNTAGRUS_PATH = os.getenv('SYNTAGRUS_PATH')
SYNTAGRUS_TEXTS_PATH = os.getenv('SYNTAGRUS_TEXTS_PATH')

TAIGA_PATH = os.getenv('TAIGA_PATH')
TAIGA_TEXTS_PATH = os.getenv('TAIGA_TEXTS_PATH')

MERGED_PATH = os.path.join(DATASETS_FOLDER_PATH, 'sintagrus_taiga_merged')
Path.mkdir(Path(MERGED_PATH), exist_ok=True)
MERGED_TEXTS_PATH = os.path.join(MERGED_PATH, 'sintagrus_taiga_merged.txt')

EXPERIMENT_NAME=os.getenv('EXPERIMENT_NAME')
DATA_SAVE_FILEPATH = os.getenv('DATA_SAVE_FILEPATH')
CHECKPOINTS_FILEPATH = os.path.join(DATA_SAVE_FILEPATH, EXPERIMENT_NAME, 'checkpoints')
DATA_INFO_FILEPATH = os.path.join(DATA_SAVE_FILEPATH, EXPERIMENT_NAME, 'data')
DATASET_SAVE_FILEPATH = os.path.join(DATA_SAVE_FILEPATH, EXPERIMENT_NAME, 'dataset') # Путь для сохранения подготовленного датасета. Подготовленный датасет включает в себя столбцы с индексами входов и выходов

# Парсинг аргументов командной строки
parser = argparse.ArgumentParser(description='Подготовка датасетов Syntagrus, Taiga или их слияния')
parser.add_argument(
    '--dataset',
    type=str,
    default='merged',
    choices=['taiga', 'syntagrus', 'merged'],
    help='Какой датасет подготовить: taiga, syntagrus или merged',
    required=True,
)
parser.add_argument(
    '--pretrained',
    action='store_true',
    help='Использование предобученного токенизатора или обучение нового. ',
)
parser.add_argument(
    '--mfp',
    type=int,
    default=1000,
    help='Выбор минимальной частоты встречаемости соседних символов для их слияния в один токен при обучении токенизатора. ' \
    'Например, при MFP = 200, символы не будут обьеденены в токен, если встретились по соседству менее 200 раз.'
)
parser.add_argument(
    '--exclude_unused_grammemes',
    action='store_true',
    help='Определяет, исключать ли граммемы, не принадлежащие слову.' \
    'По умолчанию, если слову гарммема не принадлежит, то целевая метка для данной граммемы - None. Если граммемы исключать, то целевая метка - padding',
)
args = parser.parse_args()

DATASET_TO_PREPARE = args.dataset
USE_PRETRAINDED_TOKENIZER = True if args.pretrained else False
MIN_FRECQUENCY_PAIR = args.mfp
EXCLUDE_UNUSED_GRAMMEMES = True if args.exclude_unused_grammemes else False
logging.info(f'''Текущие параметры обработки датасета и конфигурация токенизатора:
             DATASET_TO_PREPARE: {DATASET_TO_PREPARE}
             USE_PRETRAINDED_TOKENIZER: {USE_PRETRAINDED_TOKENIZER}
             MIN_FRECQUENCY_PAIR: {MIN_FRECQUENCY_PAIR}''')

Path.mkdir(Path(DATA_SAVE_FILEPATH, EXPERIMENT_NAME), exist_ok=True)
Path.mkdir(Path(CHECKPOINTS_FILEPATH), exist_ok=True)
Path.mkdir(Path(DATA_INFO_FILEPATH), exist_ok=True)
Path.mkdir(Path(DATASET_SAVE_FILEPATH), exist_ok=True)
logging.info('Пути для сохранения файлов созданы')

# Определение датасетов для подготовки
if DATASET_TO_PREPARE == 'syntagrus':
    DATASET_PATH = SYNTAGRUS_PATH
    CORPUS_TEXTS_PATH = SYNTAGRUS_TEXTS_PATH
elif DATASET_TO_PREPARE == 'taiga':
    DATASET_PATH = TAIGA_PATH
    CORPUS_TEXTS_PATH = TAIGA_TEXTS_PATH
elif DATASET_TO_PREPARE == 'merged':
    DATASET_PATH = MERGED_PATH
    CORPUS_TEXTS_PATH = MERGED_TEXTS_PATH
else:
    raise ValueError('Неверный параметр используемого датасета')

UNK_TOKEN = '<UNK>'
MASK_TOKEN = '<MASK>'
PAD_TOKEN = '<PAD>'
BOS_TOKEN='<BOS>'
EOS_TOKEN = '<EOS>'
ADD_BOS_EOS_TOKENS = False

WORD_REPRESENTATION = 'tokens' # tokens; letters; both  Уровень представления слова (токены, буквы, токены + буквы)

VOCABULARY_SIZE = 10000

MAX_WORDS_COUNT = 0
MAX_SUBTOKENS_COUNT = 0
MAX_LETTERS_COUNT = 12

def find_max_words_source_len(dataframe:pd.DataFrame)->int:
    '''Возвращает максимальную длину НЕ ТОКЕНИЗИРОВАННОЙ входной последовательности в датафрейме'''
    max_words_tokens = 0
    for i in range(len(dataframe)):
        max_words_tokens = max(len(dataframe.loc[i, 'source_words']), max_words_tokens)
    return max_words_tokens

def find_max_tokens_source_len(dataframe:pd.DataFrame, tokenizer)->int:
    '''Возвращает максимальную длину ТОКЕНИЗИРОВАННОЙ входной последовательности в датафрейме'''
    max_source_tokens = 0
    for i in range(len(dataframe)):
        max_source_tokens = max(len(tokenizer.encode(dataframe.loc[i, 'source_text']).tokens), max_source_tokens)
    return max_source_tokens

def find_max_subtokens_cnt(dataframe:pd.DataFrame)->int:
    '''Возвращает максимальное количество субтокенов слов в датафрейме'''
    max_subtokens_cnt = 0
    for row in range(len(dataframe)):
        for token_lst in dataframe.loc[row, 'tokens']:
            max_subtokens_cnt =  max(max_subtokens_cnt, len(token_lst))
    return max_subtokens_cnt

def find_max_letters_cnt(dataframe:pd.DataFrame)->int:
    '''Возвращает максимальное количество букв слов в датафрейме'''
    max_letters_cnt = 0
    for i in range(len(dataframe)):
        row_text_list = dataframe.loc[i, 'source_words']
        for word in row_text_list:
            max_letters_cnt = max(max_letters_cnt, len(word))
    return max_letters_cnt

def get_subtokens_cnt(dataframe:pd.DataFrame):
    subtokens_cnt = []
    subtokens_dict = {}
    for i in range(len(dataframe)):
        row = dataframe.loc[i, 'tokens']
        for words in row:
            subtokens_cnt.append(len(words))
            subtokens_dict[len(words)] = words
    return subtokens_cnt, subtokens_dict

def tokenize_dataset(dataframe:pd.DataFrame, tokenizer, return_length=False):
    dataframe['tokens'] = [[] for _ in range(len(dataframe))]
    if return_length:
        dataframe['length'] = 0
    for row in range(len(dataframe)):
        for word in dataframe.loc[row, 'source_words']:
            tokens = tokenizer.encode(word).tokens
            dataframe.loc[row, 'tokens'].append(tokens)
            if return_length:
                 dataframe.loc[row, 'length'] += len(tokens)
    return dataframe

logging.info('Считывание датасета')
if DATASET_TO_PREPARE == 'merged':
    train_df = pd.concat([
        pd.read_parquet(os.path.join(TAIGA_PATH, f'taiga_train.parquet')),
        pd.read_parquet(os.path.join(SYNTAGRUS_PATH, f'syntagrus_train.parquet'))])
    
    validation_df = pd.concat([
        pd.read_parquet(os.path.join(TAIGA_PATH, f'taiga_dev.parquet')),
        pd.read_parquet(os.path.join(SYNTAGRUS_PATH, f'syntagrus_dev.parquet'))])
    
    test_df = pd.concat([
        pd.read_parquet(os.path.join(TAIGA_PATH, f'taiga_test.parquet')),
        pd.read_parquet(os.path.join(SYNTAGRUS_PATH, f'syntagrus_test.parquet'))])
else:
    train_df = pd.read_parquet(os.path.join(DATASET_PATH, f'{DATASET_TO_PREPARE}_train.parquet'))
    validation_df = pd.read_parquet(os.path.join(DATASET_PATH, f'{DATASET_TO_PREPARE}_dev.parquet'))
    test_df = pd.read_parquet(os.path.join(DATASET_PATH, f'{DATASET_TO_PREPARE}_test.parquet'))

if USE_PRETRAINDED_TOKENIZER:
    logging.info('Инициализация предобученного токенизатора')
    tokenizer = BPETokenizer.from_pretrained(f'{CHECKPOINTS_FILEPATH}/tokenizer.json')
else:
    logging.info('Обучение нового токенизатора')
    tokenizer = BPETokenizer.train([CORPUS_TEXTS_PATH], VOCABULARY_SIZE, MIN_FRECQUENCY_PAIR, unk_token=UNK_TOKEN, pad_token=PAD_TOKEN)
    tokenizer.save(f'{CHECKPOINTS_FILEPATH}/tokenizer.json')

train_df = train_df.reset_index().drop(columns=['index'])
test_df = test_df.reset_index().drop(columns=['index'])
validation_df = validation_df.reset_index().drop(columns=['index'])

logging.info('Токенизация датасета')
train_df = tokenize_dataset(train_df, tokenizer)
test_df = tokenize_dataset(test_df, tokenizer)
validation_df = tokenize_dataset(validation_df, tokenizer)

# Проанализируем текущую конфигурацию токенизации слов
dataframes = {'train' : train_df, 'test' : test_df, 'validation_df' : validation_df}
for df_name, df in dataframes.items():
    subtokens_cnt, subtokens_dict = get_subtokens_cnt(df)
    np_arr = np.array(subtokens_cnt)
    print(f'Характеристики распределения количества токенов в датафрейме {df_name}')
    print(f'Медиана = {np.median(np_arr)}')
    print(f'Среднее = {np.mean(np_arr)}')
    print(f'Максимум = {np.max(np_arr)}')
    quantile_level = 0.98
    quantile = np.quantile(np_arr, quantile_level)
    print(f'Квантиль уровня {quantile_level} = {quantile}')
    # for count, subtoken in subtokens_dict.items():
    #     if count > quantile:
    #         print(f'Аномальное значение субтокенов слова. Всего {count} субтокенов в слове {subtoken}')
        
    print('\n', '='*20)

# # Счетчик ссылок url
# urls_counter = 0
# for df_name, df in dataframes.items():
#     for row_idx in range(len(df)):
#         cur_words = df.loc[row_idx, 'source_words']
#         for word_idx, word in enumerate(cur_words):
#             if 'www' in word:
#                 urls_counter += 1
#                 # print(word_idx)
#                 # print(row_idx)
#                 for column_name in df.columns:
#                     pass
#                     # print(f'Столбец {column_name} : {df.loc[row_idx, column_name][word_idx]}')
# print(f'urls_counter = {urls_counter}')

MAX_WORDS_COUNT = max(find_max_words_source_len(test_df), find_max_words_source_len(validation_df))
# Берем за максимальное количество субтокенов слова значения квантиля уровня 0.98. Слова, большие этого значения будут обрезаться, но их невероятно мало
MAX_SUBTOKENS_COUNT = int(quantile)
MAX_LETTERS_COUNT = max(find_max_letters_cnt(test_df), find_max_letters_cnt(validation_df))
if ADD_BOS_EOS_TOKENS:
    MAX_WORDS_COUNT += 2

logging.info(f'Максимальное количество слов {MAX_WORDS_COUNT}')
logging.info(f'Максимальное количество субслов в словах {MAX_SUBTOKENS_COUNT}')
logging.info(f'Максимальное количество букв в словах {MAX_LETTERS_COUNT}')

# Оставляем в обучающей выборке только строки длины не большей, чем в тестовой выборке (удаляются всего 2 записи)
train_df = train_df.loc[(train_df['source_words'].apply(len) <= MAX_WORDS_COUNT)]
train_df = train_df.reset_index()

target_names = ['upos', 'head', 'deprel', 'Mood', 'NumType', 'VerbForm',
       'ExtPos', 'Reflex', 'Polarity', 'Typo', 'NameType', 'InflClass',
       'Person', 'Poss', 'Animacy', 'Degree', 'Foreign', 'Variant', 'Number',
       'Gender', 'NumForm', 'Aspect', 'Case', 'PronType', 'Tense', 'Abbr', 'Voice']
source_name = 'source_text'


# Либо используем подготовленные словари
if USE_PRETRAINDED_TOKENIZER:
    logging.info('Инициализация словарей из json')
    # Всегда используем merged в случае предобченного токенизатора, поскольку токенизатор должен быть обучен на как можно более разнообразной выборке
    source_vocab = Vocabulary.from_json(f'{DATA_INFO_FILEPATH}/merged_source_vocab.json')
    with open(f'{DATA_INFO_FILEPATH}/merged_target_vocabs.json', 'r', encoding='utf-8') as file:
        target_vocabs_dict = json.load(file)
    target_vocabs = {target_name: Vocabulary.from_serializable(target_vocabs_dict[target_name]) for target_name in target_names}
    letters_vocab = Vocabulary.from_json(f'{DATA_INFO_FILEPATH}/merged_letters_vocab.json')
# Либо инициализируем новые (В случае обучения нового токенизатора)
else:
    logging.info('Заполнение словарей с нуля')
    source_vocab = Vocabulary(bos_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, mask_token=MASK_TOKEN, unk_token=UNK_TOKEN, add_bos_eos_tokens=ADD_BOS_EOS_TOKENS)
    target_vocabs = {target_name: Vocabulary(bos_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN,\
                                            mask_token=MASK_TOKEN, unk_token=UNK_TOKEN, add_bos_eos_tokens=ADD_BOS_EOS_TOKENS) for target_name in target_names}
    target_vocabs = {}
    for target_name in target_names:
        target_vocabs[target_name] = Vocabulary(bos_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN,
                                                mask_token=MASK_TOKEN, unk_token=UNK_TOKEN, add_bos_eos_tokens=ADD_BOS_EOS_TOKENS)
        # Если хотим исключить отсутствующие граммемы из обучающих данных, то заменяем их на паддинг
        if EXCLUDE_UNUSED_GRAMMEMES:
            target_vocabs[target_name].token_to_idx['None'] = target_vocabs[target_name].pad_idx

    # Заполненеие словаря входных токенов и словаря таргет меток
    for df in [train_df, validation_df, test_df]:
        for row in range(len(df)):
            for token_lst in df.loc[row, 'tokens']:
                source_vocab.add_tokens(token_lst)
            for target_name in target_names:
                target_vocabs[target_name].add_tokens(df.loc[row, target_name])

    # Словарь для букв
    letters_vocab = Vocabulary(bos_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, mask_token=MASK_TOKEN, unk_token=UNK_TOKEN, add_bos_eos_tokens=False)

    for df in [train_df, validation_df, test_df]:
        for i in range(len(df)):
            for word in df.loc[i, 'source_words']:
                letters_vocab.add_tokens(list(word))

    # Сохранение словарей
    source_vocab.to_json(f'{DATA_INFO_FILEPATH}/{DATASET_TO_PREPARE}_source_vocab.json')
    target_vocabs_dict = {target_name : target_vocabs[target_name].to_serializable() for target_name in target_names}
    with open(f'{DATA_INFO_FILEPATH}/{DATASET_TO_PREPARE}_target_vocabs.json', 'w', encoding='utf-8') as file:
        json.dump(target_vocabs_dict, file)
    letters_vocab.to_json(f'{DATA_INFO_FILEPATH}/{DATASET_TO_PREPARE}_letters_vocab.json')

pad_idx = source_vocab.pad_idx
trg_vocabs_len = {key:len(target_vocabs[key]) for key in target_names}

print(f'Длина словаря токенов = {len(source_vocab)}')
# print(f'Длина словаря символов = {len(letters_vocab)}')
for key in target_names:
    print(f'Длина словаря признака {key} = {len(target_vocabs[key])}')

# Проверка на соответсвие грамматических атрибутов в разных корпусах.
# for target_name in target_names:
#     print(f'признак {target_name}')
#     for syn_key in syntagrus_labels[target_name].keys():
#         if syn_key not in taiga_labels[target_name]:
#             print(f'Ключа {syn_key} нет в корпуса тайга')
#     for tai_key in taiga_labels[target_name].keys():
#         if tai_key not in syntagrus_labels[target_name]:
#             print(f'Ключа {tai_key} нет в корпуса синтагрус')

vectorizer = Vectorizer(source_vocab, target_vocabs, letters_vocab, WORD_REPRESENTATION, pad_idx)

logging.info('Получение индексов токенов и создание соответстующего столбца в датафрейме')
train_df['input_ids'] = None
for target_name in target_names:
    train_df[f'{target_name}_ids'] = None

for i in range(len(train_df)):
    train_df.at[i, 'input_ids'] = vectorizer.vectorize_tokens(MAX_SUBTOKENS_COUNT, MAX_WORDS_COUNT, train_df.loc[i, 'tokens'])
    trg_vectoized = vectorizer.vectorize_targets(train_df.loc[i], target_names, MAX_WORDS_COUNT, ADD_BOS_EOS_TOKENS)
    for target_name in target_names:
        train_df.at[i, f'{target_name}_ids'] = trg_vectoized[target_name]

validation_df['input_ids'] = None
for target_name in target_names:
    validation_df[f'{target_name}_ids'] = None

for i in range(len(validation_df)):
    validation_df.at[i, 'input_ids'] = vectorizer.vectorize_tokens(MAX_SUBTOKENS_COUNT, MAX_WORDS_COUNT, validation_df.loc[i, 'tokens'])
    trg_vectoized = vectorizer.vectorize_targets(validation_df.loc[i], target_names, MAX_WORDS_COUNT, ADD_BOS_EOS_TOKENS)
    for target_name in target_names:
        validation_df.at[i, f'{target_name}_ids'] = trg_vectoized[target_name]

test_df['input_ids'] = None
for target_name in target_names:
    test_df[f'{target_name}_ids'] = None

for i in range(len(test_df)):
    test_df.at[i, 'input_ids'] = vectorizer.vectorize_tokens(MAX_SUBTOKENS_COUNT, MAX_WORDS_COUNT, test_df.loc[i, 'tokens'])
    trg_vectoized = vectorizer.vectorize_targets(test_df.loc[i], target_names, MAX_WORDS_COUNT, ADD_BOS_EOS_TOKENS)
    for target_name in target_names:
        test_df.at[i, f'{target_name}_ids'] = trg_vectoized[target_name]

train_df = train_df.drop(columns=[*[target_name for target_name in target_names], 'index', 'lemmas', 'xpos', 'feats', 'misc', 'source_text'])
validation_df = validation_df.drop(columns=[*[target_name for target_name in target_names], 'lemmas', 'xpos', 'feats', 'misc', 'source_text'])
test_df = test_df.drop(columns=[*[target_name for target_name in target_names], 'lemmas', 'xpos', 'feats', 'misc', 'source_text'])

logging.info('Сохранение полученной конфигурации в файл')
with open(f'{DATA_INFO_FILEPATH}/{DATASET_TO_PREPARE}_vocabs_configuration.json', 'w', encoding='utf-8') as file:
    json.dump({
    'MIN_FRECQUENCY_PAIR' : MIN_FRECQUENCY_PAIR,
    'MAX_WORDS_COUNT' : MAX_WORDS_COUNT,
    'MAX_SUBTOKENS_COUNT' : MAX_SUBTOKENS_COUNT,
    'MAX_LETTERS_COUNT' : MAX_LETTERS_COUNT,
    'SOURCE_VOCAB_LEN' :  len(source_vocab),
    'LETTERS_VOCAB_LEN' : len(letters_vocab),
    'TRG_VOCABS_LEN' : trg_vocabs_len,
    'PAD_IDX' : pad_idx}, file, indent=4, ensure_ascii=False)

logging.info('Сохранение полученных датасетов в формате .parquet')
if 'Clitic' in train_df.columns:
    train_df = train_df.drop(columns=['Clitic'])
train_df.to_parquet(os.path.join(DATASET_SAVE_FILEPATH, f'{DATASET_TO_PREPARE}_prepared_train.parquet'), engine="fastparquet", index=False)
test_df.to_parquet(os.path.join(DATASET_SAVE_FILEPATH, f'{DATASET_TO_PREPARE}_prepared_test.parquet'), engine="fastparquet", index=False)
validation_df.to_parquet(os.path.join(DATASET_SAVE_FILEPATH, f'{DATASET_TO_PREPARE}_prepared_dev.parquet'), engine="fastparquet", index=False)
logging.info('Готово!')