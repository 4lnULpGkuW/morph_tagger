from conllu import parse_incr
import pandas as pd
import os
import sys
from dotenv import load_dotenv
from pathlib import Path
import logging

# Переопределяем параметры логгирования для вывода сообщений уровня info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

if sys.platform == 'linux':
    load_dotenv(dotenv_path=(Path('.')/'.env.linux'))
elif sys.platform == 'win32':
    load_dotenv(dotenv_path=(Path('.')/'.env.win'))
else:
    raise ValueError('Ваша операционная система не поддерживается!')

DATASETS_FOLDER_PATH = os.getenv('DATASETS_FOLDER_PATH')
SYNTAGRUS_VERSION = os.getenv('SYNTAGRUS_VERSION', '2.16') # Допустимые занчения: 2.3; 2.16 | В версии 2.3 меньше тренировочных примеров, по сравнению с 2.16. Точность на тестовой выборке практически не меняется
SYNTAGRUS_PATH = os.getenv('SYNTAGRUS_PATH')
SYNTAGRUS_TEXTS_PATH = os.getenv('SYNTAGRUS_TEXTS_PATH')

TAIGA_PATH = os.getenv('TAIGA_PATH')
TAIGA_TEXTS_PATH = os.getenv('TAIGA_TEXTS_PATH')

MERGED_PATH = os.path.join(DATASETS_FOLDER_PATH, 'sintagrus_taiga_merged')
MERGED_TEXTS_PATH = os.path.join(DATASETS_FOLDER_PATH, 'sintagrus_taiga_merged.txt')

DATASET_TO_PREPARE = 'merged' # taiga, syntagrus or merged

# Словарик для определения имен файлов в зависимости от выбранной опции
datasets_filenames = {
    'syntagrus': {
        '2.16' : {
            'train_list' : ['ru_syntagrus-ud-train-a.conllu', 'ru_syntagrus-ud-train-b.conllu', 'ru_syntagrus-ud-train-c.conllu'],
            'test_list' : ['ru_syntagrus-ud-test.conllu', 'ru_syntagrus-ud-dev.conllu'],
        },
        '2.3' : {
            'train_list' : ['ru_syntagrus-ud-train.conllu'],
            'test_list' : ['ru_syntagrus-ud-test.conllu', 'ru_syntagrus-ud-dev.conllu'],
        },
    },
    'taiga' : {
        'train_list' : ['ru_taiga-ud-train-a.conllu', 'ru_taiga-ud-train-b.conllu', 'ru_taiga-ud-train-c.conllu', 'ru_taiga-ud-train-d.conllu', 'ru_taiga-ud-train-e.conllu'],
        'test_list' : ['ru_taiga-ud-dev.conllu', 'ru_taiga-ud-test.conllu'],
    }
}

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

# Выполняем "развертку" вложенных элементов
def unfold_nested_elements(df, feat_column:str='feats'):
    # Выполняем предварительный проход для поиска всех грамматических атрибутов
    nested_features = set()
    
    # Используем iterrows() для безопасного доступа к строкам
    for idx, row in df.iterrows():
        df_row = row[feat_column]
        # Проверяем, что значение является списком
        if isinstance(df_row, list):
            for item in df_row:
                if isinstance(item, dict):
                    for feature in item.keys():
                        nested_features.add(feature)
    
    # Заполняем None для всех грамматических атрибутов
    for feature in nested_features:
        if feature not in df.columns:
            # Создаем список значений для каждого слова в предложении
            df[feature] = df[feat_column].apply(
                lambda x: ['None'] * len(x) if isinstance(x, list) else []
            )
    
    # Заполняем реальные значения для грамматических атрибутов
    for idx, row in df.iterrows():
        df_row = row[feat_column]
        if isinstance(df_row, list):
            for dict_idx, item in enumerate(df_row):
                if isinstance(item, dict):
                    for feature, value in item.items():
                        # Обновляем значение для конкретной позиции
                        current_list = df.at[idx, feature]
                        if isinstance(current_list, list) and dict_idx < len(current_list):
                            current_list[dict_idx] = value
                            df.at[idx, feature] = current_list
    
    return df

def unpack_conllu_and_create_dataframes(dataset_path, train_list, test_list):
    '''Выполняет распаковку формата conllu'''
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    dev_df = pd.DataFrame()
    for dataset_name in train_list:
        with open(os.path.join(dataset_path, dataset_name), 'r', encoding='utf-8') as data_file:
            parsed_list = list(parse_incr(data_file))

        source_text = []
        source_words = []
        lemmas = []
        upos = []
        xpos = []
        feats = []
        head = []
        deprel = []
        deps = []
        misc = []

        for seq in parsed_list:
            source_text.append(seq.metadata['text'])
            source_words.append([str(token) for token in seq])
            lemmas.append([grammem['lemma'] for grammem in seq])
            upos.append([grammem['upos'] for grammem in seq])
            xpos.append([grammem['xpos'] for grammem in seq])
            feats.append([grammem['feats'] for grammem in seq])
            head.append([grammem['head'] for grammem in seq])
            deprel.append([grammem['deprel'] for grammem in seq])
            deps.append([grammem['deps'] for grammem in seq])
            misc.append([grammem['misc'] for grammem in seq])

        temp_df = pd.DataFrame({
        'source_text': source_text,
        'source_words': source_words,
        'lemmas': lemmas,
        'upos': upos,
        'xpos': xpos,
        'feats': feats,
        'head': head,
        'deprel': deprel,
        'deps' : deps,
        'misc': misc,
        }, dtype='object')

        train_df = pd.concat((train_df, temp_df))

    for datapath in test_list:
        data_file = open(os.path.join(dataset_path, datapath), 'r', encoding='utf-8')
        parsed_list = list(parse_incr(data_file))

        source_text = []
        source_words = []
        lemmas = []
        upos = []
        xpos = []
        feats = []
        head = []
        deprel = []
        deps = []
        misc = []

        for seq in parsed_list:
            source_text.append(seq.metadata['text'])
            source_words.append([str(token) for token in seq])
            lemmas.append([grammem['lemma'] for grammem in seq])
            upos.append([grammem['upos'] for grammem in seq])
            xpos.append([grammem['xpos'] for grammem in seq])
            feats.append([grammem['feats'] for grammem in seq])
            head.append([grammem['head'] for grammem in seq])
            deprel.append([grammem['deprel'] for grammem in seq])
            deps.append([grammem['deps'] for grammem in seq])
            misc.append([grammem['misc'] for grammem in seq])

        temp_df = pd.DataFrame({
        'source_text': source_text,
        'source_words': source_words,
        'lemmas': lemmas,
        'upos': upos,
        'xpos': xpos,
        'feats': feats,
        'head': head,
        'deprel': deprel,
        'deps' : deps,
        'misc': misc,
        }, dtype='object')

        if 'test' in datapath:
            test_df = temp_df
        else:
            dev_df = temp_df
    return (train_df, dev_df, test_df)

if DATASET_TO_PREPARE == 'taiga':
    train_df, dev_df, test_df = unpack_conllu_and_create_dataframes(TAIGA_PATH, datasets_filenames['taiga']['train_list'], datasets_filenames['taiga']['test_list'])
elif DATASET_TO_PREPARE == 'syntagrus':
    train_df, dev_df, test_df = unpack_conllu_and_create_dataframes(SYNTAGRUS_PATH, datasets_filenames['syntagrus'][SYNTAGRUS_VERSION]['train_list'],\
                                                                    datasets_filenames['syntagrus'][SYNTAGRUS_VERSION]['test_list'])
elif DATASET_TO_PREPARE == 'merged':
    syn_train_df, syn_dev_df, syn_test_df = unpack_conllu_and_create_dataframes(SYNTAGRUS_PATH, datasets_filenames['syntagrus'][SYNTAGRUS_VERSION]['train_list'],\
                                                                    datasets_filenames['syntagrus'][SYNTAGRUS_VERSION]['test_list'])
    
    taiga_train_df, taiga_dev_df, taiga_test_df = unpack_conllu_and_create_dataframes(TAIGA_PATH, datasets_filenames['taiga']['train_list'],\
                                                                                      datasets_filenames['taiga']['test_list'])
    
    train_df = pd.concat([syn_train_df, taiga_train_df], axis=0)
    dev_df = pd.concat([syn_dev_df, taiga_dev_df], axis=0)
    test_df = pd.concat([syn_test_df, taiga_test_df], axis=0)

train_df = train_df.reset_index()
train_df = unfold_nested_elements(train_df)
train_df['head'] = train_df['head'].apply(lambda x: [str(num) for num in x])

test_df = test_df.reset_index()
test_df = unfold_nested_elements(test_df)
test_df['head'] = test_df['head'].apply(lambda x: [str(num) for num in x])

dev_df = dev_df.reset_index()
dev_df = unfold_nested_elements(dev_df)
dev_df['head'] = dev_df['head'].apply(lambda x: [str(num) for num in x])

logging.info(f'Длина тренировочного датасета {len(train_df)}')
logging.info(f'Длина тестового датасета {len(test_df)}')
logging.info(f'Длина валидационного датасета {len(dev_df)}')

for target_name in train_df.columns:
    if target_name not in test_df.columns:
        logging.warning(f'target_name {target_name} from train_df not in test_df.columns')
    if target_name not in dev_df.columns:
        logging.warning(f'target_name {target_name} from train_df not in dev_df.columns')
for target_name in test_df.columns:
    if target_name not in train_df.columns:
        logging.warning(f'target_name {target_name} from test_df not in train_df.columns')
    if target_name not in dev_df.columns:
        logging.warning(f'target_name {target_name} from test_df not in dev_df.columns')
for target_name in dev_df.columns:
    if target_name not in test_df.columns:
        logging.warning(f'target_name {target_name} dev_df from not in test_df.columns')
    if target_name not in train_df.columns:
        logging.warning(f'target_name {target_name} dev_df from not in train_df.columns')

# Создание текстового файла всех исходных текстов обучающей выборки
with open(CORPUS_TEXTS_PATH, "w", encoding="utf-8") as file:
    for raw_text in train_df['source_text']:
        file.write(raw_text + '\n')

    for raw_text in test_df['source_text']:
        file.write(raw_text + '\n')

    for raw_text in dev_df['source_text']:
        file.write(raw_text + '\n')

train_df.drop(columns=['index']).to_parquet(os.path.join(DATASET_PATH, f'{DATASET_TO_PREPARE}_train.parquet'), engine="fastparquet", index=False)
test_df.drop(columns=['index']).to_parquet(os.path.join(DATASET_PATH, f'{DATASET_TO_PREPARE}_test.parquet'), engine="fastparquet", index=False)
dev_df.drop(columns=['index']).to_parquet(os.path.join(DATASET_PATH, f'{DATASET_TO_PREPARE}_dev.parquet'), engine="fastparquet", index=False)


