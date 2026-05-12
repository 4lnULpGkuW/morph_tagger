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

# Парсинг аргумента командной строки
parser = argparse.ArgumentParser(description='Обучение модели морфологического классификатора')
parser.add_argument(
    '--dataset',
    type=str,
    default='merged',
    choices=['taiga', 'syntagrus', 'merged'],
    help='Выбор датасета для обучения: taiga, syntagrus или merged (слияние taiga и syntagrus). default = merged',
    required=True,
)
parser.add_argument(
    '--pretrained',
    action='store_true',
    help='Использование предобученной модели или обучение новой.',
)
parser.add_argument(
    '--epochs',
    type=int,
    default=35,
    help='Выбор Количества эпох обучения. default = 35'
)
parser.add_argument(
    '--batch',
    type=int,
    default=96,
    help='Выбор размера батча для обучения. default = 96'
)
parser.add_argument(
    '--device',
    choices=['cpu', 'cuda'],
    default='cuda',
    help='Устройство для вычислений. default = cuda'
)
parser.add_argument(
    '--checkpoint_epoch',
    type=int,
    default=2,
    help='Сохранение параметров модели и метрик обучения каждые checkpoint_epoch эпох. default = 2'
)
parser.add_argument(
    '--mask_prob',
    type=float,
    default=0.15,
    help='Вероятность маскирования токена (0.0 - 1.0). default = 0.15'
)
parser.add_argument(
    '--mask_alpha',
    type=float,
    default=0.0,
    help='Аддитивный приоритет замаскированных токенов: loss = mean(ce * w), где w=1+mask_alpha для маски, w=1 для остальных. 0.0 = без приоритизации. default = 0.0'
)
parser.add_argument(
    '--target_pos_idx',
    type=int,
    default=-1,
    help='Индекс POS для таргетированного маскирования. -1 = случайное маскирование всех токенов (оригинальное поведение). default = -1'
)
parser.add_argument(
    '--use_expand',
    action='store_true',
    help='Разворачивать батч: создавать отдельную копию предложения под каждую выбранную маску. По умолчанию выключено.'
)
parser.add_argument('--max_masks_per_sentence', type=int, default=None,
    help='Максимум масок на предложение при таргетированном маскировании. None = без ограничения.')

# Параметры обучения модели
USE_CLASSES_WEIGHTS = False
CLASSES_WEIGHTS_SCALER = 12

SHUFFLE = True
DROP_LAST = True
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
EPOCHS = args.epochs
CHECKPOINT_EPOCH = args.checkpoint_epoch
DEVICE = args.device
DEVICE = DEVICE if torch.cuda.is_available() else 'cpu'
MASK_PROB = args.mask_prob
MASK_ALPHA = args.mask_alpha
TARGET_POS_IDX = args.target_pos_idx
USE_EXPAND = args.use_expand
MAX_MASKS_PER_SENTENCE = args.max_masks_per_sentence
logging.info(f'''Текущие параметры обработки датасета и конфигурация токенизатора:
             DATASET_TO_PREPARE: {DATASET_TO_PREPARE}
             BATCH_SIZE: {BATCH_SIZE}
             CHECKPOINT_EPOCH: {CHECKPOINT_EPOCH}
             DEVICE: {DEVICE}
             MASK_PROB: {MASK_PROB}
             MASK_ALPHA: {MASK_ALPHA}
             TARGET_POS_IDX: {TARGET_POS_IDX} (-1 = все токены случайно)
             USE_EXPAND: {USE_EXPAND}
             MAX_MASKS_PER_SENTENCE: {MAX_MASKS_PER_SENTENCE}
             USE_PRETRAINED: {USE_PRETRAINED}''')

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

def apply_masking(batch_dict, mask_prob, device, pad_idx, mask_idx, target_pos_idx=-1, use_expand=False, max_masks_per_sentence=None):
    '''Применяет маскирование к батчу.

    target_pos_idx = -1  -> случайное маскирование всех токенов (оригинальное поведение)
    target_pos_idx >= 0  -> маскируются только слова с указанным upos-индексом

    use_expand = False   -> батч не дублируется, маска применяется напрямую
    use_expand = True    -> под каждое выбранное слово создаётся отдельная копия предложения

    Возвращает (out_batch_dict, input_ids, letters, word_mask):
        out_batch_dict — батч с таргетами (может быть развёрнутым при use_expand)
        input_ids      — [B', S, T] с применёнными масками
        letters        — [B', S, L] с применёнными масками (или None)
        word_mask      — [B', S] bool, True для замаскированных слов
    '''
    batch_size = batch_dict['input_ids'].shape[0]
    seq_len = batch_dict['upos'].shape[1]

    if not use_expand:
        input_ids = batch_dict['input_ids'].clone()
        letters = batch_dict.get('letters', None)
        if letters is not None:
            letters = letters.clone()

        if target_pos_idx == -1:
            # Случайное маскирование на уровне сабтокенов [B, S, T] — оригинальное поведение
            prob_matrix = torch.full(input_ids.shape, mask_prob, device=device)
            prob_matrix.masked_fill_(input_ids == pad_idx, 0.0)
            mask_3d = torch.bernoulli(prob_matrix).bool()
        else:
            # Таргетированное маскирование: сэмплируем на уровне слов [B, S],
            # затем разворачиваем до [B, S, T] чтобы закрыть все сабтокены слова
            pos_candidates = (batch_dict['upos'] == target_pos_idx) & (batch_dict['upos'] != pad_idx)
            prob_matrix_2d = torch.zeros(batch_size, seq_len, device=device)
            prob_matrix_2d[pos_candidates] = mask_prob
            word_selected = torch.bernoulli(prob_matrix_2d).bool()  # [B, S]
            
            # Ограничение: не более max_masks существительных на предложение
            if max_masks_per_sentence is not None:
                for i in range(word_selected.shape[0]):
                    selected_pos = word_selected[i].nonzero(as_tuple=True)[0]
                    if len(selected_pos) > max_masks_per_sentence:
                        keep = selected_pos[torch.randperm(len(selected_pos))[:max_masks_per_sentence]]
                        word_selected[i] = False
                        word_selected[i][keep] = True

            mask_3d = word_selected.unsqueeze(-1).expand_as(input_ids).clone()
            mask_3d &= (input_ids != pad_idx)

        input_ids[mask_3d] = mask_idx
        if letters is not None:
            letters[mask_3d] = pad_idx

        # Схлопываем до [B, S]: слово замаскировано если хотя бы один его сабтокен замаскирован
        word_mask = mask_3d.any(dim=-1)
        return batch_dict, input_ids, letters, word_mask

    else:
        # Expand: каждое выбранное слово порождает отдельную копию предложения с одной маской
        new_batch_dict = {k: [] for k in batch_dict.keys()}
        new_word_masks = []

        for i in range(batch_size):
            if target_pos_idx == -1:
                # Для expand при случайном маскировании сэмплируем на уровне слов
                valid = (batch_dict['upos'][i] != pad_idx)
                prob_vec = torch.zeros(seq_len, device=device)
                prob_vec[valid] = mask_prob
                selected = torch.bernoulli(prob_vec).bool().nonzero(as_tuple=True)[0]
            else:
                candidates = (
                    (batch_dict['upos'][i] == target_pos_idx) &
                    (batch_dict['upos'][i] != pad_idx)
                ).nonzero(as_tuple=True)[0]

                if len(candidates) > 0:
                    prob_vec = torch.full((len(candidates),), mask_prob, device=device)
                    sel_mask = torch.bernoulli(prob_vec).bool()
                    selected = candidates[sel_mask]
                else:
                    selected = torch.tensor([], dtype=torch.long, device=device)

            if len(selected) > 0:
                # Создаем отдельный дубль под каждую выбранную маску
                for idx in selected:
                    for k in batch_dict.keys():
                        new_batch_dict[k].append(batch_dict[k][i])
                    wm = torch.zeros(seq_len, dtype=torch.bool, device=device)
                    wm[idx] = True
                    new_word_masks.append(wm)
            else:
                # Если не выпало ни одной маски — добавляем оригинал как чистый контекст
                for k in batch_dict.keys():
                    new_batch_dict[k].append(batch_dict[k][i])
                new_word_masks.append(torch.zeros(seq_len, dtype=torch.bool, device=device))

        for k in new_batch_dict.keys():
            new_batch_dict[k] = torch.stack(new_batch_dict[k])

        word_mask = torch.stack(new_word_masks)  # [B', S]

        input_ids = new_batch_dict['input_ids'].clone()
        letters = new_batch_dict.get('letters', None)
        if letters is not None:
            letters = letters.clone()

        # word_mask [B', S] — маскируем все сабтокены выбранных слов
        input_ids[word_mask] = mask_idx
        if letters is not None:
            letters[word_mask] = pad_idx

        return new_batch_dict, input_ids, letters, word_mask

def compute_loss(predictions:dict[str:torch.tensor], targets:dict[str:list[int]], target_names:list[str], mask_target:torch.Tensor, mask_alpha:float, pad_idx:int=0):
    predictions, targets = normalize_sizes(predictions, targets, target_names)
    mask_target_flat = mask_target.contiguous().view(-1)

    losses = {}
    total_loss = 0
    for key in target_names:
        preds = predictions[key]
        targs = targets[key]

        # Cross Entropy без редукции для поштучного контроля
        ce_loss = torch.nn.functional.cross_entropy(preds, targs, ignore_index=pad_idx, reduction='none')

        # Все токены с весом 1.0, замаскированные получают дополнительный буст
        weights = torch.ones_like(ce_loss)
        weights[mask_target_flat & (targs != pad_idx)] = 1.0 + mask_alpha

        valid = targs != pad_idx
        loss_feature = (ce_loss * weights)[valid].mean()

        losses[key] = loss_feature
        total_loss += loss_feature

    return total_loss, losses

def compute_metrics(predictions, targets, target_names, mask_target, pad_idx=0, average='macro'):
    # mask_target всегда [B, S] — word_mask из apply_masking или zero_mask из валидации
    metrics_dict = {}
    first_key = target_names[0]
    batch_size, seq_len = targets[first_key].size()
    device = predictions[first_key].device
    correct_words_all = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    
    for key in target_names:
        _, pred_indices = predictions[key].max(dim=-1)  # [B, S]
        
        # Маска значимых токенов
        mask = targets[key] != pad_idx  # [B, S]
        
        correct = (pred_indices == targets[key])  # [B, S]
        
        # Общая правильность слова
        correct_words_all = correct_words_all & correct
        
        # Вычисляем метрики для текущего признака
        errors_per_sentence = ((pred_indices != targets[key]) & mask).sum(dim=1)  # [B]
        sentence_correct = errors_per_sentence == 0  # [B]
        sentence_accuracy = sentence_correct.float().mean().item()
        
        # Фильтруем паддинг
        pred_filtered = pred_indices[mask].cpu().numpy()
        target_filtered = targets[key][mask].cpu().numpy()
        
        precision, recall, f1, _ = precision_recall_fscore_support(target_filtered, pred_filtered, average=average, zero_division=0)
        
        accuracy = (pred_filtered == target_filtered).mean()

        # Точность только на замаскированных токенах
        m_mask = mask_target & mask
        if m_mask.any():
            pred_masked = pred_indices[m_mask].cpu().numpy()
            target_masked = targets[key][m_mask].cpu().numpy()
            accuracy_masked = (pred_masked == target_masked).mean()
        else:
            accuracy_masked = 0.0
        
        metrics_dict[key] = {
            'accuracy': accuracy,
            'accuracy_masked': float(accuracy_masked),
            'sentence_accuracy': sentence_accuracy,
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }
    
    # Берем маску от первого признака (везде паддинг одинаков)
    mask = targets[first_key] != pad_idx
    
    # Точность предсказания всех морфем в слове
    word_accuracy = correct_words_all[mask].float().mean().item()
    
    # Точность предсказания предложения целиком (все слова в предложении верны)
    sentence_errors = (~correct_words_all & mask).sum(dim=1)  # [B]
    sentence_correct_global = sentence_errors == 0
    sentence_accuracy_global = sentence_correct_global.float().mean().item()
    
    metrics_dict['word_accuracy'] = word_accuracy
    metrics_dict['sentence_accuracy_global'] = sentence_accuracy_global
    
    return metrics_dict


logging.info('Загрука датасетов...')
train_df = pd.read_parquet(os.path.join(DATASET_SAVE_FILEPATH, f'{DATASET_TO_PREPARE}_prepared_train.parquet'))
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
MASK_IDX = vocabs_config['MASK_IDX']

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
    
    model = torch.load(MODEL_SAVE_FILEPATH, weights_only=False, map_location=torch.device(DEVICE))
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
        train_epoch_metrics = {key:{'accuracy' : 0.0, 'accuracy_masked': 0.0, 'sentence_accuracy' : 0.0, 'precision' : 0.0,
                                    'recall' : 0.0, 'f1' : 0.0, 'mean_loss' : 0.0} for key in target_names}
        train_epoch_metrics['word_accuracy'] = 0.0
        train_epoch_metrics['sentence_accuracy_global'] = 0.0
        
        model.train()
        for batch_idx, batch_dict in enumerate(batch_generator):

            optimizer.zero_grad()

            exp_batch_dict, input_ids, letters, word_mask = apply_masking(
                batch_dict, MASK_PROB, DEVICE, PAD_IDX, MASK_IDX,
                target_pos_idx=TARGET_POS_IDX, use_expand=USE_EXPAND,
                max_masks_per_sentence=MAX_MASKS_PER_SENTENCE
            )

            if WORD_REPRESENTATION == 'tokens':
                predictions = model(tokens=input_ids, letters=None)
            elif WORD_REPRESENTATION == 'letters':
                predictions = model(tokens=None, letters=letters)
            else:
                predictions = model(tokens=input_ids, letters=letters)

            cur_metrics = compute_metrics(predictions, exp_batch_dict, target_names, word_mask, PAD_IDX)
            total_loss, train_losses = compute_loss(predictions, exp_batch_dict, target_names, word_mask, MASK_ALPHA, PAD_IDX)

            total_loss.backward()
            optimizer.step()

            # Метрики
            epoch_running_train_loss += (total_loss.item() - epoch_running_train_loss) / (batch_idx + 1)
            epoch_sum_train_loss += total_loss.item()

            for key in target_names:
                for metric, value in cur_metrics[key].items():
                    train_epoch_metrics[key][metric] += (value - train_epoch_metrics[key][metric]) / (batch_idx + 1)
                train_epoch_metrics[key]['mean_loss'] += (train_losses[key].item() - train_epoch_metrics[key]['mean_loss']) / (batch_idx + 1)
            train_epoch_metrics['word_accuracy'] += (cur_metrics['word_accuracy'] - train_epoch_metrics['word_accuracy']) / (batch_idx + 1)
            train_epoch_metrics['sentence_accuracy_global'] += (cur_metrics['sentence_accuracy_global'] - train_epoch_metrics['sentence_accuracy_global']) / (batch_idx + 1)
        train_end_time = time.time()

        dataset.set_dataframe_split('validation')
        batch_generator = generate_batches(dataset, BATCH_SIZE, SHUFFLE, DROP_LAST, DEVICE)
        epoch_sum_valid_loss = 0.0
        epoch_running_valid_loss = 0.0
        valid_epoch_metrics = {key:{'accuracy' : 0.0, 'accuracy_masked': 0.0, 'sentence_accuracy' : 0.0, 'precision' : 0.0,
                                    'recall' : 0.0, 'f1' : 0.0, 'mean_loss' : 0.0} for key in target_names}
        valid_epoch_metrics['word_accuracy'] = 0.0
        valid_epoch_metrics['sentence_accuracy_global'] = 0.0
        
        model.eval()
        valid_start_time = time.time()

        with torch.no_grad():
            for batch_idx, batch_dict in enumerate(batch_generator):
                
                # При валидации передаем оригинальные неискаженные данные (без маскирования)
                if WORD_REPRESENTATION == 'tokens':
                    predictions = model(tokens=batch_dict['input_ids'], letters=None)
                elif WORD_REPRESENTATION == 'letters':
                    predictions = model(tokens=None, letters=batch_dict['letters'])
                else:
                    predictions = model(tokens=batch_dict['input_ids'], letters=batch_dict['letters'])

                zero_mask = torch.zeros(
                    batch_dict['input_ids'].shape[0],
                    batch_dict['upos'].shape[1],
                    dtype=torch.bool,
                    device=DEVICE
                )

                cur_metrics = compute_metrics(predictions, batch_dict, target_names, zero_mask, PAD_IDX)
                total_loss, valid_losses = compute_loss(predictions, batch_dict, target_names, zero_mask, 0.0, PAD_IDX)

                # Средние потери и точность
                epoch_running_valid_loss += (total_loss.item() - epoch_running_valid_loss) / (batch_idx + 1)
                epoch_sum_valid_loss += total_loss.item()

                for key in target_names:
                    for metric, value in cur_metrics[key].items():
                        valid_epoch_metrics[key][metric] += (value - valid_epoch_metrics[key][metric]) / (batch_idx + 1)
                    valid_epoch_metrics[key]['mean_loss'] += (valid_losses[key].item() - valid_epoch_metrics[key]['mean_loss']) / (batch_idx + 1)
                # Обновляем агрегированные метрики
                valid_epoch_metrics['word_accuracy'] += (cur_metrics['word_accuracy'] - valid_epoch_metrics['word_accuracy']) / (batch_idx + 1)
                valid_epoch_metrics['sentence_accuracy_global'] += (cur_metrics['sentence_accuracy_global'] - valid_epoch_metrics['sentence_accuracy_global']) / (batch_idx + 1)
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
            print(f"Train: Ошибка на признаке {key}: {train_epoch_metrics[key]['mean_loss']}")
            print(f"Train: Точность на признаке {key}: {train_epoch_metrics[key]['accuracy']*100}%")
            print(f"Train: Точность (ТОЛЬКО НА МАСКАХ) на признаке {key}: {train_epoch_metrics[key]['accuracy_masked']*100}%")
            print(f"Train: Точность предложения на признаке {key}: {valid_epoch_metrics[key]['sentence_accuracy']*100}%")
            print(f"Train: precision на признаке {key}: {train_epoch_metrics[key]['precision']*100}%")
            print(f"Train: recall на признаке {key}: {train_epoch_metrics[key]['recall']*100}%")
            print(f"Train: f1-score на признаке {key}: {train_epoch_metrics[key]['f1']*100}%")
        print('-'*20)
        print('-'*20)
        print(f"Train: Точность предсказания всех морфем в слове: {train_epoch_metrics['word_accuracy']*100}%")
        print(f"Train: Точность предсказания предложения целиком: {train_epoch_metrics['sentence_accuracy_global']*100}%")
        print(f'Время выполнения {train_end_time - train_start_time}')

        print('-'*40)
        print(f'Validation: Средняя ошибка эпохи {epoch_running_valid_loss}')
        for key in target_names:
            print('-'*20)
            print(f"Validation: Ошибка на признаке {key}: {valid_epoch_metrics[key]['mean_loss']}")
            print(f"Validation: Точность на признаке {key}: {valid_epoch_metrics[key]['accuracy']*100}%")
            print(f"Validation: Точность предложения на признаке {key}: {valid_epoch_metrics[key]['sentence_accuracy']*100}%")
            print(f"Validation: precision на признаке {key}: {valid_epoch_metrics[key]['precision']*100}%")
            print(f"Validation: recall на признаке {key}: {valid_epoch_metrics[key]['recall']*100}%")
            print(f"Validation: f1-score на признаке {key}: {valid_epoch_metrics[key]['f1']*100}%")
        print('-'*20)
        print('-'*20)
        print(f"Validation: Точность предсказания всех морфем в слове: {valid_epoch_metrics['word_accuracy']*100}%")
        print(f"Validation: Точность предсказания предложения целиком: {valid_epoch_metrics['sentence_accuracy_global']*100}%")
        print(f'Время выполнения {valid_end_time - valid_start_time}')

        # Блок с сохранением результатов обучения и изменением learning rate
        if epoch % CHECKPOINT_EPOCH == 0:
            logging.info('Сохранение результатов обучения...')
            save_results_to_file(model, os.path.join(CHECKPOINTS_FILEPATH, f'iter_{epoch}_{WORD_REPRESENTATION}_model_params.pt'),\
                                 train_states, validation_states)
            torch.save(model, MODEL_SAVE_FILEPATH)
        # Реализация нелинейного расписания изменения learning rate
        if epoch == 15:
            LEARNING_RATE = 5e-5
            logging.info(f'Изменение скорости обучения на эпохе {epoch}. Новая скорость обучения: {LEARNING_RATE}')
            optimizer = optim.AdamW(model.parameters(), 5e-5, weight_decay=WEIGHT_DECAY)
        if epoch == 25:
            LEARNING_RATE = 1e-5
            logging.info(f'Изменение скорости обучения на эпохе {epoch}. Новая скорость обучения: {LEARNING_RATE}')
            optimizer = optim.AdamW(model.parameters(), LEARNING_RATE, weight_decay=WEIGHT_DECAY)

except KeyboardInterrupt:
    logging.info('Принудительная остановка.')
logging.info('Сохранение результатов обучения...')
save_results_to_file(model, MODEL_SAVE_FILEPATH, train_states, validation_states)