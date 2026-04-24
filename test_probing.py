import os
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from scripts.custom_dataset import CustomDataset
from model.model import MHAModel

load_dotenv(dotenv_path=(Path('.') / '.env'))

DATA_SAVE_FILEPATH = os.getenv('DATA_SAVE_FILEPATH')
EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TEST_DATA_PATH = os.path.join(
    DATA_SAVE_FILEPATH, EXPERIMENT_NAME, 'dataset', 'merged_prepared_test.parquet'
)

MODEL_CHECKPOINT = os.path.join(
    DATA_SAVE_FILEPATH, EXPERIMENT_NAME, 'checkpoints', 'final_tokens_model_params.pt'
)

PROBING_SAVE_DIR = os.path.join(
    DATA_SAVE_FILEPATH, EXPERIMENT_NAME, 'probing'
)

HOOK_NAMES = ['morph', 'enc1', 'enc2', 'enc3', 'enc4']

captured = {}


def make_hook(name):
    def hook_fn(module, input, output):
        captured[name] = output
    return hook_fn


def register_hooks(model):
    handles = []
    original_forward = model.forward

    def patched_forward(tokens, letters, **kwargs):
        tokens_key_padding_mask = (tokens == model.padding_idx)

        tokens_embed = model.tokens_embedings(tokens)
        B, S, T, Et = tokens_embed.size()
        tokens_embed_flat = tokens_embed.reshape(B * S, T, Et)
        mask_flat = tokens_key_padding_mask.reshape(B * S, T)

        if model.word_subtokens_pos_encoding is not None and model.word_subtokens_pos_encoding_value != 'rope':
            tokens_embed_flat = model.word_subtokens_pos_encoding(tokens_embed_flat, mask_flat)

        tokens_processed = model.subtokens_attention(tokens_embed_flat, mask_flat)

        agg_scores = model.aggregation_ff(tokens_processed) * model.one_over_tokens_dim_sqrt * 0.5
        agg_scores = torch.nn.functional.softmax(agg_scores, dim=-2)
        tokens_processed = (tokens_processed * agg_scores).sum(dim=1).reshape(B, S, Et)

        captured['morph'] = tokens_processed.detach()
        return original_forward(tokens, letters, **kwargs)

    model.forward = patched_forward

    for i, encoder in enumerate(model.encoder_stack):
        handle = encoder.register_forward_hook(make_hook(f'enc{i+1}'))
        handles.append(handle)

    return handles, original_forward


def collect_embeddings_and_logits(model, loader, target_names):
    X = {name: [] for name in HOOK_NAMES}
    y = {t: [] for t in target_names}
    logits_dict = {t: [] for t in target_names}

    for batch in tqdm(loader, desc='Collect'):
        tokens = batch['input_ids'].to(DEVICE)
        letters = torch.zeros(
            (tokens.shape[0], model.max_words_count, model.max_letters_count),
            dtype=torch.long,
            device=DEVICE
        )

        captured.clear()

        with torch.inference_mode():
            outputs = model(tokens, letters)

        upos = batch['upos']
        # Убираем squeeze. Маска будет [B, S]
        valid_mask = (upos != 0)
        mask_device = valid_mask.to(DEVICE)

        for name in HOOK_NAMES:
            emb = captured[name] # Форма [B, S, D]
            # Индексация [B, S, D][B, S] вернет [TotalValid, D]
            X[name].append(emb[mask_device].cpu())

        for t in target_names:
            lbl = batch[t] # Форма [B, S]
            # Убираем squeeze. Индексация [B, S][B, S] вернет [TotalValid]
            y[t].append(lbl[valid_mask].cpu())

        for t in target_names:
            logits = outputs[t] # Форма [B, S, C]
            # Убираем squeeze. Индексация [B, S, C][B, S] вернет [TotalValid, C]
            logits_dict[t].append(logits[mask_device].cpu())

    X = {k: torch.cat(v).numpy() for k, v in X.items()}
    y = {k: torch.cat(v).numpy() for k, v in y.items()}
    logits_dict = {k: torch.cat(v).numpy() for k, v in logits_dict.items()}

    return X, y, logits_dict


def compute_metrics(y_true, y_pred):
    return {
        'acc': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'n': int(len(y_true)),
    }


def print_stage_table(hook_name, stage_results, target_names):
    print('\n' + '=' * 90)
    print(f'{hook_name:^90}')
    print('=' * 90)
    print(f'{"Признак":>10} | {"Acc":>10} | {"Precision":>10} | {"Recall":>10} | {"F1":>10} | {"N":>8}')
    print('-' * 90)

    rows = []

    for t in target_names:
        r = stage_results.get(t)
        if r is None:
            continue

        rows.append(r)

        print(
            f'{t:>10} | '
            f'{r["acc"]*100:9.2f}% | '
            f'{r["precision"]*100:9.2f}% | '
            f'{r["recall"]*100:9.2f}% | '
            f'{r["f1"]*100:9.2f}% | '
            f'{r["n"]:8d}'
        )

    print('-' * 90)

    if rows:
        acc = np.mean([r['acc'] for r in rows])
        prec = np.mean([r['precision'] for r in rows])
        rec = np.mean([r['recall'] for r in rows])
        f1 = np.mean([r['f1'] for r in rows])

        print(
            f'{"СРЕДНЕЕ":>10} | '
            f'{acc*100:9.2f}% | '
            f'{prec*100:9.2f}% | '
            f'{rec*100:9.2f}% | '
            f'{f1*100:9.2f}% | '
            f'{"—":>8}'
        )

    print('=' * 90)


def main():
    print('Device:', DEVICE)

    model = MHAModel.from_pretrained(MODEL_CHECKPOINT, DEVICE)
    model.to(DEVICE).eval()

    target_names = list(model.classifiers_names_params.keys())

    handles, original_forward = register_hooks(model)

    df = pd.read_parquet(TEST_DATA_PATH)

    dataset = CustomDataset(
        train_df=df,
        test_df=df,
        target_names=target_names,
        max_subtokens_count=model.max_word_subtokens_count,
        max_words_count=model.max_words_count,
        max_letters_count=model.max_letters_count
    )
    dataset.set_dataframe_split('test')

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    X_dict, y_dict, logits_dict = collect_embeddings_and_logits(
        model, loader, target_names
    )

    for h in handles:
        h.remove()
    model.forward = original_forward

    results = {}

    # full model baseline
    results['full'] = {}
    for t in target_names:
        y_true = y_dict[t]
        logits = logits_dict[t]

        y_pred = logits.argmax(axis=-1)
        results['full'][t] = compute_metrics(y_true, y_pred)

    # probing classifiers
    for hook_name in HOOK_NAMES:
        path = os.path.join(PROBING_SAVE_DIR, f'probing_{hook_name}.joblib')

        if not os.path.exists(path):
            print(f'Нет файла: {path}')
            continue

        pkg = joblib.load(path)

        scaler = pkg['scaler']
        models = pkg['models']

        X = X_dict[hook_name]
        X_scaled = scaler.transform(X).astype(np.float32)

        results[hook_name] = {}

        for t in target_names:
            if t not in models:
                continue

            y_true = y_dict[t]
            if len(y_true) == 0:
                continue

            clf = models[t]
            y_pred = clf.predict(X_scaled)

            results[hook_name][t] = compute_metrics(y_true, y_pred)

    for hook_name in ['full'] + HOOK_NAMES:
        if hook_name in results:
            print_stage_table(hook_name, results[hook_name], target_names)

    print('\nDone')


if __name__ == '__main__':
    main()