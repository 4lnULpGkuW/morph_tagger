import os
import gc
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from joblib import Parallel, delayed

from scripts.custom_dataset import CustomDataset
from model.model import MHAModel

load_dotenv(dotenv_path=(Path('.') / '.env'))

DATA_SAVE_FILEPATH = os.getenv('DATA_SAVE_FILEPATH')
EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_DATA_PATH = os.path.join(
    DATA_SAVE_FILEPATH, EXPERIMENT_NAME, 'dataset', 'merged_prepared_train.parquet'
)

MODEL_CHECKPOINT = os.path.join(
    DATA_SAVE_FILEPATH, EXPERIMENT_NAME, 'checkpoints', 'final_tokens_model_params.pt'
)

PROBING_SAVE_DIR = os.path.join(
    DATA_SAVE_FILEPATH, EXPERIMENT_NAME, 'probing'
)

HOOK_NAMES = ['morph', 'enc1', 'enc2', 'enc3', 'enc4']

# ====== НАСТРОЙКИ ======
MAX_ROWS = 15000              # None = full parquet
MAX_VALID_WORDS = 15000       # None = все слова
BATCH_SIZE = 32
PARALLEL_JOBS = 2
RANDOM_STATE = 42
# =======================

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


def sample_dataframe(df):
    if MAX_ROWS is None:
        return df
    return df.sample(n=MAX_ROWS, random_state=RANDOM_STATE)


def collect_data(model, loader, target_names):
    X = {name: [] for name in HOOK_NAMES}
    y = {t: [] for t in target_names}
    total = 0

    for batch in tqdm(loader, desc='Collect'):
        tokens = batch['input_ids'].to(DEVICE)

        letters = torch.zeros(
            (tokens.shape[0], model.max_words_count, model.max_letters_count),
            dtype=torch.long,
            device=DEVICE
        )

        captured.clear()
        with torch.inference_mode():
            model(tokens, letters)

        upos = batch['upos']
        if upos.dim() == 3:
            upos = upos.squeeze(0)

        valid_mask = (upos != 0)
        mask_device = valid_mask.to(DEVICE)

        for name in HOOK_NAMES:
            emb = captured[name]
            X[name].append(emb[mask_device].cpu())

        for t in target_names:
            lbl = batch[t]
            if lbl.dim() == 3:
                lbl = lbl.squeeze(0)
            y[t].append(lbl[valid_mask].cpu())

        total += valid_mask.sum().item()
        if MAX_VALID_WORDS is not None and total >= MAX_VALID_WORDS:
            break

    X = {k: torch.cat(v).numpy() for k, v in X.items()}
    y = {k: torch.cat(v).numpy() for k, v in y.items()}

    return X, y


def train_one(X_scaled, y, target_name):
    if len(np.unique(y)) < 2:
        return target_name, None, None

    clf = SGDClassifier(
        loss='log_loss',
        max_iter=1000,
        tol=1e-3,
        average=True,
        random_state=RANDOM_STATE,
        early_stopping=False
    )

    clf.fit(X_scaled, y)

    y_pred = clf.predict(X_scaled)

    metrics = {
        'acc': accuracy_score(y, y_pred),
        'f1': f1_score(y, y_pred, average='macro', zero_division=0),
        'n': len(y)
    }

    return target_name, clf, metrics


def train_stage(hook_name, X, y_dict, target_names):
    print(f'\n=== {hook_name} ===')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    results = Parallel(n_jobs=PARALLEL_JOBS)(
        delayed(train_one)(X_scaled, y_dict[t], t)
        for t in target_names
    )

    package = {
        'hook': hook_name,
        'scaler': scaler,
        'models': {},
        'metrics': {}
    }

    for t, clf, metrics in results:
        if clf is not None:
            package['models'][t] = clf
            package['metrics'][t] = metrics

    path = os.path.join(PROBING_SAVE_DIR, f'probing_{hook_name}.joblib')
    joblib.dump(package, path)

    print(f'Saved: {path}')

    del X_scaled, package
    gc.collect()


def main():
    print('Device:', DEVICE)

    os.makedirs(PROBING_SAVE_DIR, exist_ok=True)

    model = MHAModel.from_pretrained(MODEL_CHECKPOINT, DEVICE)
    model.to(DEVICE).eval()

    target_names = list(model.classifiers_names_params.keys())

    handles, original_forward = register_hooks(model)

    df = pd.read_parquet(TRAIN_DATA_PATH)
    df = sample_dataframe(df)

    dataset = CustomDataset(
        train_df=df,
        test_df=df,
        target_names=target_names,
        max_subtokens_count=model.max_word_subtokens_count,
        max_words_count=model.max_words_count,
        max_letters_count=model.max_letters_count
    )
    dataset.set_dataframe_split('train')

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    X_dict, y_dict = collect_data(model, loader, target_names)

    for hook in HOOK_NAMES:
        train_stage(hook, X_dict[hook], y_dict, target_names)

    for h in handles:
        h.remove()
    model.forward = original_forward

    print('\nDone')


if __name__ == '__main__':
    main()