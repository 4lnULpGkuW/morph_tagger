import os
import json
import torch
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
from dotenv import load_dotenv

# Импорты ваших скриптов
from model.model import MHAModel
from scripts.vocabulary import Vocabulary
from scripts.vectorizer import Vectorizer
from scripts.tokenizer import SeparatorTokenizer, BPETokenizer

# --- КОНФИГУРАЦИЯ ---
load_dotenv(dotenv_path=(Path('.') / '.env'))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME')
MODEL_CHECKPOINT = os.path.join(EXPERIMENT_NAME, 'checkpoints', 'final_tokens_model_params.pt')
TOKENIZER_JSON = os.path.join('checkpoints', 'tokenizer.json')
ABBR_JSON_PATH = os.path.join('dataset', 'abbr.json')
SOURCE_VOCAB_PATH = 'data/merged_source_vocab.json'
TARGET_VOCABS_PATH = 'data/merged_target_vocabs.json'
PROBING_SAVE_DIR = 'probing'

# Маппинг: [Ключ в abbr.json] -> [Ключ в словаре/модели]
CAT_MAP = {
    'pos': 'upos',
    'case': 'Case',
    'number': 'Number',
    'gender': 'Gender'
}

HOOK_NAMES = ['morph', 'enc1', 'enc2', 'enc3', 'enc4']

# --- МЕХАНИКА ХУКОВ ---
captured = {}

def make_hook(name):
    def hook_fn(module, input, output):
        captured[name] = output.detach()
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

        if model.word_subtokens_pos_encoding is not None:
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

# --- МЕТРИКИ ---
def compute_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return None
    return {
        'acc': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'n': int(len(y_true)),
    }

def print_results_table(title, results_dict):
    print(f"\n{'='*95}\n{title:^95}\n{'='*95}")
    print(f"{'Признак':>15} | {'Acc':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | {'N':>8}")
    print("-" * 95)

    rows = []
    for _, model_cat in CAT_MAP.items():
        r = results_dict.get(model_cat)
        if not r:
            continue
        rows.append(r)
        print(
            f"{model_cat:>15} | "
            f"{r['acc']*100:9.2f}% | "
            f"{r['precision']*100:9.2f}% | "
            f"{r['recall']*100:9.2f}% | "
            f"{r['f1']*100:9.2f}% | "
            f"{r['n']:8d}"
        )

    print("-" * 95)

    if rows:
        mean_acc = np.mean([r['acc'] for r in rows])
        mean_prec = np.mean([r['precision'] for r in rows])
        mean_rec = np.mean([r['recall'] for r in rows])
        mean_f1 = np.mean([r['f1'] for r in rows])

        print(
            f"{'СРЕДНЕЕ':>15} | "
            f"{mean_acc*100:9.2f}% | "
            f"{mean_prec*100:9.2f}% | "
            f"{mean_rec*100:9.2f}% | "
            f"{mean_f1*100:9.2f}% | "
            f"{'—':>8}"
        )

    print("=" * 95)

def select_by_mask(arr, mask):
    return arr[mask] if arr is not None else arr

# --- MAIN ---
def main():
    model = MHAModel.from_pretrained(MODEL_CHECKPOINT, DEVICE)
    model.to(DEVICE).eval()
    bpe_tokenizer = BPETokenizer.from_pretrained(TOKENIZER_JSON)

    with open(SOURCE_VOCAB_PATH, 'r', encoding='utf-8') as f:
        src_vocab = Vocabulary.from_serializable(json.load(f))
    with open(TARGET_VOCABS_PATH, 'r', encoding='utf-8') as f:
        all_trg_data = json.load(f)
        trg_vocabs = {k: Vocabulary.from_serializable(v) for k, v in all_trg_data.items()}

    vectorizer = Vectorizer(src_vocab, trg_vocabs, Vocabulary(add_bos_eos_tokens=False), 'tokens')
    handles, original_forward = register_hooks(model)

    with open(ABBR_JSON_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    X_collector = {name: [] for name in HOOK_NAMES}
    Y_collector = {cat: [] for cat in CAT_MAP.values()}
    is_abbr_flag = []
    logits_collector = {cat: [] for cat in CAT_MAP.values()}

    vocab_limit = model.tokens_embedings.num_embeddings

    for item in tqdm(test_data, desc="Inferencing"):
        words = SeparatorTokenizer.tokenize(item['sentence'])
        word_subtokens = [bpe_tokenizer.encode(w, add_special_tokens=False).tokens for w in words]

        input_ids = vectorizer.vectorize_tokens(
            model.max_word_subtokens_count,
            model.max_words_count,
            word_subtokens
        )
        input_tensor = torch.tensor([input_ids], device=DEVICE)
        input_tensor[input_tensor >= vocab_limit] = src_vocab.unk_idx

        captured.clear()
        with torch.no_grad():
            outputs = model(
                input_tensor,
                torch.zeros(
                    (1, model.max_words_count, model.max_letters_count),
                    dtype=torch.long,
                    device=DEVICE
                )
            )

        for i, t_info in enumerate(item['tokens']):
            if i >= model.max_words_count:
                break

            if 'pos' not in t_info:
                continue

            is_abbr_flag.append(t_info.get('is_target', False))

            for test_key, model_key in CAT_MAP.items():
                true_idx = trg_vocabs[model_key].get_index(t_info.get(test_key, 'none'))
                Y_collector[model_key].append(true_idx)
                logits_collector[model_key].append(outputs[model_key][0, i].cpu().numpy())

            for h_name in HOOK_NAMES:
                X_collector[h_name].append(captured[h_name][0, i].cpu().numpy())

    is_abbr_flag = np.array(is_abbr_flag, dtype=bool)
    is_not_abbr_flag = ~is_abbr_flag

    for k in X_collector:
        X_collector[k] = np.array(X_collector[k])
    for k in Y_collector:
        Y_collector[k] = np.array(Y_collector[k])

    full_results_all = {}
    full_results_abbr = {}
    full_results_nonabbr = {}

    for cat in CAT_MAP.values():
        preds = np.argmax(logits_collector[cat], axis=-1)
        full_results_all[cat] = compute_metrics(Y_collector[cat], preds)
        full_results_abbr[cat] = compute_metrics(Y_collector[cat][is_abbr_flag], preds[is_abbr_flag])
        full_results_nonabbr[cat] = compute_metrics(Y_collector[cat][is_not_abbr_flag], preds[is_not_abbr_flag])

    print_results_table("FULL MODEL: TOTAL", full_results_all)
    print_results_table("FULL MODEL: ABBREVIATIONS ONLY", full_results_abbr)
    print_results_table("FULL MODEL: WITHOUT ABBREVIATIONS", full_results_nonabbr)

    for h_name in HOOK_NAMES:
        path = os.path.join(EXPERIMENT_NAME, PROBING_SAVE_DIR, f'probing_{h_name}.joblib')
        if not os.path.exists(path):
            print(f"Skipping {h_name}: file not found at {path}")
            continue

        pkg = joblib.load(path)
        scaler = pkg['scaler']
        clfs = pkg['models']

        X_scaled = scaler.transform(X_collector[h_name])

        layer_results_all = {}
        layer_results_abbr = {}
        layer_results_nonabbr = {}

        for cat in CAT_MAP.values():
            if cat not in clfs:
                continue

            preds = clfs[cat].predict(X_scaled)

            layer_results_all[cat] = compute_metrics(Y_collector[cat], preds)
            layer_results_abbr[cat] = compute_metrics(Y_collector[cat][is_abbr_flag], preds[is_abbr_flag])
            layer_results_nonabbr[cat] = compute_metrics(Y_collector[cat][is_not_abbr_flag], preds[is_not_abbr_flag])

        print_results_table(f"PROBING LAYER: {h_name.upper()} (TOTAL)", layer_results_all)
        print_results_table(f"PROBING LAYER: {h_name.upper()} (ABBREVIATIONS)", layer_results_abbr)
        print_results_table(f"PROBING LAYER: {h_name.upper()} (WITHOUT ABBREVIATIONS)", layer_results_nonabbr)

    for h in handles:
        h.remove()
    model.forward = original_forward

if __name__ == '__main__':
    main()