import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):
    """
    Обучаемое позиционное кодирование.  

    Parameters:
        embed_dim (int): Размерность эмбеддингов.
        max_seq_len (int): Максимальная длина последовательности.
        batch_first (bool): Указывает порядок батча и временного измерения:
            * True → [B, N, D]
            * False → [N, B, D]
    """
    def __init__(self, embed_dim:int, max_seq_len:int, batch_first:bool):
        super().__init__()
        self.batch_first = batch_first
        self.pos_embedings = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, x:torch.Tensor, key_padding_mask=None):
        """
        Добавляет позиционное кодирование к входному тензору.

        Parameters:
            x (torch.Tensor): Тензор токенов с размером [B, N, D] или [N, B, D]

        Returns:
            torch.Tensor: Сумма исходного тензора и позиционного эмбеддинга.
            Размерность совпадает со входом.
        """
        # x [B, N, D]
        if self.batch_first:
            seq_len = x.size(1)
            pos_idx = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, S]
        else:
            seq_len = x.size(0)
            pos_idx = torch.arange(seq_len, device=x.device).unsqueeze(1)  # [S, 1]
        
        pos_embed = self.pos_embedings(pos_idx) # [1, S, D] | [S, 1, D]

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(-1)
            pos_embed = pos_embed * (~mask).float()
        
        return x + pos_embed


class EncoderBlock(nn.Module):
    def __init__(self, attention_dim, num_heads, dropout, dim_encoder_ff, bias:bool=True, batch_first:bool=True):
        """
        Блок энкодера с многоловым вниманием и полносвязной сетью (Feed-Forward).

        Parameters:
            attention_dim (int): Размерность внимания.
            num_heads (int): Количество голов в MultiHeadAttention.
            dropout (float): Вероятность dropout для регуляризации.
            dim_encoder_ff (int): Размерность скрытого слоя Feed-Forward.
            bias (bool, default=True): Использовать смещение в линейных слоях.
            batch_first (bool, default=True): Указывает порядок батча.
        """
        super().__init__()
        self.query_ff = nn.Linear(attention_dim, attention_dim, bias)
        self.key_ff = nn.Linear(attention_dim, attention_dim, bias)
        self.value_ff = nn.Linear(attention_dim, attention_dim, bias)

        self.norm1 = nn.LayerNorm(attention_dim)
        self.attention = nn.MultiheadAttention(attention_dim, num_heads, dropout, bias=bias, batch_first=batch_first)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(attention_dim)

        self.encoder_ff =  nn.Sequential(nn.Linear(attention_dim, dim_encoder_ff, bias), nn.ReLU(), nn.Dropout(dropout),\
                                         nn.Linear(dim_encoder_ff, attention_dim, bias))
    
    def forward(self, x, key_padding_mask):
        """
        Выполняет один слой энкодера: Attention + Feed-Forward.

        Parameters:
            x (torch.Tensor): Тензор входных данных [B, N, D] или [N, B, D].
            key_padding_mask (torch.Tensor): Маска паддинга для внимания.
                True → masked positions; False → unmasked.

        Returns:
            torch.Tensor: Обновлённый тензор после Attention и Feed-Forward.
            Размерность совпадает с входом.
        """
        x = self.norm1(x)
        query, key, value = (self.query_ff(x), self.key_ff(x), self.value_ff(x))

        attention_out, attention_out_weights = self.attention(query, key, value, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attention_out) # residual
        encoder_out = self.encoder_ff(self.norm2(x))
        return (x + encoder_out)


class MHAModel(nn.Module):
    def __init__(self, max_seq_len:int, num_embeddings:int, embedding_dim:int, attention_dim:int, num_heads:int, num_layers:int, dim_classifier_ff_hidden:int, dim_encoder_ff:int,\
                 classifiers_names_params:dict[str, int], dropout:float, temperature:float, batch_first:bool, bias:bool=True, padding_idx:int=0):
        """
        Модель Multi-Head Attention (MHA) для классификации различных признаков.
        Включает:
            - эмбеддинг токенов
            - обучаемое позиционное кодирование
            - стек из EncoderBlock слоёв
            - финальные классификаторы (для каждого признака)

        Parameters:
            max_seq_len (int): Максимальная длина последовательности.
            num_embeddings (int): Количество токенов (длина словаря).
            embedding_dim (int): Размерность векторов эмбеддингов токенов.
            attention_dim (int): Размерность векторов внимания.
            num_heads (int): Количество голов в Multi-Head Attention.
            num_layers (int): Количество EncoderBlock слоёв.
            dim_classifier_ff_hidden (int): Размер скрытого слоя финальных классификаторов.
            dim_encoder_ff (int): Размер скрытого слоя Feed-Forward внутри EncoderBlock.
            classifiers_names_params (dict[str, int]): Словарь {название признака : размер словаря}.
                Ожидается, что ключ – название признака, а значение - размерность словаря выходного класса.
            dropout (float): Вероятность dropout для регуляризации.
            temperature (float): Температура для softmax в финальных классификаторах.
            batch_first (bool, default=True): Указывает порядок батча.
            bias (bool, default=True): Использовать смещение в линейных слоях.
            padding_idx (int, default=0): Индекс паддинга для токенов.
        """
        # classifiers_names_params: ожидается словарь, где ключ - название признака, а значение - размерность словаря признака
        super().__init__()

        self.max_seq_len = max_seq_len
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_classifier_ff_hidden = dim_classifier_ff_hidden
        self.dim_encoder_ff = dim_encoder_ff
        self.classifiers_names_params = classifiers_names_params
        self.dropout = dropout
        self.temperature = temperature
        self.batch_first = batch_first
        self.bias = bias
        self.padding_idx = padding_idx

        self.embedings = nn.Embedding(num_embeddings, embedding_dim, padding_idx)
        self.positional_encoding = LearnablePositionalEncoding(embedding_dim, max_seq_len, batch_first)
        self.embed_to_encod_proj = nn.Linear(embedding_dim, attention_dim, bias)
        self.norm1 = nn.LayerNorm(attention_dim)
        self.encoder_stack = nn.ModuleList([EncoderBlock(attention_dim, num_heads, dropout, dim_encoder_ff, bias, batch_first) for _ in range(num_layers)])
        self.norm2 = nn.LayerNorm(attention_dim)

        self.final_classifiers = nn.ModuleDict({key:nn.Sequential(
            nn.Linear(attention_dim, dim_classifier_ff_hidden, bias), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_classifier_ff_hidden, value, bias))\
                for key, value in classifiers_names_params.items()})
        # self.final_classifiers = nn.ModuleDict({target_name:nn.Linear(attention_dim, target_cls) for target_name, target_cls in classifiers_names_params.items()})

    def get_hyperparams(self)->dict:
        return {
            'max_seq_len':self.max_seq_len,
            'num_embeddings':self.num_embeddings,
            'embedding_dim':self.embedding_dim,
            'attention_dim':self.attention_dim,
            'num_heads':self.num_heads,
            'num_layers':self.num_layers,
            'dim_classifier_ff_hidden':self.dim_classifier_ff_hidden,
            'dim_encoder_ff':self.dim_encoder_ff,
            'classifiers_names_params':self.classifiers_names_params,
            'dropout':self.dropout,
            'temperature':self.temperature,
            'batch_first':self.batch_first,
            'bias':self.bias,
            'padding_idx':self.padding_idx
        }

    def forward(self, x, apply_softmax:bool=False)->dict[str:torch.Tensor]:
        """
        Выполняет проход модели.

        Parameters:
            x (torch.Tensor): Тензор токенов с размером [B, S] (batch-first).
                - B: batch size
                - S: sequence length
            apply_softmax (bool, default=False): Если True, применяет softmax в финальных классификаторах.

        Returns:
            dict[str:torch.Tensor]: Словарь ключ‑значение с логитами для каждого признака.
                Каждый тензор имеет размер [B, S, num_classes_key].
        """
        # x [B, S]
        key_padding_mask = (x == self.padding_idx)
        x = self.embedings(x) # [B, S, E]
        x = self.positional_encoding(x)
        if x.size(-1) != self.attention_dim:
            x = self.embed_to_encod_proj(x) # [B, S, D]
        x = self.norm1(x)
        for layer in range(self.num_layers):
            x = self.encoder_stack[layer](x, key_padding_mask)
        # x [B, S, D]
        x = self.norm2(x)
        logits = {}
        for key, value in self.classifiers_names_params.items():
            logits[key] = self.final_classifiers[key](x) # [B, S, num_classes_key]

        if apply_softmax:
            for key, _ in logits.items():
                logits[key] = nn.functional.softmax(logits[key]/self.temperature, dim=-1)
        return logits
