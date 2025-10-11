import torch
import torch.nn as nn
from model.positional_encoding import LearnablePositionalEncoding, SinusoidalPositionalEncoding


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

        attention_out = attention_out * (~key_padding_mask).unsqueeze(-1).float()

        x = x + self.dropout(attention_out) # residual
        encoder_out = self.encoder_ff(self.norm2(x))

        encoder_out = encoder_out * (~key_padding_mask).unsqueeze(-1).float()

        return (x + encoder_out)


class MHAModel(nn.Module):
    def __init__(self, max_seq_len:int, num_embeddings:int, embedding_dim:int, attention_dim:int, num_heads:int, num_layers:int, dim_classifier_ff_hidden:int, dim_encoder_ff:int,\
                 classifiers_names_params:dict[str, int], pos_encoding:str, dropout:float, temperature:float, batch_first:bool, init_weights:bool=True, bias:bool=True, padding_idx:int=0):
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
        self.pos_encoding = pos_encoding
        self.dropout = dropout
        self.temperature = temperature
        self.batch_first = batch_first
        self.init_weights = init_weights
        self.bias = bias
        self.padding_idx = padding_idx

        self.embedings = nn.Embedding(num_embeddings, embedding_dim, padding_idx)
        if pos_encoding == 'sin':
            self.positional_encoding = SinusoidalPositionalEncoding(embedding_dim, max_seq_len, batch_first)
        elif pos_encoding == 'learnable':
            self.positional_encoding = LearnablePositionalEncoding(embedding_dim, max_seq_len, batch_first)
        else:
            raise ValueError(f'Неверное значение параметра позиционного кодирования {pos_encoding}')
        
        self.embed_to_encod_proj = nn.Linear(embedding_dim, attention_dim, bias)
        self.encoder_stack = nn.ModuleList([EncoderBlock(attention_dim, num_heads, dropout, dim_encoder_ff, bias, batch_first) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(attention_dim)

        self.final_classifiers = nn.ModuleDict({key:nn.Sequential(
            nn.Linear(attention_dim, dim_classifier_ff_hidden, bias), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_classifier_ff_hidden, value, bias))\
                for key, value in classifiers_names_params.items()})
        if init_weights:
            print('using weights initialisation')
            self._init_weights()

    def _init_weights(self):
        """
        Инициализация весов модели
        """
        # Инициализация эмбеддингов
        nn.init.kaiming_normal_(self.embedings.weight, nonlinearity='relu')
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedings.weight[self.padding_idx].fill_(0)

        # Инициализация позиционных эмбеддингов
        if self.pos_encoding == 'learnable':
            nn.init.kaiming_normal_(self.positional_encoding.pos_embedings.weight, nonlinearity='relu')

        # Инициализация проекционного слоя
        nn.init.kaiming_uniform_(self.embed_to_encod_proj.weight, nonlinearity='relu')
        if self.bias:
            nn.init.constant_(self.embed_to_encod_proj.bias, 0.0)

        # Инициализация энкодеров
        for encoder in self.encoder_stack:
            self._init_encoder_weights(encoder)

        # Инициализация классификаторов
        for classifier in self.final_classifiers.values():
            self._init_classifier_weights(classifier)

    def _init_encoder_weights(self, encoder):
        """Инициализация весов энкодера"""
        # Линейные слои внимания
        nn.init.kaiming_uniform_(encoder.query_ff.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(encoder.key_ff.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(encoder.value_ff.weight, nonlinearity='relu')
        
        if self.bias:
            nn.init.constant_(encoder.query_ff.bias, 0.0)
            nn.init.constant_(encoder.key_ff.bias, 0.0)
            nn.init.constant_(encoder.value_ff.bias, 0.0)

        # Feed-forward сеть энкодера
        for layer in encoder.encoder_ff:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if self.bias:
                    nn.init.constant_(layer.bias, 0.0)

    def _init_classifier_weights(self, classifier):
        """Инициализация весов классификатора"""
        for layer in classifier:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if self.bias:
                    nn.init.constant_(layer.bias, 0.0)

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
            'pos_encoding':self.pos_encoding,
            'dropout':self.dropout,
            'temperature':self.temperature,
            'batch_first':self.batch_first,
            'init_weights':self.init_weights,
            'bias':self.bias,
            'padding_idx':self.padding_idx
        }

    def forward(self, x, subtokens_cnt, apply_softmax: bool = False) -> dict[str, torch.Tensor]:
        # print(f'x.size() {x.size()}')
        # print(f'subtokens_cnt.size() {subtokens_cnt.size()}')

        # Переносим subtokens_cnt на тот же device, что и x
        subtokens_cnt = subtokens_cnt.to(x.device)
        
        # Сохраняем исходную маску паддинга для токенов
        key_padding_mask_tokens = (x == self.padding_idx)
        
        # Эмбеддинг и позиционное кодирование
        x = self.embedings(x)  # [B, extended_S, E]
        x = self.positional_encoding(x, key_padding_mask_tokens)

        # print(f'after positionals x.size() {x.size()}')

        # Создаем матрицу для агрегации токенов в слова
        B, extended_S, D = x.size()
        S = subtokens_cnt.size(1)
        
        # Вычисляем cumulative sum для определения границ слов
        cumsum_cnt = torch.cumsum(subtokens_cnt, dim=1)  # [B, S]
        
        # Создаем range [0, extended_S) для всех батчей
        token_indices = torch.arange(extended_S, device=x.device).unsqueeze(0).expand(B, -1)  # [B, extended_S]
        
        # ВЕКТОРИЗОВАННОЕ создание маски без циклов
        start_indices = torch.cat([
            torch.zeros(B, 1, device=x.device, dtype=torch.long),
            cumsum_cnt[:, :-1]
        ], dim=1)  # [B, S]
        
        # Создаем маску используя broadcasting
        start_expanded = start_indices.unsqueeze(-1)  # [B, S, 1]
        end_expanded = start_expanded + subtokens_cnt.unsqueeze(-1)  # [B, S, 1]
        token_indices_expanded = token_indices.unsqueeze(1)  # [B, 1, extended_S]
        
        word_mask = (token_indices_expanded >= start_expanded) & (token_indices_expanded < end_expanded)
        word_mask = word_mask.float()  # [B, S, extended_S]
        
        # Агрегируем токены в слова (суммируем)
        x = torch.bmm(word_mask, x)  # [B, S, D]

        # print(f'after magic algorithm x.size() {x.size()}')
        
        # Обновляем key_padding_mask для слов
        key_padding_mask = (subtokens_cnt == 0)  # [B, S]
        
        # Проекция
        if x.size(-1) != self.attention_dim:
            x = self.embed_to_encod_proj(x)  # [B, S, D]
        
        # Проход через энкодер
        for layer in range(self.num_layers):
            x = self.encoder_stack[layer](x, key_padding_mask)
        # Нормализация
        x = self.norm(x)
        
        # Классификация
        logits = {}
        for key, value in self.classifiers_names_params.items():
            logits[key] = self.final_classifiers[key](x)  # [B, S, num_classes_key]

        if apply_softmax:
            for key in logits:
                logits[key] = nn.functional.softmax(logits[key] / self.temperature, dim=-1)
        
        return logits