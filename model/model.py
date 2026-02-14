import torch
import torch.nn as nn
import math
from model.positional_encoding import LearnablePositionalEncoding, SinusoidalPositionalEncoding, RoPE


class EncoderBlock(nn.Module):
    def __init__(self, main_attention_dim, main_num_heads, dropout, main_encoder_ff_dim, bias:bool=True, batch_first:bool=True):
        """Блок энкодера с многоголовым вниманием и Feed-Forward сетью"""
        super().__init__()
        self.query_ff = nn.Linear(main_attention_dim, main_attention_dim, bias)
        self.key_ff = nn.Linear(main_attention_dim, main_attention_dim, bias)
        self.value_ff = nn.Linear(main_attention_dim, main_attention_dim, bias)

        self.norm1 = nn.LayerNorm(main_attention_dim)
        self.attention = nn.MultiheadAttention(main_attention_dim, main_num_heads, dropout, bias=bias, batch_first=batch_first)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(main_attention_dim)

        self.encoder_ff =  nn.Sequential(
            nn.Linear(main_attention_dim, main_encoder_ff_dim, bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(main_encoder_ff_dim, main_attention_dim, bias)
        )
    
    def forward(self, x, key_padding_mask):
        """Выполняет один слой энкодера: Attention + Feed-Forward"""
        x = self.norm1(x)
        query, key, value = (self.query_ff(x), self.key_ff(x), self.value_ff(x))

        attention_out, attention_out_weights = self.attention(
            query, key, value, 
            key_padding_mask=key_padding_mask
        )

        # Обнуляем выход для padding позиций
        attention_out = attention_out * (~key_padding_mask).unsqueeze(-1).float()

        x = x + self.dropout(attention_out)
        encoder_out = self.encoder_ff(self.norm2(x))

        # Обнуляем выход для padding позиций снова
        encoder_out = encoder_out * (~key_padding_mask).unsqueeze(-1).float()

        return x + encoder_out

class MHAModel(nn.Module):
    def __init__(self, max_words_count:int, max_word_subtokens_count:int, max_letters_count:int,
                letters_num_embeddings:int, tokens_num_embeddings:int, tokens_embedding_dim:int, letters_embeddings_dim:int,
                main_attention_dim:int, main_num_heads:int, main_num_layers:int, classifier_ff_hidden_dim:int, main_encoder_ff_dim:int,
                classifiers_names_params:dict[str, int], words_pos_encoding:str, word_subtokens_pos_encoding:str,\
                letters_in_word_pos_encoding:str, rope_base:int, letters_in_word_attention_dim:int, dropout:float, temperature:float,\
                batch_first:bool, word_representation:str, init_weights:bool=True, bias:bool=True, padding_idx:int=0, device:str='cpu'):
        '''
        Модель для морфологического анализа с использованием механизма внимания.
        
        Args:
            max_words_count: Максимальное количество слов в предложении
            max_word_subtokens_count: Максимальное количество субтокенов в слове
            max_letters_count: Максимальное количество букв в слове
            letters_num_embeddings: Размер словаря для буквенных эмбеддингов
            tokens_num_embeddings: Размер словаря для токенных эмбеддингов
            tokens_embedding_dim: Размерность токенных эмбеддингов
            letters_embeddings_dim: Размерность буквенных эмбеддингов
            main_attention_dim: Размерность вектора внимания в основном энкодере
            main_num_heads: Количество голов внимания в основном энкодере
            main_num_layers: Количество слоев в основном энкодере
            classifier_ff_hidden_dim: Размерность скрытого слоя классификаторов
            main_encoder_ff_dim: Размерность скрытого слоя Feed-Forward в энкодере
            classifiers_names_params: Словарь, где ключ - название признака, а значение - размерность словаря признака
            words_pos_encoding: Тип позиционного кодирования для слов ('sin', 'learnable' или None)
            tokens_pos_encoding: Тип позиционного кодирования для токенов ('sin', 'learnable' или None)
            word_subtokens_pos_encoding: Тип позиционного кодирования для субтокенов слова ('sin', 'learnable' или None)
            letters_in_word_pos_encoding: Тип позиционного кодирования для букв в слове ('sin', 'learnable' или None)
            aggregation_moment: Момент агрегации (не используется в текущей реализации)
            letters_in_word_attention_dim: Размерность вектора внимания для букв в слове
            dropout: Вероятность dropout
            temperature (deprecated): Температура для softmax
            batch_first (deprecated): Если True, то входной тензор имеет размерность [batch, seq_len, features]
            word_representation: Способ представления слова ('tokens', 'letters' или 'both')
            init_weights: Инициализировать ли веса модели
            bias: Использовать bias в линейных слоях
            padding_idx: Индекс паддинга
        '''
        super().__init__()

        self.max_words_count = max_words_count
        self.max_word_subtokens_count = max_word_subtokens_count
        self.max_letters_count = max_letters_count
        self.letters_num_embeddings = letters_num_embeddings
        self.tokens_num_embeddings = tokens_num_embeddings
        self.tokens_embedding_dim = tokens_embedding_dim
        self.letters_embeddings_dim = letters_embeddings_dim
        self.all_letters_embeddings_dim = letters_embeddings_dim * max_letters_count
        self.main_attention_dim = main_attention_dim
        self.main_num_heads = main_num_heads
        self.main_num_layers = main_num_layers
        self.classifier_ff_hidden_dim = classifier_ff_hidden_dim
        self.main_encoder_ff_dim = main_encoder_ff_dim
        self.classifiers_names_params = classifiers_names_params
        self.words_pos_encoding_value = words_pos_encoding
        self.word_subtokens_pos_encoding_value = word_subtokens_pos_encoding
        self.letters_pos_encoding_value = letters_in_word_pos_encoding
        self.rope_base = rope_base
        self.letters_in_word_attention_dim = letters_in_word_attention_dim
        self.dropout = dropout
        self.temperature = temperature
        self.batch_first = batch_first
        self.word_representation = word_representation
        self.init_weights = init_weights
        self.bias = bias
        self.padding_idx = padding_idx
        self.device = device

        # Эмбеддинги и механизмы внимания в зависимости от представления слова
        if word_representation != 'letters':
            # Эмбеддинги токенов
            self.tokens_embedings = nn.Embedding(tokens_num_embeddings, tokens_embedding_dim, padding_idx)
            
            # Внимание для субтокенов
            self.one_over_tokens_dim_sqrt = 1 / (math.sqrt(self.tokens_embedding_dim))
            self.subtokens_q = nn.Linear(tokens_embedding_dim, tokens_embedding_dim, bias)
            self.subtokens_k = nn.Linear(tokens_embedding_dim, tokens_embedding_dim, bias)
            self.subtokens_v = nn.Linear(tokens_embedding_dim, tokens_embedding_dim, bias)
            self.subtokens_norm = nn.LayerNorm(tokens_embedding_dim)
            self.subtokens_attention_ff = nn.Sequential(
                nn.Linear(tokens_embedding_dim, tokens_embedding_dim*2), 
                nn.GELU(), 
                nn.Dropout(dropout),
                nn.Linear(tokens_embedding_dim*2, tokens_embedding_dim), 
                nn.GELU(), 
                nn.Dropout(dropout))

            # Позиционное кодировани на уровне токенов слова
            if word_subtokens_pos_encoding == 'sin':
                self.word_subtokens_pos_encoding = SinusoidalPositionalEncoding(tokens_embedding_dim, max_word_subtokens_count, device)
            elif word_subtokens_pos_encoding == 'learnable':
                self.word_subtokens_pos_encoding = LearnablePositionalEncoding(tokens_embedding_dim, max_word_subtokens_count, padding_idx, device)
            elif word_subtokens_pos_encoding == 'rope':
                self.word_subtokens_pos_encoding = RoPE(tokens_embedding_dim, max_word_subtokens_count, rope_base, device)
            else:
                self.word_subtokens_pos_encoding = None

            # Позиционное кодирование на уровне слов
            if words_pos_encoding == 'sin':
                self.words_pos_encoding = SinusoidalPositionalEncoding(tokens_embedding_dim, max_words_count, device)
            elif words_pos_encoding == 'learnable':
                self.words_pos_encoding = LearnablePositionalEncoding(tokens_embedding_dim, max_words_count, padding_idx, device)
            else:
                self.words_pos_encoding = None            

        if word_representation != 'tokens':
            # Эмбеддинги букв
            self.letters_embeddings = nn.Embedding(letters_num_embeddings, letters_embeddings_dim, padding_idx)
            
            # Внимание для букв
            self.one_over_letters_dim_sqrt = 1 / (math.sqrt(self.letters_embeddings_dim))
            self.letters_q = nn.Linear(letters_embeddings_dim, letters_in_word_attention_dim, bias)
            self.letters_k = nn.Linear(letters_embeddings_dim, letters_in_word_attention_dim, bias)
            self.letters_v = nn.Linear(letters_embeddings_dim, letters_in_word_attention_dim, bias)
            self.letters_norm = nn.LayerNorm(letters_in_word_attention_dim)
            self.letters_attention_ff = nn.Sequential(
                nn.Linear(letters_in_word_attention_dim, letters_in_word_attention_dim*2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(letters_in_word_attention_dim*2, letters_in_word_attention_dim))
            
            # Char FF для буквенных представлений (см статью RNN Morph за авторством Ильи Гусева)
            self.char_ff = nn.Sequential(
                nn.Linear(self.all_letters_embeddings_dim, self.main_attention_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.main_attention_dim, self.all_letters_embeddings_dim),
                nn.GELU(),
                nn.Dropout(dropout))

            # Позиционное кодирование для букв
            if letters_in_word_pos_encoding == 'sin':
                self.letters_in_word_pos_encoding = SinusoidalPositionalEncoding(letters_embeddings_dim, max_letters_count, device)
            elif letters_in_word_pos_encoding == 'learnable':
                self.letters_in_word_pos_encoding = LearnablePositionalEncoding(letters_embeddings_dim, max_letters_count, padding_idx, device)
            elif letters_in_word_pos_encoding == 'rope':
                self.letters_in_word_pos_encoding = RoPE(letters_embeddings_dim, max_letters_count, rope_base, device)
            else:
                self.letters_in_word_pos_encoding = None

        # Сеть "эксперт" для определения коэффициентов при аггрегации субтокенов слова
        self.aggregation_ff = nn.Sequential(nn.Linear(tokens_embedding_dim, 2*tokens_embedding_dim),
                                            nn.GELU(),
                                            nn.Dropout(dropout),
                                            nn.Linear(2*tokens_embedding_dim, 1))

        # Проекционный слой для приведения к main_attention_dim
        if word_representation == 'both':
            self.embed_to_encod_proj = nn.Linear(self.all_letters_embeddings_dim + self.tokens_embedding_dim, main_attention_dim, bias)
        elif word_representation == 'tokens':
            self.embed_to_encod_proj = nn.Linear(self.tokens_embedding_dim, main_attention_dim, bias)
        else:
            self.embed_to_encod_proj = nn.Linear(self.all_letters_embeddings_dim, main_attention_dim, bias)

        # Стек энкодеров
        self.encoder_stack = nn.ModuleList([
            EncoderBlock(main_attention_dim, main_num_heads, dropout, main_encoder_ff_dim, bias, batch_first) 
            for _ in range(main_num_layers)])
        self.norm = nn.LayerNorm(main_attention_dim)

        # Классификаторы
        self.final_classifiers = nn.ModuleDict({
            key: nn.Sequential(
                nn.Linear(main_attention_dim, classifier_ff_hidden_dim, bias), 
                nn.GELU(), 
                nn.Dropout(dropout), 
                nn.Linear(classifier_ff_hidden_dim, value, bias))
                for key, value in classifiers_names_params.items()})

        if init_weights:
            print('Используется "умная" инициализация весов модели')
            self._init_weights()

    def _init_weights(self):
        """Инициализация весов модели"""
        # Инициализация эмбеддингов
        if self.word_representation != 'letters':
            nn.init.normal_(self.tokens_embedings.weight, mean=0.0, std=0.02)
            if self.padding_idx is not None:
                with torch.no_grad():
                    self.tokens_embedings.weight[self.padding_idx].fill_(0)
            
            # Инициализация слоев внимания субтокенов
            for layer in [self.subtokens_q, self.subtokens_k, self.subtokens_v]:
                nn.init.kaiming_uniform_(layer.weight)
                if self.bias:
                    nn.init.constant_(layer.bias, 0.0)
            
            # Инициализация FF слоев субтокенов
            for layer in self.subtokens_attention_ff:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight)
                    if self.bias:
                        nn.init.constant_(layer.bias, 0.0)

        if self.word_representation != 'tokens':
            nn.init.normal_(self.letters_embeddings.weight, mean=0.0, std=0.02)
            if self.padding_idx is not None:
                with torch.no_grad():
                    self.letters_embeddings.weight[self.padding_idx].fill_(0)
            
            # Инициализация слоев внимания букв
            for layer in [self.letters_q, self.letters_k, self.letters_v]:
                nn.init.kaiming_uniform_(layer.weight)
                if self.bias:
                    nn.init.constant_(layer.bias, 0.0)
            
            # Инициализация FF слоев букв
            for layer in self.letters_attention_ff:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight)
                    if self.bias:
                        nn.init.constant_(layer.bias, 0.0)
            
            for layer in self.char_ff:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight)
                    if self.bias:
                        nn.init.constant_(layer.bias, 0.0)

        # Инициализация проекционного слоя
        nn.init.kaiming_uniform_(self.embed_to_encod_proj.weight)
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
        for layer in [encoder.query_ff, encoder.key_ff, encoder.value_ff]:
            nn.init.kaiming_uniform_(layer.weight)
            if self.bias:
                nn.init.constant_(layer.bias, 0.0)
        
        for layer in encoder.encoder_ff:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                if self.bias:
                    nn.init.constant_(layer.bias, 0.0)

    def _init_classifier_weights(self, classifier):
        """Инициализация весов классификатора"""
        for layer in classifier:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                if self.bias:
                    nn.init.constant_(layer.bias, 0.0)

    def get_hyperparams(self)->dict:
        """Возвращает гиперпараметры модели в виде словаря
        
        Returns:
            Словарь с гиперпараметрами модели
        """
        return {
            'max_words_count': self.max_words_count,
            'max_word_subtokens_count': self.max_word_subtokens_count,
            'max_letters_count': self.max_letters_count,
            'letters_num_embeddings': self.letters_num_embeddings,
            'tokens_num_embeddings': self.tokens_num_embeddings,
            'tokens_embedding_dim': self.tokens_embedding_dim,
            'letters_embeddings_dim': self.letters_embeddings_dim,
            'main_attention_dim': self.main_attention_dim,
            'main_num_heads': self.main_num_heads,
            'main_num_layers': self.main_num_layers,
            'classifier_ff_hidden_dim': self.classifier_ff_hidden_dim,
            'main_encoder_ff_dim': self.main_encoder_ff_dim,
            'classifiers_names_params': self.classifiers_names_params,
            'words_pos_encoding': self.words_pos_encoding_value,
            'word_subtokens_pos_encoding': self.word_subtokens_pos_encoding_value,
            'letters_in_word_pos_encoding': self.letters_pos_encoding_value,
            'rope_base' : self.rope_base,
            'letters_in_word_attention_dim': self.letters_in_word_attention_dim,
            'dropout': self.dropout,
            'temperature': self.temperature,
            'batch_first': self.batch_first,
            'word_representation': self.word_representation,
            'init_weights': self.init_weights,
            'bias': self.bias,
            'padding_idx': self.padding_idx
        }
    
    def subtokens_attention(self, tokens:torch.Tensor, tokens_padding_mask:torch.Tensor):
        """Механизм внимания между субтокенами слова"""
        # tokens [B*S, T, E]
        residual_tokens = tokens
        subtokens_q, subtokens_k, subtokens_v = (
            self.subtokens_q(tokens),
            self.subtokens_k(tokens),
            self.subtokens_v(tokens))
        
        if self.word_subtokens_pos_encoding_value == 'rope':
            subtokens_q = self.word_subtokens_pos_encoding(subtokens_q)
            subtokens_k = self.word_subtokens_pos_encoding(subtokens_k)
        
        # Вычисление scores внимания
        score = torch.matmul(subtokens_q, subtokens_k.transpose(-2, -1)) * self.one_over_tokens_dim_sqrt
        
        mask = tokens_padding_mask.unsqueeze(1).expand(-1, tokens.size(1), -1)
        score = score.masked_fill(mask, -1e8)
        
        score = nn.functional.softmax(score, dim=-1)
        tokens = torch.matmul(score, subtokens_v)

        # Residual connection + norm + masking
        tokens = tokens + residual_tokens
        tokens = tokens * (~tokens_padding_mask).unsqueeze(-1).float()
        tokens = self.subtokens_norm(tokens)

        # FF layer + residual
        residual_output = tokens
        tokens = self.subtokens_attention_ff(tokens)
        tokens = tokens + residual_output
        output = tokens * (~tokens_padding_mask).unsqueeze(-1).float()
        return output

    def letters_in_one_word_attention(self, letters:torch.Tensor, letters_padding_mask:torch.Tensor):
        """Механизм внимания между буквами одного слова"""
        # letters [B*S, L, Le]
        residual_letters = letters  # Добавлено для residual connection
        letters_q, letters_k, letters_v = (
            self.letters_q(letters), 
            self.letters_k(letters), 
            self.letters_v(letters))
        
        if self.letters_pos_encoding_value == 'rope':
            letters_q = self.letters_in_word_pos_encoding(letters_q)
            letters_k = self.letters_in_word_pos_encoding(letters_k)
        
        # attention scores
        score = torch.matmul(letters_q, letters_k.transpose(-2, -1)) * self.one_over_letters_dim_sqrt
        
        # Создание корректной маски для матрицы внимания
        mask = letters_padding_mask.unsqueeze(1).expand(-1, letters.size(1), -1)
        score = score.masked_fill(mask, -1e8)
        score = nn.functional.softmax(score, dim=-1)
        
        output = torch.matmul(score, letters_v)
        
        # Residual connection
        output = output + residual_letters
        output = output * (~letters_padding_mask).unsqueeze(-1).float()
        output = self.letters_norm(output)
        
        # FF layer with residual
        residual_output = output
        output = self.letters_attention_ff(output)
        output = output + residual_output
        output = output * (~letters_padding_mask).unsqueeze(-1).float()
        
        return output

    def forward(self, tokens:torch.Tensor, letters:torch.Tensor, apply_softmax:bool = False) -> dict[str, torch.Tensor]:
            # tokens [B, S, T], letters [B, S, L]
            
            # Создание масок
            tokens_key_padding_mask = (tokens == self.padding_idx)  # [B, S, T]
            letters_padding_mask = (letters == self.padding_idx)    # [B, S, L]
            words_padding_mask = tokens_key_padding_mask.all(dim=-1)  # [B, S] - слово padding, если все его токены padding
            
            if self.word_representation != 'letters':
                # Эмбеддинг токенов
                tokens_embed = self.tokens_embedings(tokens)  # [B, S, T, Et]
                
                # reshape для обратной совместимости с ранее написанными методами
                B, S, T, Et = tokens_embed.size()
                tokens_embed = tokens_embed.reshape(B*S, T, Et)
                tokens_key_padding_mask_flat = tokens_key_padding_mask.reshape(B*S, T)

                # Позиционное кодирование на уровне субтокенов
                if (self.word_subtokens_pos_encoding is not None) and (self.word_subtokens_pos_encoding_value != 'rope'):
                    tokens_embed = self.word_subtokens_pos_encoding(tokens_embed, tokens_key_padding_mask_flat)

                # Внимание между субтокенами
                tokens_processed = self.subtokens_attention(tokens_embed, tokens_key_padding_mask_flat)
                
                # Агрегация через оперделение весов субтокенов слова
                aggregation_scores = self.aggregation_ff(tokens_processed) * self.one_over_tokens_dim_sqrt * 0.5 # Уменьшаем абсолютное значение выходов, чтобы сгладить значения после softmax
                aggregation_scores = nn.functional.softmax(aggregation_scores, dim=-2) # [B, S, T, 1]
                tokens_processed = tokens_processed * aggregation_scores # Поэлементное умножение с broadcasting
                # Агрегация (суммирование по субтокенам слова)
                tokens_processed = tokens_processed.sum(dim=1)  # [B*S, Et]
                tokens_processed = tokens_processed.reshape(B, S, Et)  # [B, S, Et]

                # Позиционное кодирование на уровне слов
                if self.words_pos_encoding is not None:
                    tokens_processed = self.words_pos_encoding(tokens_processed, words_padding_mask)
                
                x = tokens_processed

            if self.word_representation != 'tokens':
                # Эмбеддинг букв
                letters_embed = self.letters_embeddings(letters)  # [B, S, L, Le]
                B, S, L, E = letters_embed.size()
                
                # Позиционное кодирование для букв
                letters_padding_mask_flat = letters_padding_mask.reshape(B*S, L)
                letters_embed = letters_embed.reshape(B*S, L, E)
                
                if (self.letters_in_word_pos_encoding is not None) and (self.letters_pos_encoding_value != 'rope'):
                    letters_embed = self.letters_in_word_pos_encoding(letters_embed, letters_padding_mask_flat)

                # Внимание для букв
                letters_processed = self.letters_in_one_word_attention(letters_embed, letters_padding_mask_flat)
                
                # Конкатенация буквенных представлений
                letters_processed = letters_processed.reshape(B, S, L*E)
                
                # FF слой для букв
                letters_ff = self.char_ff(letters_processed)
                letters_aggregated = letters_processed + letters_ff  # Residual connection
                letters_aggregated = letters_aggregated * (~words_padding_mask).unsqueeze(-1).float()

            # Объединение представлений в зависимости от word_representation
            if self.word_representation == 'both':
                x = torch.cat([x, letters_aggregated], dim=-1)  # [B, S, Et + L*E]
            elif self.word_representation == 'letters':
                x = letters_aggregated

            # Проекция к размерности main_attention_dim
            if x.size(-1) != self.main_attention_dim:
                x = self.embed_to_encod_proj(x)  # [B, S, D]

            # Проход через стек энкодеров
            for encoder in self.encoder_stack:
                x = encoder(x, words_padding_mask)

            # Финальная нормализация
            x = self.norm(x)
            
            # Классификация
            logits = {}
            for key in self.classifiers_names_params:
                logits[key] = self.final_classifiers[key](x)

            if apply_softmax:
                for key in logits:
                    logits[key] = nn.functional.softmax(logits[key] / self.temperature, dim=-1)

            return logits