import torch
import torch.nn as nn
import math
from model.positional_encoding import LearnablePositionalEncoding, SinusoidalPositionalEncoding


class EncoderBlock(nn.Module):
    def __init__(self, main_attention_dim, main_num_heads, dropout, main_encoder_ff_dim, bias:bool=True, batch_first:bool=True):
        """
        Блок энкодера с многоловым вниманием и полносвязной сетью (Feed-Forward).

        Parameters:
            main_attention_dim (int): Размерность внимания.
            main_num_heads (int): Количество голов в MultiHeadAttention.
            dropout (float): Вероятность dropout для регуляризации.
            main_encoder_ff_dim (int): Размерность скрытого слоя Feed-Forward.
            bias (bool, default=True): Использовать смещение в линейных слоях.
            batch_first (bool, default=True): Указывает порядок батча.
        """
        super().__init__()
        self.query_ff = nn.Linear(main_attention_dim, main_attention_dim, bias)
        self.key_ff = nn.Linear(main_attention_dim, main_attention_dim, bias)
        self.value_ff = nn.Linear(main_attention_dim, main_attention_dim, bias)

        self.norm1 = nn.LayerNorm(main_attention_dim)
        self.attention = nn.MultiheadAttention(main_attention_dim, main_num_heads, dropout, bias=bias, batch_first=batch_first)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(main_attention_dim)

        self.encoder_ff =  nn.Sequential(nn.Linear(main_attention_dim, main_encoder_ff_dim, bias), nn.GELU(), nn.Dropout(dropout),\
                                         nn.Linear(main_encoder_ff_dim, main_attention_dim, bias))
    
    def forward(self, x, key_padding_mask):
        """
        Выполняет один слой энкодера: Attention + Feed-Forward.

        Parameters:
            x (torch.Tensor): Тензор входных данных [B, N, D] или [N, B, D].
            key_padding_mask (torch.Tensor): Маска паддинга для внимания.
                True -> masked positions; False -> unmasked.

        Returns:
            torch.Tensor: Обновлённый тензор после Attention и Feed-Forward.
            Размерность совпадает со входом.
        """
        x = self.norm1(x)
        query, key, value = (self.query_ff(x), self.key_ff(x), self.value_ff(x))

        attention_out, attention_out_weights = self.attention(query, key, value, key_padding_mask=key_padding_mask)

        attention_out = attention_out * (~key_padding_mask).unsqueeze(-1).float()

        x = x + self.dropout(attention_out)
        encoder_out = self.encoder_ff(self.norm2(x))

        encoder_out = encoder_out * (~key_padding_mask).unsqueeze(-1).float()

        return (x + encoder_out)


class MHAModel(nn.Module):
    def __init__(self, max_words_count:int, max_tokens_count:int, max_word_subtokens_count:int, max_letters_count:int,\
                letters_num_embeddings:int, tokens_num_embeddings:int, tokens_embedding_dim:int, letters_embeddings_dim:int,\
                main_attention_dim:int, main_num_heads:int, main_num_layers:int, classifier_ff_hidden_dim:int, main_encoder_ff_dim:int,\
                classifiers_names_params:dict[str, int], words_pos_encoding:str, tokens_pos_encoding:str, word_subtokens_pos_encoding:str, letters_in_word_pos_encoding:str,\
                subtokens_aggregation:str, aggregation_moment:str, letters_in_word_attention_dim:int,\
                dropout:float, temperature:float, batch_first:bool, word_representation:str, init_weights:bool=True, bias:bool=True, padding_idx:int=0):
        '''
        words_pos_encoding (None | sin | learnable) - позиционное кодирование на уровне СЛОВ ПРЕДЛОЖЕНИЯ
        tokens_pos_encoding (None | sin | learnable) - позиционное кодирование на уровне ТОКЕНОВ ПРЕДЛОЖЕНИЯ
        word_subtokens_pos_encoding (None | sin | learnable) - позиционное кодирование на уровне ТОКЕНОВ ОДНОГО СЛОВА (sin пока не реализован)
        subtokens_aggregation (sum | mean) - способ агрегации субтокенов одного слова
        aggregation_moment (early | late) - агрегация до MHA или после MHA
        classifiers_names_params: ожидается словарь, где ключ - название признака, а значение - размерность словаря признака
        '''
        super().__init__()

        self.max_words_count = max_words_count
        self.max_tokens_count = max_tokens_count
        self.max_word_subtokens_count = max_word_subtokens_count
        self.max_letters_count = max_letters_count
        self.letters_num_embeddings = letters_num_embeddings
        self.tokens_num_embeddings = tokens_num_embeddings
        self.tokens_embedding_dim = tokens_embedding_dim
        self.letters_embeddings_dim = letters_embeddings_dim
        self.all_letters_embeddings_dim = letters_embeddings_dim*max_letters_count
        self.main_attention_dim = main_attention_dim
        self.main_num_heads = main_num_heads
        self.main_num_layers = main_num_layers
        self.classifier_ff_hidden_dim = classifier_ff_hidden_dim
        self.main_encoder_ff_dim = main_encoder_ff_dim
        self.classifiers_names_params = classifiers_names_params
        self.words_pos_encoding_value = words_pos_encoding
        self.tokens_pos_encoding_value = tokens_pos_encoding
        self.word_subtokens_pos_encoding_value = word_subtokens_pos_encoding
        self.letters_pos_encoding_value = letters_in_word_pos_encoding
        self.subtokens_aggregation = subtokens_aggregation
        self.aggregation_moment = aggregation_moment
        self.letters_in_word_attention_dim = letters_in_word_attention_dim
        self.dropout = dropout
        self.temperature = temperature
        self.batch_first = batch_first
        self.word_representation = word_representation
        self.init_weights = init_weights
        self.bias = bias
        self.padding_idx = padding_idx

        if word_representation != 'letters':
            # Эмбединги токенов входного предложения
            self.tokens_embedings = nn.Embedding(tokens_num_embeddings, tokens_embedding_dim, padding_idx)
            # Определение матриц механизма внимания для агрегации субтокенов
            if subtokens_aggregation == 'attention':
                # При использовании разделения внимания, маска аггрегации должна иметь различный размер
                self.one_over_tokens_dim_sqrt = 1 / (math.sqrt(self.tokens_embedding_dim))
                self.aggregation_q = nn.Linear(tokens_embedding_dim, tokens_embedding_dim, bias)
                self.aggregation_k = nn.Linear(tokens_embedding_dim, tokens_embedding_dim, bias)

            # Блок с определением позиционного кодирования на различных уровнях
            # Позиционное кодирование на уровне слов
            if words_pos_encoding == 'sin':
                self.words_pos_encoding = SinusoidalPositionalEncoding(tokens_embedding_dim, max_words_count, batch_first)
            elif words_pos_encoding == 'learnable':
                self.words_pos_encoding = LearnablePositionalEncoding(tokens_embedding_dim, max_words_count, batch_first, padding_idx)
            else:
                self.words_pos_encoding = None
            # Позиционное кодирование на уровне токенов
            if tokens_pos_encoding == 'sin':
                self.tokens_pos_encoding = SinusoidalPositionalEncoding(tokens_embedding_dim, max_tokens_count, batch_first)
            elif tokens_pos_encoding == 'learnable':
                self.tokens_pos_encoding = LearnablePositionalEncoding(tokens_embedding_dim, max_tokens_count, batch_first, padding_idx)
            else:
                self.tokens_pos_encoding = None
            # Позиционное кодирование на уровне субтокенов слова
            if word_subtokens_pos_encoding == 'sin':
                raise ValueError('Синусоидальное позиционное кодирование для субтокенов слова пока не реализовано')
            elif word_subtokens_pos_encoding == 'learnable':
                self.word_subtokens_pos_encoding = nn.Embedding(max_word_subtokens_count, tokens_embedding_dim, padding_idx)
            else:
                self.word_subtokens_pos_encoding = None


        if word_representation != 'tokens':
            # Матрицы внимания для букв слова
            self.one_over_letters_dim_sqrt = 1 / (math.sqrt(self.letters_embeddings_dim))
            self.letters_q = nn.Linear(letters_embeddings_dim, letters_in_word_attention_dim, bias)
            self.letters_k = nn.Linear(letters_embeddings_dim, letters_in_word_attention_dim, bias)
            self.letters_v = nn.Linear(letters_embeddings_dim, letters_in_word_attention_dim, bias)
            self.letters_attention_ff = nn.Sequential(nn.Linear(letters_in_word_attention_dim, letters_in_word_attention_dim*2), nn.GELU(), 
                                                      nn.Dropout(dropout), nn.Linear(letters_in_word_attention_dim*2, letters_embeddings_dim))
            # Полносвязная сеть для обработки векторов буквенных представлений
            self.char_ff = nn.Sequential(nn.Linear(self.all_letters_embeddings_dim, self.main_attention_dim), nn.GELU(), nn.Dropout(dropout),\
                                        nn.Linear(self.main_attention_dim, self.all_letters_embeddings_dim), nn.GELU(), nn.Dropout(dropout))
            # Эмбединги букв входного предложения
            self.letters_embeddings = nn.Embedding(letters_num_embeddings, letters_embeddings_dim, padding_idx)
            # Позиционное кодирование на уровне букв слова
            if letters_in_word_pos_encoding == 'sin':
                self.letters_in_word_pos_encoding = SinusoidalPositionalEncoding(letters_embeddings_dim, max_letters_count, batch_first)
            elif letters_in_word_pos_encoding == 'learnable':
                self.letters_in_word_pos_encoding = LearnablePositionalEncoding(letters_embeddings_dim, max_letters_count, batch_first, padding_idx)
            else:
                self.letters_in_word_pos_encoding = None


        if word_representation == 'both':
            # Используется, если tokens_embedding_dim + (max_letters_count * letters_embeddings_dim) < main_attention_dim
            self.embed_to_encod_proj = nn.Linear(self.all_letters_embeddings_dim + self.tokens_embedding_dim, main_attention_dim, bias)
        elif word_representation == 'tokens':
            self.embed_to_encod_proj = nn.Linear(self.tokens_embedding_dim, main_attention_dim, bias)
        else:
            self.embed_to_encod_proj = nn.Linear(self.all_letters_embeddings_dim, main_attention_dim, bias)

        self.encoder_stack = nn.ModuleList([EncoderBlock(main_attention_dim, main_num_heads, dropout, main_encoder_ff_dim, bias, batch_first) for _ in range(main_num_layers)])
        self.norm = nn.LayerNorm(main_attention_dim)

        self.final_classifiers = nn.ModuleDict({key:nn.Sequential(
            nn.Linear(main_attention_dim, classifier_ff_hidden_dim, bias), nn.GELU(), nn.Dropout(dropout), nn.Linear(classifier_ff_hidden_dim, value, bias))\
                for key, value in classifiers_names_params.items()})
        if init_weights:
            print('using weights initialisation')
            self._init_weights()

    def _init_weights(self):
        """
        Инициализация весов модели
        """
        if self.word_representation != 'letters':
            # Инициализация эмбеддингов
            nn.init.normal_(self.tokens_embedings.weight, mean=0.0, std=0.02)
            if self.padding_idx is not None:
                with torch.no_grad():
                    self.tokens_embedings.weight[self.padding_idx].fill_(0)

            # Инициализация позиционных эмбеддингов
            if self.words_pos_encoding_value == 'learnable':
                nn.init.kaiming_normal_(self.words_pos_encoding.pos_embedings.weight)
            if self.tokens_pos_encoding_value == 'learnable':
                nn.init.kaiming_normal_(self.tokens_pos_encoding.pos_embedings.weight)
            if self.word_subtokens_pos_encoding_value == 'learnable':
                nn.init.kaiming_normal_(self.word_subtokens_pos_encoding.weight)

            if self.subtokens_aggregation == 'attention':
                nn.init.kaiming_uniform_(self.aggregation_q.weight)
                nn.init.kaiming_uniform_(self.aggregation_k.weight)
                if self.bias:
                    nn.init.constant_(self.aggregation_q.bias, 0.0)
                    nn.init.constant_(self.aggregation_k.bias, 0.0)

        if self.word_representation != 'tokens':
            nn.init.normal_(self.letters_embeddings.weight, mean=0.0, std=0.02)
            if self.padding_idx is not None:
                with torch.no_grad():
                    self.letters_embeddings.weight[self.padding_idx].fill_(0)

        # Инициализация проекционного слоя
        nn.init.kaiming_uniform_(self.embed_to_encod_proj.weight)
        if self.bias:
            nn.init.constant_(self.embed_to_encod_proj.bias, 0.0)

        for encoder in self.encoder_stack:
            self._init_encoder_weights(encoder)

        # Инициализация классификаторов
        for classifier in self.final_classifiers.values():
            self._init_classifier_weights(classifier)

    def _init_encoder_weights(self, encoder):
        """Инициализация весов энкодера"""
        # Линейные слои внимания
        nn.init.kaiming_uniform_(encoder.query_ff.weight)
        nn.init.kaiming_uniform_(encoder.key_ff.weight)
        nn.init.kaiming_uniform_(encoder.value_ff.weight)
        
        if self.bias:
            nn.init.constant_(encoder.query_ff.bias, 0.0)
            nn.init.constant_(encoder.key_ff.bias, 0.0)
            nn.init.constant_(encoder.value_ff.bias, 0.0)

        # Feed-forward сеть энкодера
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
        return {
            'max_words_count':self.max_words_count,
            'max_tokens_count':self.max_tokens_count,
            'max_word_subtokens_count':self.max_word_subtokens_count,
            'max_letters_count':self.max_letters_count,
            'letters_num_embeddings':self.letters_num_embeddings,
            'tokens_num_embeddings':self.tokens_num_embeddings,
            'tokens_embedding_dim':self.tokens_embedding_dim,
            'letters_embeddings_dim':self.letters_embeddings_dim,
            'main_attention_dim':self.main_attention_dim,
            'main_num_heads':self.main_num_heads,
            'main_num_layers':self.main_num_layers,
            'classifier_ff_hidden_dim':self.classifier_ff_hidden_dim,
            'main_encoder_ff_dim':self.main_encoder_ff_dim,
            'classifiers_names_params':self.classifiers_names_params,
            'words_pos_encoding':self.words_pos_encoding_value,
            'tokens_pos_encoding':self.tokens_pos_encoding_value,
            'word_subtokens_pos_encoding':self.word_subtokens_pos_encoding_value,
            'subtokens_aggregation':self.subtokens_aggregation,
            'aggregation_moment':self.aggregation_moment,
            'dropout':self.dropout,
            'temperature':self.temperature,
            'batch_first':self.batch_first,
            'init_weights':self.init_weights,
            'bias':self.bias,
            'padding_idx':self.padding_idx
        }

    def subtokens_to_word_mask(self, subtokens_cnt:torch.Tensor, extended_S:int)->torch.Tensor:
        '''Создает матрицу для суммирования суб-токенов в слово'''
        B, S = subtokens_cnt.size() # S - истинный размер предложения, количество слов в нем
        
        # Вычисляем кумулятивную сумму для определения границ слов. Каждое значение показывает, правую границу слова в количестве токенов от начала предложения
        cumsum_cnt = torch.cumsum(subtokens_cnt, dim=1)  # [B, S]
        
        # Создаем range [0, extended_S) для всех батчей
        token_indices = torch.arange(extended_S, device=subtokens_cnt.device).unsqueeze(0).expand(B, -1)  # [B, extended_S]
        
        # Начальные индексы каждого слова в предложении.
        # Например, тензор [0, 3, 5] означает, что первое слово состоит из 3 токенов (индексы 0, 1, 2), второе слово из 2 токенов (3, 4), а третье слово - из оставшихся
        start_indices = torch.cat([torch.zeros(B, 1, device=subtokens_cnt.device, dtype=torch.long),cumsum_cnt[:, :-1]], dim=1)  # [B, S]
        
        # Создаем маски
        start_expanded = start_indices.unsqueeze(-1)  # [B, S, 1]
        end_expanded = start_expanded + subtokens_cnt.unsqueeze(-1)  # [B, S, 1]. Конечные индексы каждого слова в предложении
        token_indices_expanded = token_indices.unsqueeze(1)  # [B, 1, extended_S]
        
        # При сравнении происходит broadcasting каждого тензора к размерности [B, S, extended_S]
        # Логика сравнения: значения j (измерение extended_S) word_mask для слова i (измерение S) будут True, если для слова i его стартовый индекс окажется не больше значений j и значения j окажутся строго меньше его конечного индекса.
        # Таким мы получаем некоторый отрезок на числовой оси token_indices_expanded
        word_mask = (token_indices_expanded >= start_expanded) & (token_indices_expanded < end_expanded)
        word_mask = word_mask.float()  # [B, S, extended_S]
        return word_mask

    # Создание весовой маски для суммирования субтокенов одного слова
    def make_word_mask_via_attention(self, x:torch.Tensor, word_mask:torch.Tensor):
        q_x, k_x = (self.aggregation_q(x), self.aggregation_k(x))
        score = torch.bmm(q_x, k_x.transpose(1, 2)) * self.one_over_tokens_dim_sqrt # [B, extended_S, extended_S]
        # Маскирование: разрешаем внимание только между субтокенами одного слова
        attention_mask = torch.bmm(word_mask.transpose(1, 2), word_mask)  # [B, extended_S, extended_S]
        score = score.masked_fill(attention_mask == 0, -1e8)
                
        score = nn.functional.softmax(score, dim=-1)
        score_word_mask = torch.bmm(word_mask, score) # [B, S, extended_S]
        word_mask = score_word_mask * word_mask # Получаем word_mask, где для каждого слова (dim=1) содержатся весовые коэффициенты его субтокенов (dim=2)
                                                # Если субтокен не принадлежит слову, то значение весового коэффициента = 0
        return word_mask

    # Расчет внимания между буквами одного слова
    def letters_in_one_word_attention(self, letters:torch.Tensor, letters_padding_mask:torch.Tensor):
        # Принимает letters [B*S, L, Le]
        letters_q, letters_k, letters_v = (self.letters_q(letters), self.letters_k(letters), self.letters_v(letters))
        
        # attention scores
        score = torch.matmul(letters_q, letters_k.transpose(-2, -1)) * self.one_over_letters_dim_sqrt  # [B*S, L, L]
        
        letters_padding_mask_expanded = letters_padding_mask.unsqueeze(-1)  # [B*S, L, 1]
        letters_padding_mask_expanded = letters_padding_mask_expanded.expand(-1, -1, letters.size(1))  # [B*S, L, L]
        
        score = score.masked_fill(letters_padding_mask_expanded, -1e8)
        score = nn.functional.softmax(score, dim=-1) # [B*S, L, L]
        
        output = torch.matmul(score, letters_v) # [B*S, L, Le]
        
        # Обнуляем выход для padding позиций
        output = output * (~letters_padding_mask).unsqueeze(-1).float()

        # Проход через полносвязную сеть после внимания
        output = self.letters_attention_ff(output)
        
        return output

    def forward(self, x:torch.Tensor, subtokens_cnt:torch.Tensor, letters:torch.Tensor, apply_softmax:bool = False) -> dict[str, torch.Tensor]:
        # x [B, extended_S]. Ясно, что extended_S = S+K, где K - количество дополнительных субтокенов, на которое было разбито слово
        # letters [B, S, L], где L - количество символов

        # паддинг маска для субтокенов
        subtokens_key_padding_mask = (x == self.padding_idx) # [B, extended_S]
        # key_padding_mask для слов
        words_key_padding_mask = (subtokens_cnt == self.padding_idx)  # [B, S]
        # паддинг маска для символов
        letters_padding_mask = (letters == self.padding_idx)
        
        if self.word_representation != 'letters':
            # Получаем маску размером [B, S, extended_S], где для каждого слова обозначены (1 | 0) его субтокены.
            # Можно сказать, что word_mask - блочная диагональная матрица, где каждый блок это плотный вектор единиц
            word_mask = self.subtokens_to_word_mask(subtokens_cnt, x.size(1))
            # Эмбеддингs
            x = self.tokens_embedings(x)

            # Позиционное кодирование на уровне субтокенов одного слова
            if self.word_subtokens_pos_encoding_value is not None:
                # Вызывая cumsum, получаем индексы токенов в предложении. Затем обнуляем позиции, которые изначально были нулем. Затем суммируем, чтобы получить матрицу [B, extended_S] с индексами токенов в слове
                subtokens_in_word_indices = (word_mask.cumsum(dim=2) * word_mask).sum(dim=1)
                word_subtokens_embed = self.word_subtokens_pos_encoding(subtokens_in_word_indices.to(dtype=torch.long))
                x = x + word_subtokens_embed

            # Позиционное кодирование на уровне токенов предложения
            if self.tokens_pos_encoding_value is not None:
                x = self.tokens_pos_encoding(x, subtokens_key_padding_mask)

            # Маска агрегации субтокенов одного слова в целое взятием среднего
            if self.subtokens_aggregation == 'mean':
                word_mask = (word_mask / (word_mask.sum(dim=1, keepdim=True) + 1e-8)) * word_mask # Делим каждую строку на сумму элементов в ней, а затем применяем маскирование, чтобы обнулить ненулевые элементы

            # Маска агрегации субтокенов одного слова в целое при помощи механизма внимания
            if self.subtokens_aggregation == 'attention':
                word_mask = self.make_word_mask_via_attention(x, word_mask)

            # Агрегация субтокенов одного слова до MHA
            x = torch.bmm(word_mask, x)  # [B, S, E]
            # Позиционное кодирование на уровне слов
            if self.words_pos_encoding_value is not None:
                x = self.words_pos_encoding(x, words_key_padding_mask) # [B, S, E]


        if self.word_representation != 'tokens':
            # Эмбеддингs
            letters_embed = self.letters_embeddings(letters) # [B, S, L, Le]
            B, S, L, E = letters_embed.size()
            # Позиционное кодирование для букв слова
            # Используем reshape для соответсвия размерностей при использовании ранее написанных функций
            letters_padding_mask = letters_padding_mask.reshape(B*S, L)
            letters_embed = letters_embed.reshape(B*S, L, E)
            if self.letters_pos_encoding_value is not None:
                letters_embed = self.letters_in_word_pos_encoding(letters_embed, letters_padding_mask)

            # внимание для букв слова
            letters_embed = self.letters_in_one_word_attention(letters_embed, letters_padding_mask) # [B, S, L, Le]

            # Конкатенируем векторные представления букв одного слова
            letters_embed = letters_embed.reshape(B, S, L*E)

            # Проход через Char_FF слой для буквенного представления
            letters_embed = letters_embed + self.char_ff(letters_embed) # [B, S, L*Le]

        if self.word_representation == 'both':
            # Конкатенируем векторные представления букв и токенов
            x = torch.cat([x, letters_embed], dim=2) # [B, S, E + L*Le]
        elif self.word_representation == 'letters':
            x = letters_embed

        # Преобразование к размерности main_attention
        if x.size(-1) != self.main_attention_dim:
            x = self.embed_to_encod_proj(x)  # [B, S, D]
        # Проход через энкодер
        for layer in range(self.main_num_layers):
            x = self.encoder_stack[layer](x, words_key_padding_mask)

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