import pandas as pd
from scripts.vocabulary import Vocabulary

class Vectorizer:
    def __init__(self, tokenizer, src_vocab: Vocabulary, trg_vocabs: dict[str: Vocabulary], letter_vocab: Vocabulary, word_representation: str, pad_idx: int = 0):
        """
        Инициализирует объект для преобразования токенов в индексы.
        
        Args:
            tokenizer: Токенизатор для обработки текста
            src_vocab (Vocabulary): Словарь для исходных токенов
            trg_vocabs (dict[str: Vocabulary]): Словари для целевых токенов
            letter_vocab (Vocabulary): Словарь для букв
            word_representation (str): Тип представления слов ('tokens', 'letters' или оба)
            pad_idx (int, optional): Индекс padding-токена. По умолчанию 0
        """
        self.tokenizer = tokenizer
        self.src_vocab = src_vocab
        self.trg_vocabs = trg_vocabs
        self.letter_vocab = letter_vocab
        self.word_representation = word_representation
        self.pad_idx = pad_idx

    def get_indices(self, tokenized_text: list[str], cw_vocab: Vocabulary, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        """
        Преобразует токены в индексы из словаря с возможным добавлением BOS и EOS токенов.
        
        Args:
            tokenized_text (list[str]): Список токенов
            cw_vocab (Vocabulary): Используемый словарь
            add_bos (bool, optional): Добавить BOS токен. По умолчанию True
            add_eos (bool, optional): Добавить EOS токен. По умолчанию True
        
        Returns:
            list[int]: Список индексов токенов
        """
        indices = []
        if add_bos:
            indices.append(cw_vocab.bos_idx)
        for token in tokenized_text:
            indices.append(cw_vocab.get_index(token))
        if add_eos:
            indices.append(cw_vocab.eos_idx)
        return indices

    def pad_sequence(self, indices: list[int], forced_max_len: int, pad_idx: int) -> list[int]:
        """
        Дополняет последовательность до заданной длины или обрезает ее.
        
        Args:
            indices (list[int]): Список индексов
            forced_max_len (int): Максимальная длина последовательности. 
                                 Если 0, используется длина исходной последовательности
            pad_idx (int): Индекс padding-токена
        
        Returns:
            list[int]: Дополненная или обрезанная последовательность
        """
        if forced_max_len > 0:
            seq_len = forced_max_len
        else:
            seq_len = len(indices)

        padded = [pad_idx] * seq_len
        padded[:min(len(indices), seq_len)] = indices[:min(len(indices), seq_len)]
        return padded

    def vectorize_tokens(self, max_tokens_count: int, max_words_count: int, source_words: list[str]) -> list[list[int]]:
        """
        Векторизует слова в токены.
        
        Args:
            max_tokens_count (int): Максимальное количество токенов на слово
            max_words_count (int): Максимальное количество слов
            source_words (list[str]): Список исходных слов
        
        Returns:
            list[list[int]]: Матрица токенов [слово][токен]
        """
        tokens = [[self.pad_idx] * max_tokens_count for _ in range(max_words_count)]
        for idx, word in enumerate(source_words):
            tokenized = self.tokenizer.encode(word).tokens
            tokens_indices = self.get_indices(tokenized, self.src_vocab, False, False)
            tokens[idx] = self.pad_sequence(tokens_indices, max_tokens_count, self.pad_idx)
        return tokens

    def vectorize_letters(self, max_letters_count: int, max_words_count: int, source_words: list[str]) -> list[list[int]]:
        """
        Векторизует слова в буквы.
        
        Args:
            max_letters_count (int): Максимальное количество букв на слово
            max_words_count (int): Максимальное количество слов
            source_words (list[str]): Список исходных слов
        
        Returns:
            list[list[int]]: Матрица букв [слово][буква]
        """
        letters = [[self.pad_idx] * max_letters_count for _ in range(max_words_count)]
        for idx, word in enumerate(source_words):
            cur_letters = list(word)
            letters_indices = self.get_indices(cur_letters, self.letter_vocab, add_bos=False, add_eos=False)
            letters[idx] = self.pad_sequence(letters_indices, max_letters_count, self.pad_idx)
        return letters

    def vectorize(self, df_row: pd.Series, target_names: list[str], max_subtokens_count: int, max_words_count: int, max_letters_count: int, add_bos_eos_tokens: bool = True) -> dict[str, list[int]]:
        """
        Основной метод векторизации, преобразует строку данных в числовые представления.
        
        Args:
            df_row (pd.Series): Строка DataFrame с данными
            target_names (list[str]): Список имен целевых столбцов
            max_subtokens_count (int): Максимальное количество токенов на слово
            max_words_count (int): Максимальное количество слов
            max_letters_count (int): Максимальное количество букв на слово
            add_bos_eos_tokens (bool, optional): Добавлять BOS/EOS токены. По умолчанию True
        
        Returns:
            dict[str, list[int]]: Словарь с векторизованными данными:
                - 'tokens': токены слов (может быть None)
                - 'letters': буквы слов (может быть None)
                - 'target': целевые векторы
        """
        source_words = df_row['source_words']
        
        if self.word_representation == 'tokens':
            tokens = self.vectorize_tokens(max_subtokens_count, max_words_count, source_words)
            letters = None
        elif self.word_representation == 'letters':
            tokens = None
            letters = self.vectorize_letters(max_letters_count, max_words_count, source_words)
        else:
            tokens = self.vectorize_tokens(max_subtokens_count, max_words_count, source_words)
            letters = self.vectorize_letters(max_letters_count, max_words_count, source_words)

        trg_vectorized = {}
        for target_name in target_names:
            trg_indices = self.get_indices(
                df_row[target_name], 
                self.trg_vocabs[target_name], 
                add_bos=add_bos_eos_tokens, 
                add_eos=add_bos_eos_tokens
            )
            trg_vectorized[target_name] = self.pad_sequence(trg_indices, max_words_count, self.pad_idx)
        
        return {
            'tokens': tokens,
            'letters': letters,
            'target': trg_vectorized
        }