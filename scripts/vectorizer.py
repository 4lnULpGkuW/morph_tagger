import torch
import numpy as np
import pandas as pd
from scripts.vocabulary import Vocabulary

class Vectorizer:
    def __init__(self, tokenizer, src_vocab:Vocabulary, trg_vocabs:dict[str:Vocabulary], letter_vocab:Vocabulary, pad_idx:int):
        """
        Инициализирует объект, который преобразует токены в индексы.

        Parameters
        ----------
        src_vocab : Vocabulary
            Словарь для исходного текста.
        trg_vocabs : dict[Vocabulary]
            Словари для целевых меток (ключи – названия целей).
        max_src_len : int
            Максимальная длина исходных последовательностей (для обучения).
        pad_idx : int
            Индекс маскировочного токена.
        """
        self.tokenizer = tokenizer
        self.src_vocab = src_vocab
        self.target_cnt = len(trg_vocabs)
        self.trg_vocabs = trg_vocabs
        self.letter_vocab = letter_vocab
        self.pad_idx = pad_idx


    def get_indices(self, tokenized_text:list[str], cw_vocab:Vocabulary, add_bos:bool=True, add_eos:bool=True)->list[int]:
        """
        Возвращает индексы токенов, с возможными добавлениями BOS и EOS.

        Parameters
        ----------
        tokenized_text : list[str]
            Токены исходного текста.
        cw_vocab : Vocabulary
            Словарь для преобразования токенов в индексы.
        add_bos : bool, default True
            Если True, добавляется индекс BOS в начало списка.
        add_eos : bool, default True
            Если True, добавляется индекс EOS в конец списка.

        Returns
        -------
        list[int]
            Индексы токенов (включая BOS/EOS по заданным флагам).
        """
        indices = []
        if add_bos:
            indices.append(cw_vocab.bos_idx)
        for token in tokenized_text:
            indices.append(cw_vocab.get_index(token))
        if add_eos:
            indices.append(cw_vocab.eos_idx)
        return indices
    
    
    def pad_sequence(self, indices:list[int], forced_max_len:int, pad_idx:int):
        """
        Паддинг последовательности до заданной длины.

        Parameters
        ----------
        indices : list[int]
            Индексы токенов.
        force_max_len : bool
            Если True, используется максимальная длина `max_src_len` для обучения,
            иначе используется текущая длина последовательности (для инференса).
        """
        # Для обучения используем максимальную длину предложения, для инференса - длину текущего предложения
        if forced_max_len > 0:
            seq_len = forced_max_len
        else:
            seq_len = len(indices)

        # Заполнение индексом маскировочного токена
        padded = [pad_idx] * seq_len
        # Заполнение индексами реальных токенов. Если количество токенов превышает заданную длину, то они просто отсекаются
        padded[:min(len(indices), seq_len)] = indices[:min(len(indices), seq_len)]
        return padded


    def vectorize(self, df_row:pd.Series, target_names:list[str], max_tokens_count:int, max_words_count:int, max_letters_count:int, add_bos_eos_tokens:bool=True)->dict[str, list[int]]:
        source_tokens = df_row['source_words']
        tokenized_source = []
        subtokens_cnt = []
        letters = [[self.pad_idx]*max_letters_count for _ in range(max_words_count)] # Изанчально заполняем паддингом
        for idx, token in enumerate(source_tokens):
            tokenized = self.tokenizer.encode(token).tokens
            # Убрать
            if len(tokenized) < 1:
                print('What!?!?')
            # Убрать
            tokenized_source.extend(tokenized)
            subtokens_cnt.append(max(len(tokenized), 1)) # Вычисляем количество субтокенов для каждого слова

            cur_letters = list(token) # Получаем список букв слова
            letters_indices = self.get_indices(cur_letters, self.letter_vocab, add_bos=False, add_eos=False) # Получаем индексы букв из словаря
            letters_vectorized = self.pad_sequence(letters_indices, max_letters_count, self.pad_idx) # Заполняем пространство паддингом
            letters[idx] = letters_vectorized

        src_indices = self.get_indices(tokenized_source, self.src_vocab, add_bos=add_bos_eos_tokens, add_eos=add_bos_eos_tokens)
        src_vectorized = self.pad_sequence(src_indices, max_tokens_count, self.pad_idx)
        subtokens_cnt = self.pad_sequence(subtokens_cnt, max_words_count, self.pad_idx)
        trg_vectorized = {}
        for target_name in target_names:
            trg_indices = self.get_indices(df_row[target_name], self.trg_vocabs[target_name], add_bos=add_bos_eos_tokens, add_eos=add_bos_eos_tokens)
            trg_vectorized[target_name] = self.pad_sequence(trg_indices, max_words_count, self.pad_idx)
        
        return {
            'src_vectorized':src_vectorized,
            'subtokens_cnt':subtokens_cnt,
            'letters':letters,
            'trg_vectorized':trg_vectorized
        }