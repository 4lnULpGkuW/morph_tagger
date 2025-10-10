import torch
import numpy as np
import pandas as pd
from scripts.vocabulary import Vocabulary

class Vectorizer:
    def __init__(self, src_vocab:Vocabulary, trg_vocabs:dict[str:Vocabulary], max_src_len:int, pad_idx:int):
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
        self.max_src_len = max_src_len
        self.src_vocab = src_vocab
        self.target_cnt = len(trg_vocabs)
        self.trg_vocabs = trg_vocabs
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
    
    
    def pad_sequence(self, indices:list[int], use_vocab_max_len:bool):
        """
        Паддинг последовательности до заданной длины.

        Parameters
        ----------
        indices : list[int]
            Индексы токенов.
        use_vocab_max_len : bool
            Если True, используется максимальная длина `max_src_len` для обучения,
            иначе используется текущая длина последовательности (для инференса).
        """
        # Для обучения используем максимальную длину предложения, для инференса - длину текущего предложения
        if use_vocab_max_len:
            seq_len = self.max_src_len
        else:
            seq_len = len(indices)
        
        padded = [self.pad_idx] * seq_len # Заполнение индексом маскировочного токена
        padded[:len(indices)] = indices # Заполнение индексами реальных токенов
        return padded


    def vectorize(self, df_row:pd.Series, target_names:list[str], use_vocab_max_len:bool=True, add_bos_eos_tokens:bool=True)->tuple[list[int], dict[str, list[int]]]:
        """
        Принимает строку датафрейма и возвращает набор индексов:
            (векторизованный source, {название target : векторизованный target}).

        Parameters
        ----------
        df_row : pd.Series
            Строка из DataFrame с токенами для source и целевых меток.
        target_names : list[str]
            Список названий целевых меток.
        use_vocab_max_len : bool, default True
            Если True, использует максимальную длину `max_src_len` (для обучения),
            иначе текущую длину последовательности.
        """
        src_indices = self.get_indices(df_row['source_tokens'], self.src_vocab, add_bos=add_bos_eos_tokens, add_eos=add_bos_eos_tokens)
        src_vectorized = self.pad_sequence(src_indices, use_vocab_max_len)
        trg_vectorized = {}
        for target_name in target_names:
            trg_indices = self.get_indices(df_row[target_name], self.trg_vocabs[target_name], add_bos=add_bos_eos_tokens, add_eos=add_bos_eos_tokens)
            # trg_indices = [self.mask_idx] # для BOS
            # trg_indices.extend(self.get_indices(df_row[target_name], self.trg_vocabs[target_name], 
            #                                 add_bos=False, add_eos=False))
            # trg_indices.append(self.mask_idx) # для EOS
            trg_vectorized[target_name] = self.pad_sequence(trg_indices, use_vocab_max_len)
        
        return (src_vectorized, trg_vectorized)