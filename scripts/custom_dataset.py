import torch
from scripts.vectorizer import Vectorizer


class CustomDataset:
    def __init__(self, vectorizer: Vectorizer, train_df, target_names: list[str],
                 max_tokens_count: int, max_words_count: int, max_letters_count: int,
                 add_bos_eos_tokens: bool = True, test_df=None, valid_df=None):
        """
        Создаёт Dataset для обучения/инференса.

        Parameters
        ----------
        vectorizer : Vectorizer
            Объект, преобразующий токены в индексы.
        train_df : pandas.DataFrame
            DataFrame с обучающим набором данных.
        target_names : list[str]
            Список названий целевых меток.
        max_tokens_count : int
            Максимальное количество токенов.
        max_words_count : int
            Максимальное количество слов.
        max_letters_count : int
            Максимальное количество букв.
        add_bos_eos_tokens : bool, optional
            Добавлять ли BOS/EOS токены, по умолчанию True.
        test_df : pandas.DataFrame, optional
            DataFrame с тестовым набором данных.
        valid_df : pandas.DataFrame, optional
            DataFrame с валидационным набором данных.
        """
        self._train_df = train_df
        self._test_df = test_df
        self._valid_df = valid_df
        self.vectorizer = vectorizer
        self.target_names = target_names
        self.max_tokens_count = max_tokens_count
        self.max_words_count = max_words_count
        self.max_letters_count = max_letters_count
        self.add_bos_eos_tokens = add_bos_eos_tokens
        self.set_dataframe_split('train')

    def set_dataframe_split(self, split: str):
        """
        Устанавливает текущий DataFrame для работы.

        Parameters
        ----------
        split : str
            Один из {'train', 'test', 'validation'}.
        """
        if split == 'train':
            self.cw_df = self._train_df
        elif split == 'test':
            self.cw_df = self._test_df
        elif split == 'validation':
            self.cw_df = self._valid_df
        else:
            raise ValueError(
                'Неверное значение параметра split. '
                'Допустимые значения: train, test, validation'
            )

    def __len__(self):
        """Возвращает количество примеров в текущем наборе данных."""
        return len(self.cw_df)

    def __getitem__(self, index: int):
        """
        Возвращает один пример данных по индексу.

        Parameters
        ----------
        index : int
            Индекс примера.

        Returns
        -------
        dict
            Словарь с ключами:
            - Имена целевых признаков: тензоры значений
            - 'tokens': тензор токенов или None
            - 'letters': тензор букв или None
        """
        row = self.cw_df.iloc[index]
        vectorized_dict = self.vectorizer.vectorize(
            row, self.target_names,
            max_tokens_count=self.max_tokens_count,
            max_words_count=self.max_words_count,
            max_letters_count=self.max_letters_count,
            add_bos_eos_tokens=self.add_bos_eos_tokens
        )

        vectorized = {}
        for key, value in vectorized_dict['target'].items():
            vectorized[key] = torch.tensor(value)

        vectorized['tokens'] = (
            torch.tensor(vectorized_dict['tokens'])
            if vectorized_dict['tokens'] is not None else None
        )
        vectorized['letters'] = (
            torch.tensor(vectorized_dict['letters'])
            if vectorized_dict['letters'] is not None else None
        )

        return vectorized