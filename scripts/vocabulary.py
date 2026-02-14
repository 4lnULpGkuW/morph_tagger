import json


class Vocabulary:
    """
    Класс для работы со словарём токен‑индекс.
    Внутри хранит два отображения: token->idx и idx->token, а также специальные токены (MASK, UNK, BOS, EOS)
    """
    def __init__(self, token_to_idx: dict = None, bos_token: str = '<BOS>', eos_token: str = '<EOS>', 
                 pad_token: str = '<PAD>', mask_token: str = '<MASK>', unk_token='<UNK>', 
                 add_bos_eos_tokens: bool = True):
        """
        Инициализирует словарь.

        Args:
            token_to_idx: Существующий словарь токен->индекс (опционально)
            bos_token: Токен начала последовательности
            eos_token: Токен конца последовательности
            pad_token: Токен заполнения
            mask_token: Токен маски
            unk_token: Токен неизвестного слова
            add_bos_eos_tokens: Флаг добавления BOS/EOS токенов в словарь
        """
        self.add_bos_eos_tokens = add_bos_eos_tokens
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.unk_token = unk_token

        if token_to_idx is not None:
            self.token_to_idx = token_to_idx
            self.idx_to_token = {value: key for key, value in token_to_idx.items()}
            self.pad_idx = self.token_to_idx[pad_token]
            self.unk_idx = self.token_to_idx[unk_token]
            self.mask_idx = self.token_to_idx[mask_token]
            if add_bos_eos_tokens:
                self.bos_idx = self.token_to_idx[bos_token]
                self.eos_idx = self.token_to_idx[eos_token]
        else:
            self.token_to_idx = {}
            self.idx_to_token = {}
            self.pad_idx = self.add_token(pad_token)
            self.unk_idx = self.add_token(unk_token)
            self.mask_idx = self.add_token(mask_token)
            if add_bos_eos_tokens:
                self.bos_idx = self.add_token(bos_token)
                self.eos_idx = self.add_token(eos_token)

    def add_token(self, token: str) -> int:
        """
        Добавляет токен в словарь.

        Args:
            token: Токен для добавления

        Returns:
            Индекс добавленного или существующего токена
        """
        if token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
            return idx
        else:
            return self.token_to_idx[token]

    def add_tokens(self, tokens: list[str]) -> list[int]:
        """
        Добавляет несколько токенов в словарь.

        Args:
            tokens: Список токенов для добавления

        Returns:
            Список индексов добавленных токенов
        """
        return [self.add_token(token) for token in tokens]

    def get_token(self, index: int) -> str:
        """
        Возвращает токен по индексу.

        Args:
            index: Индекс токена

        Returns:
            Токен или unk_token, если индекс не найден
        """
        if index in self.idx_to_token:
            return self.idx_to_token[index]
        return self.unk_token
    
    def get_tokens(self, indices:list[int]):
        '''Возвращает список токенов'''
        return [self.get_token(index) for index in indices]


    def get_index(self, token: str) -> int:
        """
        Возвращает индекс по токену.

        Args:
            token: Токен для поиска

        Returns:
            Индекс токена или unk_idx, если токен не найден
        """
        if token in self.token_to_idx:
            return self.token_to_idx[token]
        return self.unk_idx

    def to_serializable(self) -> dict:
        """
        Возвращает словарь, пригодный для сериализации (JSON).

        Returns:
            Словарь с данными для сериализации
        """
        return {
            'token_to_idx': self.token_to_idx,
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
            'pad_token': self.pad_token,
            'mask_token': self.mask_token,
            'unk_token': self.unk_token,
            'add_bos_eos_tokens': self.add_bos_eos_tokens}

    @classmethod
    def from_serializable(cls, serializable: dict):
        """
        Создаёт объект Vocabulary из сериализованного представления.

        Args:
            serializable: Словарь с сериализованными данными

        Returns:
            Экземпляр Vocabulary
        """
        return cls(**serializable)

    def to_json(self, filepath: str):
        """
        Сохраняет словарь в JSON-файл.

        Args:
            filepath: Путь к файлу для сохранения
        """
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(self.to_serializable(), file, ensure_ascii=False)

    @classmethod
    def from_json(cls, filepath: str):
        """
        Загружает словарь из JSON-файла.

        Args:
            filepath: Путь к файлу для загрузки

        Returns:
            Экземпляр Vocabulary
        """
        with open(filepath, 'r', encoding='utf-8') as file:
            return cls.from_serializable(json.load(file))

    def __len__(self) -> int:
        """Возвращает размер словаря."""
        return len(self.token_to_idx)