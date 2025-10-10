import json

class Vocabulary:
    """
    Класс для работы со словарём токен‑индекс.
    
    Внутри хранит два отображения: token->idx и idx->token, а также специальные токены (MASK, UNK, BOS, EOS)
    """
    def __init__(self, token_to_idx:dict=None, bos_token:str='<BOS>', eos_token:str='<EOS>', pad_token:str='<PAD>', mask_token:str='<MASK>', unk_token='<UNK>', add_bos_eos_tokens:bool=True):
        """
        Инициализирует словарь.

        Parameters
        ----------
        token_to_idx : dict, optional
            Предоставляемый словарь токен : индекс. Если None, словарь создаётся пустым.
        bos_token : str, default '<BOS>'
            Токен начала предложения.
        eos_token : str, default '<EOS>'
            Токен окончания предложения.
        pad_token : str, default '<PAD>'
            Токен паддинга для заполнения недостающих позиций.
        mask_token : str, default '<MASK>'
            Токен маскировки.
        unk_token : str, default '<UNK>'
            Токен неизвестного слова.
        add_bos_eos_tokens : bool, default True
            Если True, добавляются BOS и EOS в словарь.
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
        else:
            self.token_to_idx = {}
            self.idx_to_token = {}
            self.pad_idx = self.add_token(pad_token)
            self.unk_idx = self.add_token(unk_token)
            self.mask_idx = self.add_token(mask_token)
            if add_bos_eos_tokens:
                self.bos_idx = self.add_token(bos_token)
                self.eos_idx = self.add_token(eos_token)

    def add_token(self, token:str)->int:
        """
        Добавляет токен в словарь. Возвращает индекс токена.

        Parameters
        ----------
        token : str
            Токен, который необходимо добавить.

        Returns
        -------
        int
            Индекс токена в словаре.
        """
        if token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
            return idx
        else:
            return self.token_to_idx[token]
        
    def add_tokens(self, tokens:list[str])->list[int]:
        """
        Добавляет несколько токенов в словарь и возвращает их индексы.

        Parameters
        ----------
        tokens : list[str]
            Список токенов для добавления.

        Returns
        -------
        list[int]
            Индексы соответствующих токенов.
        """
        return [self.add_token(token) for token in tokens]
    
    def get_token(self, index:int) -> str:
        if index in self.idx_to_token:
            return self.idx_to_token[index]
        return self.unk_token

    def get_index(self, token:str) -> int:
        if token in self.token_to_idx:
            return self.token_to_idx[token]
        return self.unk_idx
        
    def to_serializable(self) -> dict:
        """
        Возвращает словарь, пригодный для сериализации (JSON).
        """
        return {
            'token_to_idx': self.token_to_idx,
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
            'pad_token': self.pad_token,
            'mask_token': self.mask_token,
            'unk_token': self.unk_token,
            'add_auxiliary_tokens': self.add_bos_eos_tokens}

    @classmethod
    def from_serializable(cls, serializable: dict):
        """
        Создаёт объект Vocabulary из сериализованного представления.

        Parameters
        ----------
        serializable : dict
            Сериализованный словарь.

        Returns
        -------
        Vocabulary
            Новый экземпляр класса.
        """
        return cls(**serializable)

    def to_json(self, filepath: str):
        """
        Сохраняет словарь в JSON‑файл.

        Parameters
        ----------
        filepath : str
            Путь к файлу для сохранения.
        """
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(self.to_serializable(), file, ensure_ascii=False)

    @classmethod
    def from_json(cls, filepath: str):
        """
        Загружает словарь из JSON‑файла.

        Parameters
        ----------
        filepath : str
            Путь к файлу для загрузки.

        Returns
        -------
        Vocabulary
            Новый экземпляр класса.
        """
        with open(filepath, 'r', encoding='utf-8') as file:
            return cls.from_serializable(json.load(file))
        
    def __len__(self)->int:
        return len(self.token_to_idx)