import regex as re
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Whitespace, Punctuation
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents

class SeparatorTokenizer:
    @staticmethod
    def tokenize(text: str, separator: str = None):
        """
        Разбивает строку на список токенов.

        Args:
            text (str): исходный текст
            separator (str | None): символ/строка, по которой происходит
                разбиение. Если None – используется умная токенизация
                с учётом знаков препинания, дефисов и сокращений.
        Returns:
            list[str]: список токенов
        """
        if separator is not None:
            return text.split(sep=separator)

        # Регулярное выражение для поиска токенов (пробелы пропускаются)
        pattern = re.compile(
            r'\.\.\.|…|--|—|'           # многосимвольные знаки
            r'\w{1,3}\.|'                # сокращения (1-3 буквы + точка)
            r'\w+(?:-\w+)*|'              # слова с дефисами
            r'[^\w\s]'                    # одиночные знаки препинания
        )
        return pattern.findall(text)


class BPETokenizer():
    @staticmethod
    def train(corpus_files:list[str], vocab_size:int, min_frequency:int, continuing_subword_prefix:str='##', unk_token:str='<UNK>', pad_token:str='<PAD>')->Tokenizer:
        '''Обучает BPE токенизатор и возвращает обьект класса tokenizer.Tokenizer'''

        tokenizer = Tokenizer(BPE(unk_token=unk_token))
        
        # нормализация
        # tokenizer.normalizer = normalizers.Sequence([
        #     NFD(),
        #     StripAccents()
        # ])
        
        # пред-токенизация
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            Whitespace(),
            Punctuation()
        ])
        
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=[unk_token, pad_token],
            show_progress=False,
            continuing_subword_prefix=continuing_subword_prefix
        )
        
        # обучение
        tokenizer.train(corpus_files, trainer)
        
        return tokenizer
    
    @staticmethod
    def from_pretrained(filepath:str)->Tokenizer:
        return Tokenizer.from_file(filepath)
