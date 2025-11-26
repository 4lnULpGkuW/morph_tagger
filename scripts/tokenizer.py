import regex as re
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Whitespace, Punctuation
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents

class SeparatorTokenizer:
    def __init__(self):
        pass

    def tokenize(self, text:str, separator:str=None):
        """
        Разбивает строку на список токенов

        Args:
            text (str): исходный текст
            separator (str | None): символ/строка, по которой происходит
                разбиение. Если None – используется стандартное split() без
                аргументов (разделитель «пробел»)
        Returns:
            list[str]: список токенов
        """
        text = re.sub(r'[\w\s]+([^\w\s]+)', r' \1 ', text)  # Отделяем прбелом знаки препинания.
                                                            # Знаки препинания, следующие друг за другом, считаются одним токеном. Например: "...", "!?!"
        text = re.sub(r'[\t\n\r\f\v]', r' ', text)
        return text.split(sep=separator)


def train_bpe_tokenizer(corpus_files:list[str], vocab_size:int, min_frequency:int, continuing_subword_prefix:str='##', unk_token:str='<UNK>', pad_token:str='<PAD>')->Tokenizer:
    '''Обучает и возвращает обьект класса tokenizer.Tokenizer'''

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

def get_bpe_tokenizer_from_file(filepath:str)->Tokenizer:
    return Tokenizer.from_file(filepath)
