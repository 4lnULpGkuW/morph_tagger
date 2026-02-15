import torch
import torch.nn as nn
import math

class RoPE(nn.Module):
    """
    Rotary Positional Encoding (RoPE) для применения в механизме внимания
    """
    def __init__(self, dim:int, max_seq_len:int, base:float = 10000.0, device:str='cpu'):
        super().__init__()
        self.dim = dim 
        self.max_seq_len = max_seq_len  # Максимальная длина последовательности
        self.base = base  # Базовое значение для вычисления частот
        
        # Вычисляем обратные частоты для каждой пары измерений
        # dim/2 частот, так как работаем с парами (cos, sin)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)  # Буфер не участвует в обучении
        
        # Создаем позиции [0, 1, 2, ..., max_seq_len-1]
        positions = torch.arange(max_seq_len).float()
        
        # Вычисляем частоты для каждой позиции: positions (max_seq_len) * inv_freq (dim/2)
        freqs = torch.einsum('i,j->ij', positions, self.inv_freq) # [max_seq_len, dim/2]
        
        # Предвычисляем косинусы и синусы для позиций
        cos_cached = torch.cos(freqs).unsqueeze(0) # [1, max_seq_len, dim/2]
        sin_cached = torch.sin(freqs).unsqueeze(0) # [1, max_seq_len, dim/2]
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)
        
    def forward(self, x:torch.Tensor, key_padding_mask:torch.Tensor = None):
        """
        Применяет RoPE к тензору
        """
        # Разделяем тензор на четные и нечетные компоненты
        x1 = x[..., 0::2] # [batch, seq, dim]
        x2 = x[..., 1::2] # [batch, seq, dim]
        
        # Применяем вращение
        rotated_even = x1 * self.cos_cached - x2 * self.sin_cached
        rotated_odd = x1 * self.sin_cached + x2 * self.cos_cached
        
        # Собираем обратно в исходный порядок
        rotated = torch.stack([rotated_even, rotated_odd], dim=-1)
        rotated = rotated.flatten(start_dim=-2)
        
        # Применяем маску паддинга
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(-1)
            rotated = rotated * (~mask).float()
        
        return rotated


class LearnablePositionalEncoding(nn.Module):
    """
    Обучаемое позиционное кодирование
    """
    def __init__(self, embed_dim:int, max_seq_len:int, padding_idx:int = 0, device:str = 'cpu'):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        
        self.pos_embeddings = nn.Embedding(max_seq_len+1, embed_dim, padding_idx=padding_idx)
        pos_idx = torch.arange(1, max_seq_len+1, device=device).unsqueeze(0) # [1, seq_size]
        
        # Регистрируем буфер
        self.register_buffer('pos_idx', pos_idx)

    def forward(self, x:torch.Tensor, key_padding_mask:torch.Tensor = None):
        batch_size = x.size(0)
        
        # Если есть маска, создаем тензор позиций для батча
        if key_padding_mask is not None:
            # Создаем позиции для всего батча [batch_size, seq_len]
            cur_pos = self.pos_idx.expand(batch_size, -1)
            cur_pos = cur_pos.masked_fill(key_padding_mask, self.padding_idx)
        else:
            cur_pos = self.pos_idx
        
        # Получаем позиционные эмбеддинги
        pos_emb = self.pos_embeddings(cur_pos)  # [batch_size, seq_size, embed_dim]
        
        # Добавляем к входным эмбеддингам
        x = x + pos_emb
        
        return x

class SinusoidalPositionalEncoding(nn.Module):
    """
    Синусоидальное позиционное кодирование (необучаемое)
    Использует синусоидальные функции для кодирования позиций
    """
    def __init__(self, embed_dim:int, max_seq_len:int, device:str='cpu'):
        super().__init__()
        position_matrix = torch.zeros(max_seq_len, embed_dim)

        positions = torch.arange(0, max_seq_len).unsqueeze(1)  # [S, 1]
        denominator = torch.exp((-math.log(10000.0) * torch.arange(0, embed_dim, 2) / embed_dim))

        position_matrix[:, 0::2] = torch.sin(positions * denominator)
        position_matrix[:, 1::2] = torch.cos(positions * denominator)

        position_matrix = position_matrix.unsqueeze(0)  # [1, S, E]

        self.register_buffer('position_matrix', position_matrix)

    def forward(self, x: torch.Tensor, key_padding_mask=None):
        B, S, E = x.size()
        cur_position_matrix = self.position_matrix.expand(B, -1, -1)
        cur_position_matrix = cur_position_matrix.masked_fill(key_padding_mask.unsqueeze(-1).expand(B, S, E))
        x = x + cur_position_matrix

        return x