import torch
import torch.nn as nn
import math


class LearnablePositionalEncoding(nn.Module):
    """
    Обучаемое позиционное кодирование.
    Использует слой nn.Embedding для изучения позиционных эмбеддингов.
    
    Args:
        embed_dim (int): Размерность эмбеддинга.
        max_seq_len (int): Максимальная длина последовательности.
        padding_idx (int): Индекс паддинга в эмбеддингах.
    """
    def __init__(self, embed_dim: int, max_seq_len: int, padding_idx: int):
        super().__init__()
        self.pos_embedings = nn.Embedding(max_seq_len, embed_dim, padding_idx)

    def forward(self, x: torch.Tensor, key_padding_mask=None):
        """
        Добавляет позиционное кодирование к входному тензору.
        
        Args:
            x (torch.Tensor): Входной тензор формы [B, S, D].
            key_padding_mask (torch.Tensor, optional): Маска паддинга формы [B, S].
            
        Returns:
            torch.Tensor: Тензор с добавленным позиционным кодированием.
        """
        seq_len = x.size(1)
        pos_idx = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        pos_embed = self.pos_embedings(pos_idx)  # [1, seq_len, D]

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(-1)
            pos_embed = pos_embed * (~mask).float()
        
        return x + pos_embed


class SinusoidalPositionalEncoding(nn.Module):
    """
    Синусоидальное позиционное кодирование (необучаемое).
    Использует синусоидальные функции для кодирования позиций.
    
    Args:
        embed_dim (int): Размерность эмбеддинга.
        max_seq_len (int): Максимальная длина последовательности.
    """
    def __init__(self, embed_dim: int, max_seq_len: int):
        super().__init__()
        position_matrix = torch.zeros(max_seq_len, embed_dim)

        positions = torch.arange(0, max_seq_len).unsqueeze(1)  # [S, 1]
        denominator = torch.exp((-math.log(10000.0) * torch.arange(0, embed_dim, 2) / embed_dim))

        position_matrix[:, 0::2] = torch.sin(positions * denominator)
        position_matrix[:, 1::2] = torch.cos(positions * denominator)

        position_matrix = position_matrix.unsqueeze(0)  # [1, S, E]

        self.register_buffer('position_matrix', position_matrix)

    def forward(self, x: torch.Tensor, key_padding_mask=None):
        """
        Добавляет синусоидальное позиционное кодирование к входному тензору.
        
        Args:
            x (torch.Tensor): Входной тензор формы [B, S, D].
            key_padding_mask (torch.Tensor, optional): Маска паддинга формы [B, S].
            
        Returns:
            torch.Tensor: Тензор с добавленным позиционным кодированием.
        """
        x = x + self.position_matrix
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(-1)
            x = x * (~mask).float()
        return x