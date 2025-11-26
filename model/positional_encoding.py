import torch
import torch.nn as nn
import math

class LearnablePositionalEncoding(nn.Module):
    """
    Обучаемое позиционное кодирование.  

    Parameters:
        embed_dim (int): Размерность эмбеддингов.
        max_seq_len (int): Максимальная длина последовательности.
        batch_first (bool): Указывает порядок батча и временного измерения:
            - True -> [B, N, D]
            - False -> [N, B, D]
    """
    def __init__(self, embed_dim:int, max_seq_len:int, batch_first:bool, padding_idx:int):
        super().__init__()
        self.batch_first = batch_first
        self.pos_embedings = nn.Embedding(max_seq_len, embed_dim, padding_idx)

    def forward(self, x:torch.Tensor, key_padding_mask=None):
        """
        Добавляет позиционное кодирование к входному тензору.

        Parameters:
            x (torch.Tensor): Тензор токенов с размером [B, S, D] или [S, B, D]

        Returns:
            torch.Tensor: Сумма исходного тензора и позиционного эмбеддинга.
            Размерность совпадает со входом.
        """
        # x [B, S, D]
        if self.batch_first:
            seq_len = x.size(1)
            pos_idx = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        else:
            seq_len = x.size(0)
            pos_idx = torch.arange(seq_len, device=x.device).unsqueeze(1)  # [seq_len, 1]
        
        pos_embed = self.pos_embedings(pos_idx) # [1, seq_len, D] | [seq_len, 1, D]

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(-1)
            pos_embed = pos_embed * (~mask).float()
        
        return x + pos_embed
    
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim:int, max_seq_len:int, batch_first:bool):
        super().__init__()
        self.batch_first = batch_first
        position_matrix = torch.zeros(max_seq_len, embed_dim)

        positions = torch.arange(0,max_seq_len).unsqueeze(1) # [S, 1]
        denominator = torch.exp((-math.log(10000.0) * torch.arange(0, embed_dim, 2) / embed_dim))

        position_matrix[:, 0::2] = torch.sin(positions*denominator)
        position_matrix[:, 1::2] = torch.cos(positions*denominator)

        position_matrix = position_matrix.unsqueeze(0) # [1, S, E]

        self.register_buffer('position_matrix', position_matrix)

    def forward(self, x:torch.Tensor, key_padding_mask=None):
        # device = x.device
        # position_matrix = self.position_matrix.to(device)
        x = x + self.position_matrix
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(-1)
            x = x * (~mask).float()
        return x
