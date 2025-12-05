"""
Multi-Horizon Forecasting with LSTM
Predicts values at multiple time horizons using specialized output heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiHorizonLSTM(nn.Module):
    """
    LSTM модель для прогнозирования на несколько горизонтов
    
    Архитектура:
    - Shared LSTM encoder для извлечения временных признаков
    - Отдельные декодеры для каждого горизонта
    - Adaptive horizon weighting
    """
    
    def __init__(
        self,
        input_size: int,
        shared_hidden_size: int = 128,
        decoder_hidden_size: int = 64,
        horizons: List[int] = [1, 3, 6, 12, 24],
        output_size: int = 1,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Args:
            input_size: Размерность входных признаков
            shared_hidden_size: Размер скрытого состояния общего LSTM
            decoder_hidden_size: Размер скрытого состояния декодеров
            horizons: Горизонты прогнозирования (часы)
            output_size: Размерность выхода (кол-во целевых переменных)
            num_layers: Кол-во слоев в LSTM
            dropout: Вероятность dropout
        """
        super().__init__()
        
        self.input_size = input_size
        self.shared_hidden_size = shared_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.horizons = sorted(horizons)
        self.output_size = output_size
        self.num_layers = num_layers
        
        # Shared LSTM encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=shared_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Horizon-specific decoders
        self.decoders = nn.ModuleDict({
            str(h): nn.LSTM(
                input_size=shared_hidden_size,
                hidden_size=decoder_hidden_size,
                num_layers=1,
                batch_first=True
            ) for h in self.horizons
        })
        
        # Output heads for each horizon
        self.output_heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(decoder_hidden_size, decoder_hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(decoder_hidden_size // 2, output_size)
            ) for h in self.horizons
        })
        
        # Adaptive horizon weighting
        self.horizon_weights = nn.Parameter(torch.ones(len(self.horizons)))
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
        
        logger.info(f"Initialized MultiHorizonLSTM with horizons: {self.horizons}")
    
    def _init_weights(self):
        """Инициализация весов"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Входные данные [batch_size, seq_len, input_size]
            
        Returns:
            Dict с предсказаниями для каждого горизонта
        """
        batch_size, seq_len, _ = x.shape
        
        # Shared encoding
        encoded, (h_n, c_n) = self.encoder(x)
        # Используем последнее скрытое состояние
        last_hidden = h_n[-1]  # [batch_size, shared_hidden_size]
        
        # Расширяем для каждого горизонта
        last_hidden_expanded = last_hidden.unsqueeze(1)  # [batch_size, 1, shared_hidden_size]
        
        predictions = {}
        
        # Для каждого горизонта свой декодер
        for i, horizon in enumerate(self.horizons):
            # Декодирование
            decoder_input = last_hidden_expanded.repeat(1, horizon, 1)
            decoder_out, _ = self.decoders[str(horizon)](decoder_input)
            
            # Предсказание (берем последний шаг декодера)
            pred = self.output_heads[str(horizon)](decoder_out[:, -1, :])
            predictions[str(horizon)] = pred
        
        return predictions
    
    def get_horizon_weights(self) -> torch.Tensor:
        """Получить нормализованные веса горизонтов"""
        return F.softmax(self.horizon_weights, dim=0)


class TemporalAttentionFusion(nn.Module):
    """
    Слияние предсказаний разных горизонтов с временным вниманием
    """
    
    def __init__(self, horizons: List[int], hidden_size: int = 64):
        super().__init__()
        self.horizons = sorted(horizons)
        self.hidden_size = hidden_size
        
        # Внимание для каждого горизонта
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            batch_first=True
        )
        
        # Проекция для слияния
        self.fusion_layer = nn.Sequential(
            nn.Linear(len(horizons) * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2)
        )
    
    def forward(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Слияние предсказаний разных горизонтов
        
        Args:
            predictions: Предсказания для каждого горизонта
            
        Returns:
            Слитый вектор признаков
        """
        # Конкатенируем все предсказания
        pred_list = [predictions[str(h)] for h in self.horizons]
        concatenated = torch.cat(pred_list, dim=1)
        
        # Проекция
        fused = self.fusion_layer(concatenated)
        return fused


def create_multi_horizon_model(
    input_size: int,
    horizons: List[int] = None,
    **kwargs
) -> MultiHorizonLSTM:
    """
    Фабричная функция для создания модели
    
    Args:
        input_size: Размерность входных признаков
        horizons: Горизонты прогнозирования
        **kwargs: Другие параметры модели
        
    Returns:
        MultiHorizonLSTM модель
    """
    if horizons is None:
        horizons = [1, 3, 6, 12, 24]
    
    model = MultiHorizonLSTM(
        input_size=input_size,
        horizons=horizons,
        **kwargs
    )
    
    return model


if __name__ == "__main__":
    # Тест модели
    model = create_multi_horizon_model(input_size=10, horizons=[1, 3, 6])
    
    # Тестовые данные
    batch_size, seq_len = 32, 50
    x = torch.randn(batch_size, seq_len, 10)
    
    # Предсказание
    predictions = model(x)
    
    print("Multi-Horizon Predictions:")
    for horizon, pred in predictions.items():
        print(f"  Horizon {horizon}: {pred.shape}")
    
    print(f"Horizon weights: {model.get_horizon_weights()}")