"""
Uncertainty Estimation for LSTM Predictions
Provides confidence intervals and uncertainty quantification using Monte Carlo Dropout and ensemble methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UncertaintyAwareLSTM(nn.Module):
    """
    LSTM с оценкой неопределенности
    
    Методы:
    - Monte Carlo Dropout для эпистемической неопределенности
    - Гетероскедастическая неопределенность через параметрические выходы
    - Калибровка уверенности
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        output_size: int = 1,
        num_layers: int = 2,
        dropout: float = 0.3,
        uncertainty_type: str = "monte_carlo"  # "monte_carlo", "heteroscedastic", "both"
    ):
        """
        Args:
            input_size: Размерность входных признаков
            hidden_size: Размер скрытого состояния
            output_size: Размерность выхода
            num_layers: Кол-во слоев LSTM
            dropout: Вероятность dropout
            uncertainty_type: Тип оценки неопределенности
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.uncertainty_type = uncertainty_type
        
        # LSTM с постоянным dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Deterministic output head
        self.deterministic_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Uncertainty estimation heads
        if uncertainty_type in ["heteroscedastic", "both"]:
            # Гетероскедастическая неопределенность
            self.aleatoric_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, output_size),  # log_variance
                nn.Softplus()  # Ensure positive variance
            )
        
        if uncertainty_type in ["monte_carlo", "both"]:
            # Enable dropout during inference for MC sampling
            self.mc_dropout = nn.Dropout(dropout)
        
        self._init_weights()
        logger.info(f"Initialized UncertaintyAwareLSTM with type: {uncertainty_type}")
    
    def _init_weights(self):
        """Инициализация весов"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, training: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Входные данные [batch_size, seq_len, input_size]
            training: Режим обучения
            
        Returns:
            Dict с предсказаниями и неопределенностью
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Используем последнее скрытое состояние
        last_hidden = h_n[-1]  # [batch_size, hidden_size]
        
        # Deterministic prediction
        prediction = self.deterministic_head(last_hidden)
        
        result = {
            "prediction": prediction
        }
        
        # Heteroscedastic uncertainty
        if self.uncertainty_type in ["heteroscedastic", "both"]:
            log_variance = self.aleatoric_head(last_hidden)
            result["log_variance"] = log_variance
            result["std_dev"] = torch.exp(0.5 * log_variance)
        
        # Monte Carlo uncertainty (enabled by keeping dropout active)
        if self.uncertainty_type in ["monte_carlo", "both"] and not training:
            # During inference, we keep dropout active for uncertainty estimation
            # This is handled in the predict_with_uncertainty method
            pass
        
        return result
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor, 
        n_samples: int = 100,
        confidence_level: float = 0.95
    ) -> Dict[str, torch.Tensor]:
        """
        Предсказание с оценкой неопределенности
        
        Args:
            x: Входные данные
            n_samples: Кол-во образцов для MC Dropout
            confidence_level: Уровень доверия для интервалов
            
        Returns:
            Dict с предсказаниями, неопределенностью и интервалами
        """
        if self.uncertainty_type == "monte_carlo":
            return self._mc_uncertainty_prediction(x, n_samples, confidence_level)
        elif self.uncertainty_type == "heteroscedastic":
            return self._heteroscedastic_prediction(x, confidence_level)
        else:  # both
            return self._combined_uncertainty_prediction(x, n_samples, confidence_level)
    
    def _mc_uncertainty_prediction(
        self, 
        x: torch.Tensor, 
        n_samples: int, 
        confidence_level: float
    ) -> Dict[str, torch.Tensor]:
        """Monte Carlo Dropout предсказание"""
        self.train()  # Enable dropout
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                output = self.forward(x, training=False)
                predictions.append(output["prediction"])
        
        predictions = torch.stack(predictions)  # [n_samples, batch_size, output_size]
        
        # Статистика по образцам
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # Доверительные интервалы
        alpha = 1 - confidence_level
        z_score = torch.distributions.Normal(0, 1).icdf(torch.tensor(1 - alpha/2))
        
        lower_bound = mean_pred - z_score * std_pred
        upper_bound = mean_pred + z_score * std_pred
        
        self.eval()  # Return to eval mode
        
        return {
            "prediction": mean_pred,
            "uncertainty": std_pred,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "samples": predictions
        }
    
    def _heteroscedastic_prediction(
        self, 
        x: torch.Tensor, 
        confidence_level: float
    ) -> Dict[str, torch.Tensor]:
        """Гетероскедастическое предсказание"""
        self.eval()
        with torch.no_grad():
            output = self.forward(x, training=False)
        
        prediction = output["prediction"]
        std_dev = output["std_dev"]
        
        # Доверительные интервалы
        alpha = 1 - confidence_level
        z_score = torch.distributions.Normal(0, 1).icdf(torch.tensor(1 - alpha/2))
        
        lower_bound = prediction - z_score * std_dev
        upper_bound = prediction + z_score * std_dev
        
        return {
            "prediction": prediction,
            "uncertainty": std_dev,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }
    
    def _combined_uncertainty_prediction(
        self, 
        x: torch.Tensor, 
        n_samples: int, 
        confidence_level: float
    ) -> Dict[str, torch.Tensor]:
        """Комбинированная оценка неопределенности"""
        # MC Dropout часть
        mc_result = self._mc_uncertainty_prediction(x, n_samples, confidence_level)
        mc_std = mc_result["uncertainty"]
        
        # Гетероскедастическая часть
        self.eval()
        with torch.no_grad():
            het_output = self.forward(x, training=False)
        het_std = het_output.get("std_dev", torch.zeros_like(mc_std))
        
        # Комбинирование
        combined_std = torch.sqrt(mc_std**2 + het_std**2)
        
        prediction = mc_result["prediction"]
        alpha = 1 - confidence_level
        z_score = torch.distributions.Normal(0, 1).icdf(torch.tensor(1 - alpha/2))
        
        lower_bound = prediction - z_score * combined_std
        upper_bound = prediction + z_score * combined_std
        
        return {
            "prediction": prediction,
            "uncertainty": combined_std,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "mc_uncertainty": mc_std,
            "het_uncertainty": het_std
        }


class ConfidenceCalibrator(nn.Module):
    """
    Калибратор уверенности для корректировки прогнозируемых вероятностей
    """
    
    def __init__(self, input_size: int = 1, hidden_size: int = 32):
        super().__init__()
        self.calibration_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, predicted_confidence: torch.Tensor) -> torch.Tensor:
        """
        Калибровка уверенности
        
        Args:
            predicted_confidence: Прогнозируемая уверенность [batch_size, 1]
            
        Returns:
            Откалиброванная уверенность [batch_size, 1]
        """
        return self.calibration_net(predicted_confidence)


def create_uncertainty_model(
    input_size: int,
    uncertainty_type: str = "both",
    **kwargs
) -> UncertaintyAwareLSTM:
    """
    Фабричная функция для создания модели с неопределенностью
    
    Args:
        input_size: Размерность входных признаков
        uncertainty_type: Тип неопределенности
        **kwargs: Другие параметры
        
    Returns:
        UncertaintyAwareLSTM модель
    """
    model = UncertaintyAwareLSTM(
        input_size=input_size,
        uncertainty_type=uncertainty_type,
        **kwargs
    )
    
    return model


if __name__ == "__main__":
    # Тест модели
    model = create_uncertainty_model(input_size=10, uncertainty_type="both")
    
    # Тестовые данные
    batch_size, seq_len = 32, 50
    x = torch.randn(batch_size, seq_len, 10)
    
    # Deterministic prediction
    det_output = model(x)
    print("Deterministic output keys:", det_output.keys())
    print("Prediction shape:", det_output["prediction"].shape)
    
    # Uncertainty prediction
    unc_output = model.predict_with_uncertainty(x, n_samples=10)
    print("\nUncertainty output keys:", unc_output.keys())
    print("Mean prediction shape:", unc_output["prediction"].shape)
    print("Uncertainty shape:", unc_output["uncertainty"].shape)
    print("Confidence interval shapes:")
    print("  Lower bound:", unc_output["lower_bound"].shape)
    print("  Upper bound:", unc_output["upper_bound"].shape)