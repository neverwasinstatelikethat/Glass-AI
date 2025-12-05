"""
AutoML tuner для автоматической настройки гиперпараметров моделей
Использует Optuna для поиска оптимальных конфигураций
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.trial import TrialState
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable
import logging
from datetime import datetime
import json
import mlflow
import mlflow.pytorch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Базовый тренер для моделей PyTorch"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        experiment_name: str = "glass_defect_prediction"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.experiment_name = experiment_name
        
        # Перенос модели на устройство
        self.model.to(self.device)
        
        # Инициализация MLflow
        mlflow.set_experiment(experiment_name)
    
    def train_epoch(self) -> float:
        """Обучение одной эпохи"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
        
        avg_loss = total_loss / total_samples
        return avg_loss
    
    def validate(self) -> Tuple[float, float]:
        """Валидация модели"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += data.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples
        
        return avg_loss, accuracy
    
    def train(
        self,
        epochs: int = 10,
        patience: int = 5,
        min_delta: float = 1e-4
    ) -> Dict[str, List[float]]:
        """Полный цикл обучения с early stopping"""
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Начало MLflow run
        with mlflow.start_run():
            # Логирование параметров модели
            mlflow.log_param("model_type", self.model.__class__.__name__)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("patience", patience)
            
            for epoch in range(epochs):
                # Обучение
                train_loss = self.train_epoch()
                train_losses.append(train_loss)
                
                # Валидация
                val_loss, val_accuracy = self.validate()
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                
                # Логирование метрик
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
                
                logger.info(f"Epoch {epoch+1}/{epochs}: "
                           f"Train Loss: {train_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f}, "
                           f"Val Acc: {val_accuracy:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Сохранение лучшей модели
                    torch.save(self.model.state_dict(), "best_model.pth")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Логирование лучшей модели
            mlflow.log_artifact("best_model.pth")
        
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies
        }


class OptunaLSTMTuner:
    """Tuner для LSTM модели с Attention"""
    
    def __init__(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        device: torch.device
    ):
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
    
    def objective(self, trial: optuna.Trial) -> float:
        """Целевая функция для оптимизации"""
        # Гиперпараметры для поиска
        hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        
        # Создание модели
        from models.lstm_predictor.attention_lstm import create_lstm_model
        model = create_lstm_model(
            input_size=self.train_data[0].shape[2],
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=self.train_data[1].shape[1],
            dropout=dropout
        )
        
        # Создание загрузчиков данных
        train_dataset = TensorDataset(
            torch.FloatTensor(self.train_data[0]), 
            torch.LongTensor(self.train_data[1])
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(self.val_data[0]), 
            torch.LongTensor(self.val_data[1])
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Оптимизатор и функция потерь
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Тренировка
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=self.device
        )
        
        # Обучение (3 эпохи для быстрой оценки)
        history = trainer.train(epochs=3, patience=2)
        
        # Возвращаем лучшую валидационную точность
        return max(history["val_accuracies"])


class OptunaViTTuner:
    """Tuner для Vision Transformer"""
    
    def __init__(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        device: torch.device
    ):
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
    
    def objective(self, trial: optuna.Trial) -> float:
        """Целевая функция для оптимизации"""
        # Гиперпараметры для поиска
        embed_dim = trial.suggest_categorical("embed_dim", [256, 512, 768])
        depth = trial.suggest_int("depth", 6, 12)
        n_heads = trial.suggest_categorical("n_heads", [4, 8, 12])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        
        # Создание модели
        from models.vision_transformer.defect_detector import create_vit_classifier
        model = create_vit_classifier(
            img_size=224,
            patch_size=16,
            in_channels=3,
            n_classes=self.train_data[1].shape[1],
            embed_dim=embed_dim,
            depth=depth,
            n_heads=n_heads
        )
        
        # Создание загрузчиков данных
        train_dataset = TensorDataset(
            torch.FloatTensor(self.train_data[0]), 
            torch.LongTensor(self.train_data[1])
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(self.val_data[0]), 
            torch.LongTensor(self.val_data[1])
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Оптимизатор и функция потерь
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Тренировка
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=self.device
        )
        
        # Обучение (3 эпохи для быстрой оценки)
        history = trainer.train(epochs=3, patience=2)
        
        # Возвращаем лучшую валидационную точность
        return max(history["val_accuracies"])


class OptunaGNNTuner:
    """Tuner для GNN модели"""
    
    def __init__(
        self,
        train_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        val_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        device: torch.device
    ):
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
    
    def objective(self, trial: optuna.Trial) -> float:
        """Целевая функция для оптимизации"""
        # Гиперпараметры для поиска
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
        num_layers = trial.suggest_int("num_layers", 2, 4)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        gnn_type = trial.suggest_categorical("gnn_type", ["GCN", "GAT"])
        
        # Создание модели
        from models.gnn_sensor_network.gnn_model import create_sensor_gnn
        model = create_sensor_gnn(
            num_sensors=10,  # Примерное количество датчиков
            input_dim=1,
            hidden_dim=hidden_dim,
            output_dim=32,
            model_type=gnn_type
        )
        
        # Для GNN нужна специальная обработка данных
        # Здесь упрощенная реализация - предполагаем, что данные уже в нужном формате
        # В реальной реализации потребуется более сложная обработка
        
        # Создание оптимизатора и функции потерь
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()  # Для регрессии аномалий
        
        # Упрощенная тренировка
        model.to(self.device)
        model.train()
        
        total_loss = 0.0
        for epoch in range(3):  # 3 эпохи для быстрой оценки
            for data_tuple in self.train_data:
                x, edge_index, target = data_tuple
                x, edge_index, target = x.to(self.device), edge_index.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(x, edge_index)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        # Возвращаем отрицательный loss (так как optuna максимизирует)
        return -total_loss / len(self.train_data)


def run_optuna_study(
    tuner_class: type,
    tuner_params: Dict[str, Any],
    n_trials: int = 20,
    study_name: str = "model_optimization"
) -> optuna.Study:
    """Запуск Optuna study для оптимизации гиперпараметров"""
    
    # Создание tuner
    tuner = tuner_class(**tuner_params)
    
    # Создание study
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage="sqlite:///optuna_study.db",
        load_if_exists=True
    )
    
    # Запуск оптимизации
    study.optimize(tuner.objective, n_trials=n_trials)
    
    # Логирование результатов
    logger.info(f"Лучшие параметры {study_name}: {study.best_params}")
    logger.info(f"Лучшее значение: {study.best_value}")
    
    # Сохранение результатов
    with open(f"{study_name}_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    
    return study


def create_automl_pipeline(
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> Dict[str, optuna.Study]:
    """Создание полного AutoML pipeline для всех моделей"""
    
    studies = {}
    
    # LSTM tuner
    if "lstm" in train_data:
        logger.info("🚀 Запуск AutoML для LSTM модели...")
        lstm_study = run_optuna_study(
            tuner_class=OptunaLSTMTuner,
            tuner_params={
                "train_data": train_data["lstm"],
                "val_data": val_data["lstm"],
                "device": device
            },
            n_trials=15,
            study_name="lstm_attention_optimization"
        )
        studies["lstm"] = lstm_study
    
    # ViT tuner
    if "vit" in train_data:
        logger.info("🚀 Запуск AutoML для ViT модели...")
        vit_study = run_optuna_study(
            tuner_class=OptunaViTTuner,
            tuner_params={
                "train_data": train_data["vit"],
                "val_data": val_data["vit"],
                "device": device
            },
            n_trials=12,
            study_name="vit_optimization"
        )
        studies["vit"] = vit_study
    
    # GNN tuner
    if "gnn" in train_data:
        logger.info("🚀 Запуск AutoML для GNN модели...")
        gnn_study = run_optuna_study(
            tuner_class=OptunaGNNTuner,
            tuner_params={
                "train_data": train_data["gnn"],
                "val_data": val_data["gnn"],
                "device": device
            },
            n_trials=10,
            study_name="gnn_optimization"
        )
        studies["gnn"] = gnn_study
    
    logger.info("✅ AutoML pipeline завершен")
    return studies


# ==================== ТЕСТИРОВАНИЕ ====================

if __name__ == "__main__":
    # Создание синтетических данных для тестирования
    print("🔍 Создание тестовых данных...")
    
    # LSTM данные (временные ряды)
    seq_len = 60
    n_features = 20
    n_classes = 5
    n_samples = 1000
    
    lstm_train_X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    lstm_train_y = np.random.randint(0, n_classes, n_samples)
    lstm_val_X = np.random.randn(200, seq_len, n_features).astype(np.float32)
    lstm_val_y = np.random.randint(0, n_classes, 200)
    
    # ViT данные (изображения)
    img_size = 224
    vit_train_X = np.random.randn(800, 3, img_size, img_size).astype(np.float32)
    vit_train_y = np.random.randint(0, n_classes, 800)
    vit_val_X = np.random.randn(200, 3, img_size, img_size).astype(np.float32)
    vit_val_y = np.random.randint(0, n_classes, 200)
    
    # GNN данные (графы)
    # Упрощенная реализация - в реальности потребуются реальные графы
    gnn_train_data = [
        (torch.randn(10, 1), torch.randint(0, 2, (2, 15)), torch.randn(10, 32))
        for _ in range(100)
    ]
    gnn_val_data = [
        (torch.randn(10, 1), torch.randint(0, 2, (2, 15)), torch.randn(10, 32))
        for _ in range(20)
    ]
    
    # Подготовка данных
    train_data = {
        "lstm": (lstm_train_X, lstm_train_y),
        "vit": (vit_train_X, vit_train_y),
        "gnn": gnn_train_data
    }
    
    val_data = {
        "lstm": (lstm_val_X, lstm_val_y),
        "vit": (vit_val_X, vit_val_y),
        "gnn": gnn_val_data
    }
    
    # Тестирование AutoML pipeline (с уменьшенным количеством trials)
    print("🔍 Тестирование AutoML pipeline...")
    
    # Для тестирования уменьшим количество trials
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # LSTM tuner test
    print("  Тестирование LSTM tuner...")
    lstm_tuner = OptunaLSTMTuner(
        train_data=train_data["lstm"],
        val_data=val_data["lstm"],
        device=device
    )
    
    # Запуск одного trial для теста
    study = optuna.create_study(direction="maximize")
    study.optimize(lstm_tuner.objective, n_trials=1)
    print(f"    Лучшее значение (LSTM): {study.best_value}")
    
    # ViT tuner test
    print("  Тестирование ViT tuner...")
    vit_tuner = OptunaViTTuner(
        train_data=train_data["vit"],
        val_data=val_data["vit"],
        device=device
    )
    
    study = optuna.create_study(direction="maximize")
    study.optimize(vit_tuner.objective, n_trials=1)
    print(f"    Лучшее значение (ViT): {study.best_value}")
    
    print("\n✅ Все тесты пройдены!")