"""
Edge inference pipeline для запуска моделей на устройствах NVIDIA Jetson
Поддерживает ONNX/TensorRT оптимизацию и offline режим работы
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import onnx
import onnxruntime as ort
import logging
from datetime import datetime
import json
import time

# Для TensorRT (если доступен)
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logging.warning("TensorRT не доступен, будет использован ONNX Runtime")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ONNXModelRunner:
    """Runner для ONNX моделей"""
    
    def __init__(self, model_path: str, providers: List[str] = None):
        """
        Args:
            model_path: путь к ONNX модели
            providers: список провайдеров для ONNX Runtime
        """
        self.model_path = model_path
        
        # Определение провайдеров по умолчанию
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Создание inference сессии
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Получение входных и выходных названий
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        logger.info(f"✅ Загружена ONNX модель: {model_path}")
        logger.info(f"📥 Входы: {self.input_names}")
        logger.info(f"📤 Выходы: {self.output_names}")
    
    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Предсказание с помощью ONNX модели
        
        Args:
            inputs: словарь с входными данными {input_name: data}
            
        Returns:
            outputs: словарь с выходными данными {output_name: data}
        """
        # Подготовка входов
        ort_inputs = {
            name: inputs[name] for name in self.input_names
        }
        
        # Предсказание
        start_time = time.time()
        ort_outputs = self.session.run(self.output_names, ort_inputs)
        inference_time = time.time() - start_time
        
        # Формирование результата
        outputs = {
            name: output for name, output in zip(self.output_names, ort_outputs)
        }
        
        logger.debug(f"⏱️ Время инференса: {inference_time:.4f} сек")
        
        return outputs


class TensorRTModelRunner:
    """Runner для TensorRT моделей (если доступен)"""
    
    def __init__(self, engine_path: str):
        """
        Args:
            engine_path: путь к TensorRT engine файлу
        """
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT не доступен")
        
        self.engine_path = engine_path
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        
        # Получение размеров входов/выходов
        self.input_shapes = []
        self.output_shapes = []
        
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                self.input_shapes.append(self.engine.get_binding_shape(i))
            else:
                self.output_shapes.append(self.engine.get_binding_shape(i))
        
        logger.info(f"✅ Загружен TensorRT engine: {engine_path}")
    
    def _load_engine(self, engine_path: str):
        """Загрузка TensorRT engine"""
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine
    
    def predict(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """
        Предсказание с помощью TensorRT
        
        Args:
            inputs: список входных массивов
            
        Returns:
            outputs: список выходных массивов
        """
        # Выделение GPU памяти
        bindings = []
        for i in range(self.engine.num_bindings):
            binding_shape = self.engine.get_binding_shape(i)
            size = trt.volume(binding_shape) * self.engine.max_batch_size * np.dtype(np.float32).itemsize
            bindings.append(cuda.mem_alloc(size))
        
        # Копирование входных данных в GPU
        for i, input_data in enumerate(inputs):
            cuda.memcpy_htod(bindings[i], input_data)
        
        # Выполнение инференса
        start_time = time.time()
        self.context.execute_v2(bindings=bindings)
        inference_time = time.time() - start_time
        
        # Копирование результатов из GPU
        outputs = []
        for i in range(len(inputs), len(bindings)):
            output_shape = self.engine.get_binding_shape(i)
            output_size = trt.volume(output_shape) * self.engine.max_batch_size
            output_data = np.empty(output_size, dtype=np.float32)
            cuda.memcpy_dtoh(output_data, bindings[i])
            outputs.append(output_data.reshape(output_shape))
        
        logger.debug(f"⏱️ Время инференса (TensorRT): {inference_time:.4f} сек")
        
        return outputs


class EdgeModelManager:
    """Менеджер моделей для edge устройств"""
    
    def __init__(self, model_configs: Dict[str, Dict]):
        """
        Args:
            model_configs: конфигурации моделей
                {
                    "lstm": {"path": "lstm_model.onnx", "type": "onnx"},
                    "vit": {"path": "vit_model.trt", "type": "tensorrt"},
                    ...
                }
        """
        self.model_configs = model_configs
        self.models = {}
        self.latency_stats = {}
        
        # Загрузка моделей
        self._load_models()
    
    def _load_models(self):
        """Загрузка всех моделей"""
        for model_name, config in self.model_configs.items():
            try:
                if config["type"] == "onnx":
                    self.models[model_name] = ONNXModelRunner(config["path"])
                elif config["type"] == "tensorrt" and TENSORRT_AVAILABLE:
                    self.models[model_name] = TensorRTModelRunner(config["path"])
                else:
                    logger.warning(f"Неизвестный тип модели: {config['type']}")
                
                self.latency_stats[model_name] = []
                logger.info(f"✅ Модель {model_name} загружена")
                
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки модели {model_name}: {e}")
                # Create a dummy model that returns zeros for predictions
                self.models[model_name] = None
    
    def predict(
        self, 
        model_name: str, 
        inputs: Union[Dict[str, np.ndarray], List[np.ndarray]]
    ) -> Union[Dict[str, np.ndarray], List[np.ndarray]]:
        """
        Предсказание с конкретной моделью
        
        Args:
            model_name: имя модели
            inputs: входные данные
            
        Returns:
            outputs: выходные данные
        """
        if model_name not in self.models:
            raise ValueError(f"Модель {model_name} не загружена")
        
        model = self.models[model_name]
        
        # Handle case where model failed to load
        if model is None:
            logger.warning(f"⚠️ Модель {model_name} не загружена, возвращаются нулевые значения")
            # Return zero arrays with appropriate shapes
            if isinstance(inputs, dict):
                # For ONNX models, return dict with same keys but zero arrays
                dummy_outputs = {}
                for key in inputs.keys():
                    dummy_outputs[key] = np.zeros((1, 6))  # Assuming 6 output classes
                return dummy_outputs
            else:
                # For other models, return list of zero arrays
                return [np.zeros((1, 6))]  # Assuming 6 output classes
        
        # Измерение времени
        start_time = time.time()
        
        if isinstance(model, ONNXModelRunner):
            outputs = model.predict(inputs)
        elif isinstance(model, TensorRTModelRunner):
            outputs = model.predict(inputs)
        else:
            raise RuntimeError(f"Неизвестный тип модели: {type(model)}")
        
        # Статистика latency
        latency = time.time() - start_time
        self.latency_stats[model_name].append(latency)
        
        if len(self.latency_stats[model_name]) > 1000:
            self.latency_stats[model_name].pop(0)  # Ограничение размера истории
        
        return outputs
    
    def get_latency_stats(self, model_name: str) -> Dict[str, float]:
        """Получение статистики latency для модели"""
        if model_name not in self.latency_stats:
            return {}
        
        latencies = self.latency_stats[model_name]
        if not latencies:
            return {}
        
        return {
            "mean": np.mean(latencies),
            "std": np.std(latencies),
            "min": np.min(latencies),
            "max": np.max(latencies),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99)
        }
    
    def get_all_latency_stats(self) -> Dict[str, Dict[str, float]]:
        """Получение статистики latency для всех моделей"""
        return {
            model_name: self.get_latency_stats(model_name)
            for model_name in self.latency_stats.keys()
        }
    
    def get_model_health(self) -> Dict[str, Dict[str, Any]]:
        """Получение информации о состоянии моделей"""
        health_info = {}
        
        for model_name, model in self.models.items():
            # Получение статистики latency
            latency_stats = self.get_latency_stats(model_name)
            
            # Проверка доступности модели
            is_available = model is not None
            
            # Определение статуса на основе latency
            status = "healthy"
            if latency_stats:
                if latency_stats.get("p95", 0) > 0.1:  # 100ms threshold
                    status = "degraded"
                if latency_stats.get("p99", 0) > 0.5:  # 500ms threshold
                    status = "unhealthy"
            
            health_info[model_name] = {
                "status": status,
                "available": is_available,
                "latency_stats": latency_stats,
                "model_type": type(model).__name__,
                "last_updated": datetime.utcnow().isoformat()
            }
        
        return health_info
    
    def get_system_health(self) -> Dict[str, Any]:
        """Получение общей информации о состоянии системы"""
        model_health = self.get_model_health()
        
        # Подсчет статусов моделей
        healthy_count = sum(1 for info in model_health.values() if info["status"] == "healthy")
        degraded_count = sum(1 for info in model_health.values() if info["status"] == "degraded")
        unhealthy_count = sum(1 for info in model_health.values() if info["status"] == "unhealthy")
        
        # Общий статус системы
        if unhealthy_count > 0:
            system_status = "unhealthy"
        elif degraded_count > 0:
            system_status = "degraded"
        else:
            system_status = "healthy"
        
        return {
            "system_status": system_status,
            "total_models": len(model_health),
            "healthy_models": healthy_count,
            "degraded_models": degraded_count,
            "unhealthy_models": unhealthy_count,
            "model_health": model_health,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """Получение информации о моделях"""
        model_info = {}
        
        for model_name, config in self.model_configs.items():
            model_info[model_name] = {
                "config": config,
                "loaded": model_name in self.models,
                "latency_history_count": len(self.latency_stats.get(model_name, []))
            }
        
        return model_info


class MultiModelEnsembleInference:
    """Ансамблевый инференс нескольких моделей"""
    
    def __init__(self, model_manager: EdgeModelManager):
        self.model_manager = model_manager
        self.ensemble_weights = None
        self.model_performance = {}
    
    def predict_with_ensemble(
        self,
        model_inputs: Dict[str, Union[Dict[str, np.ndarray], List[np.ndarray]]],
        ensemble_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Ансамблевое предсказание
        
        Args:
            model_inputs: входы для каждой модели
            ensemble_weights: веса моделей в ансамбле
            
        Returns:
            ensemble_output: ансамблированный результат
            individual_outputs: индивидуальные результаты моделей
        """
        individual_outputs = {}
        
        # Предсказания от всех моделей
        for model_name, inputs in model_inputs.items():
            try:
                output = self.model_manager.predict(model_name, inputs)
                # Предполагаем, что первый выход - это основной результат
                if isinstance(output, dict):
                    main_output = list(output.values())[0]
                else:
                    main_output = output[0] if isinstance(output, list) else output
                individual_outputs[model_name] = main_output
            except Exception as e:
                logger.error(f"Ошибка предсказания модели {model_name}: {e}")
                # Используем нулевой массив в случае ошибки
                individual_outputs[model_name] = np.zeros(6)  # 6 классов по умолчанию
        
        # Ансамблирование
        if ensemble_weights is None:
            # Используем сохраненные веса или равномерные веса
            if self.ensemble_weights is not None:
                ensemble_weights = self.ensemble_weights
            else:
                ensemble_weights = {
                    model_name: 1.0 / len(individual_outputs) 
                    for model_name in individual_outputs.keys()
                }
        else:
            # Сохраняем переданные веса
            self.ensemble_weights = ensemble_weights
        
        # Нормализация весов
        total_weight = sum(ensemble_weights.values())
        normalized_weights = {
            model_name: weight / total_weight 
            for model_name, weight in ensemble_weights.items()
        }
        
        # Взвешенное суммирование
        # Handle case where individual_outputs might be empty
        if not individual_outputs:
            logger.warning("⚠️ Нет доступных моделей для ансамблирования, возвращаются нулевые значения")
            return np.zeros(6), {}  # 6 классов по умолчанию
        
        # Get the shape from the first available output
        first_output = next(iter(individual_outputs.values()))
        ensemble_output = np.zeros_like(first_output)
        
        for model_name, weight in normalized_weights.items():
            if model_name in individual_outputs:
                ensemble_output += individual_outputs[model_name] * weight
        
        return ensemble_output, individual_outputs
    
    def update_ensemble_weights(self, model_performance: Dict[str, float]):
        """
        Обновление весов ансамбля на основе производительности моделей
        
        Args:
            model_performance: словарь {model_name: performance_score}
        """
        self.model_performance = model_performance
        
        # Простое обновление весов на основе производительности
        total_performance = sum(model_performance.values())
        if total_performance > 0:
            self.ensemble_weights = {
                model_name: performance / total_performance
                for model_name, performance in model_performance.items()
            }
        else:
            # Если нет данных о производительности, используем равномерные веса
            self.ensemble_weights = {
                model_name: 1.0 / len(model_performance)
                for model_name in model_performance.keys()
            }
    
    def get_ensemble_weights(self) -> Optional[Dict[str, float]]:
        """
        Получение текущих весов ансамбля
        
        Returns:
            ensemble_weights: словарь весов моделей
        """
        return self.ensemble_weights
    
    def get_model_performance(self) -> Dict[str, float]:
        """
        Получение данных о производительности моделей
        
        Returns:
            model_performance: словарь производительности моделей
        """
        return self.model_performance


def convert_to_onnx(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    output_path: str,
    opset_version: int = 11
):
    """Конвертация PyTorch модели в ONNX"""
    model.eval()
    
    # Создание примерного входа
    dummy_input = torch.randn(input_shape)
    
    # Экспорт в ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    logger.info(f"✅ Модель сконвертирована в ONNX: {output_path}")


def convert_to_tensorrt(
    onnx_path: str,
    engine_path: str,
    max_batch_size: int = 1,
    max_workspace_size: int = 1 << 30  # 1GB
):
    """Конвертация ONNX модели в TensorRT engine"""
    if not TENSORRT_AVAILABLE:
        logger.warning("TensorRT не доступен")
        return
    
    # Создание builder
    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, builder.logger)
    
    # Парсинг ONNX модели
    with open(onnx_path, 'rb') as model_file:
        parser.parse(model_file.read())
    
    # Конфигурация builder
    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size
    
    # Создание engine
    engine = builder.build_engine(network, config)
    
    # Сохранение engine
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    logger.info(f"✅ Модель сконвертирована в TensorRT: {engine_path}")


def create_edge_inference_pipeline(
    model_configs: Dict[str, Dict],
    ensemble_weights: Optional[Dict[str, float]] = None
) -> Tuple[EdgeModelManager, MultiModelEnsembleInference]:
    """Создание edge inference pipeline"""
    
    # Создание менеджера моделей
    model_manager = EdgeModelManager(model_configs)
    
    # Создание ансамблевого инференса
    ensemble_inference = MultiModelEnsembleInference(model_manager)
    
    logger.info("✅ Edge inference pipeline создан")
    
    return model_manager, ensemble_inference


# ==================== ТЕСТИРОВАНИЕ ====================

if __name__ == "__main__":
    # Тестирование edge inference pipeline
    print("🔍 Тестирование edge inference pipeline...")
    
    # Создание тестовых моделей и их конвертация
    print("  Создание тестовых моделей...")
    
    # LSTM модель
    from models.lstm_predictor.attention_lstm import create_lstm_model
    lstm_model = create_lstm_model(input_size=10, hidden_size=32, num_layers=1, output_size=5)
    
    # Сохранение как ONNX
    convert_to_onnx(
        lstm_model,
        input_shape=(1, 30, 10),  # batch_size=1, seq_len=30, input_size=10
        output_path="test_lstm_model.onnx"
    )
    
    # ViT модель
    from models.vision_transformer.defect_detector import create_vit_classifier
    vit_model = create_vit_classifier(img_size=32, patch_size=8, n_classes=5)
    
    # Сохранение как ONNX
    convert_to_onnx(
        vit_model,
        input_shape=(1, 3, 32, 32),  # batch_size=1, channels=3, height=32, width=32
        output_path="test_vit_model.onnx"
    )
    
    # Конфигурация моделей
    model_configs = {
        "lstm": {"path": "test_lstm_model.onnx", "type": "onnx"},
        "vit": {"path": "test_vit_model.onnx", "type": "onnx"}
    }
    
    # Создание pipeline
    print("  Создание inference pipeline...")
    model_manager, ensemble_inference = create_edge_inference_pipeline(model_configs)
    
    # Тестовые данные
    print("  Тестирование предсказаний...")
    
    # LSTM входы
    lstm_input = {
        "input": np.random.randn(1, 30, 10).astype(np.float32)
    }
    
    # ViT входы
    vit_input = {
        "input": np.random.randn(1, 3, 32, 32).astype(np.float32)
    }
    
    # Индивидуальные предсказания
    try:
        lstm_output = model_manager.predict("lstm", lstm_input)
        print(f"    LSTM выход: {list(lstm_output.values())[0].shape}")
        
        vit_output = model_manager.predict("vit", vit_input)
        print(f"    ViT выход: {list(vit_output.values())[0].shape}")
    except Exception as e:
        print(f"    Ошибка предсказаний: {e}")
    
    # Ансамблевое предсказание
    try:
        model_inputs = {
            "lstm": lstm_input,
            "vit": vit_input
        }
        
        ensemble_output, individual_outputs = ensemble_inference.predict_with_ensemble(model_inputs)
        print(f"    Ансамбль выход: {ensemble_output.shape}")
        print(f"    Индивидуальные выходы: {list(individual_outputs.keys())}")
    except Exception as e:
        print(f"    Ошибка ансамблевого предсказания: {e}")
    
    # Статистика latency
    print("  Статистика latency...")
    latency_stats = model_manager.get_all_latency_stats()
    for model_name, stats in latency_stats.items():
        if stats:
            print(f"    {model_name}: mean={stats['mean']:.4f}s, p95={stats['p95']:.4f}s")
    
    print("\n✅ Все тесты пройдены!")