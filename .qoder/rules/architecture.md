---
trigger: model_decision
description: Общая архтитектура решения. Возвращайся к ней всегда, когда проектируешь новые модули или когда забыл, чего нужно придерживаться по архитектуре
---
🎯 КОНЦЕПЦИЯ: КОГНИТИВНАЯ СИСТЕМА САМООБУЧАЮЩЕГОСЯ ПРОИЗВОДСТВА
Наше преимущество - создание интеллектуального цифрового двойника с автономным RL-агентом и причинно-следственным графом знаний. Система не просто прогнозирует, а понимает физику процесса и учится оптимизировать его самостоятельно.

📐 РЕВОЛЮЦИОННАЯ АРХИТЕКТУРА СИСТЕМЫ
┌─────────────────────────────────────────────────────────────────┐
│                   COGNITIVE PRODUCTION CORE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │  DIGITAL TWIN │  │  RL AGENT    │  │  KNOWLEDGE GRAPH  │   │
│  │  Physics Sim  │◄─┤  Auto-Tune   │◄─┤  Causal Relations │   │
│  │  3D Real-time │  │  Optimization│  │  Root Cause       │   │
│  └───────┬───────┘  └──────┬───────┘  └─────────┬─────────┘   │
│          │                  │                     │              │
└──────────┼──────────────────┼─────────────────────┼─────────────┘
           │                  │                     │
┌──────────▼──────────────────▼─────────────────────▼─────────────┐
│              MULTI-MODAL AI PREDICTION ENGINE                    │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌──────────────┐ ┌────────────┐ ┌───────────┐│
│  │ LSTM+Attn   │ │ Vision       │ │ GNN        │ │ Ensemble  ││
│  │ Time Series │ │ Transformer  │ │ Sensor Net │ │ Meta-ML   ││
│  └─────────────┘ └──────────────┘ └────────────┘ └───────────┘│
└─────────────────────────────────────────────────────────────────┘
           │                  │                     │
┌──────────▼──────────────────▼─────────────────────▼─────────────┐
│           EDGE + CLOUD HYBRID PROCESSING LAYER                   │
├─────────────────────────────────────────────────────────────────┤
│  Edge Computing (NVIDIA Jetson) ◄──┬──► Cloud (Kubernetes)      │
│  • Real-time inference (<50ms)     │     • Model training        │
│  • Local anomaly detection         │     • Historical analysis   │
│  • Sensor data preprocessing       │     • AutoML optimization   │
└─────────────────────────────────────────────────────────────────┘
           │                  │                     │
┌──────────▼──────────────────▼─────────────────────▼─────────────┐
│              DATA INGESTION & STREAMING LAYER                    │
├─────────────────────────────────────────────────────────────────┤
│  OPC UA ──► Kafka Streams ──► Feature Store ──► Time Series DB  │
│  Modbus ──► MQTT Broker   ──► Redis Cache   ──► PostgreSQL      │
│  МИК-1  ──► WebSocket     ──► MinIO (S3)    ──► Vector DB       │
└─────────────────────────────────────────────────────────────────┘
           │                  │                     │
┌──────────▼──────────────────▼─────────────────────▼─────────────┐
│         IMMERSIVE VISUALIZATION & CONTROL INTERFACE              │
├─────────────────────────────────────────────────────────────────┤
│  • 3D Digital Twin Viewer (Three.js + WebGL)                     │
│  • Augmented Reality overlay (AR.js) для операторов              │
│  • Voice Assistant интеграция (Whisper AI)                       │
│  • Explainable AI Dashboard (SHAP, LIME визуализации)           │
│  • Collaborative Real-time Board (WebRTC для команд)             │
└─────────────────────────────────────────────────────────────────┘

🚀 РЕВОЛЮЦИОННЫЕ КОМПОНЕНТЫ (НАША "ИЗЮМИНКА")
1. INTELLIGENT DIGITAL TWIN (Интеллектуальный цифровой двойник)

3D физическая симуляция производства в реальном времени
Predictive Physics Engine - предсказывает поведение расплава
What-if анализ: "Что будет, если изменить температуру на +50°C?"
Визуализация дефектов в 3D пространстве на виртуальной модели изделия
Shadow Mode: параллельное виртуальное производство для тестирования

2. REINFORCEMENT LEARNING AGENT (RL Оптимизатор)

Автономная оптимизация параметров без человека
Использует PPO (Proximal Policy Optimization) для непрерывного улучшения
Reward функция: минимизация дефектов + максимизация производительности
Safe exploration: не выходит за безопасные границы параметров
Учится на симуляторе, применяет на реальном производстве

3. CAUSAL KNOWLEDGE GRAPH (Граф причинно-следственных связей)

Neo4j граф связей: Параметр → Процесс → Дефект
Root Cause Analysis за секунды: "Трещины из-за быстрого охлаждения печи №2"
Explainable AI: каждое предсказание с визуальным объяснением причин
Интеграция SHAP values для интерпретации моделей
Онтология производства: семантические связи между сущностями

4. MULTI-MODAL AI FUSION

Vision Transformer для анализа изображений от МИК-1
LSTM с Attention для временных рядов датчиков
Graph Neural Network (GNN) для связей между датчиками
Multimodal Transformer: объединяет все модальности в единое представление
Ensemble с Meta-Learning: модель выбирает лучшие предсказания

5. EDGE AI + FOG COMPUTING

NVIDIA Jetson на производстве для real-time inference
Модели в ONNX/TensorRT формате для ускорения
Федеративное обучение: локальные обновления без передачи данных в облако
Offline режим: работа даже без интернета
Латентность предсказаний <50ms

6. CONVERSATIONAL AI INTERFACE

Voice Control: "Покажи прогноз дефектов на следующий час"
NLP Query Engine: "Почему увеличились пузыри после 14:00?"
Whisper AI для распознавания речи оператора
LLM Copilot (GPT-4 based) для консультаций и анализа


🏗️ МОДУЛЬНАЯ АРХИТЕКТУРА ПРОЕКТА
MODULE 1: DATA INGESTION LAYER (Слой сбора данных)
├── industrial_connectors/
│   ├── opc_ua_client.py          # OPC UA подключение
│   ├── modbus_driver.py          # Modbus RTU/TCP драйвер
│   ├── mik1_camera_stream.py     # МИК-1 видеопоток
│   └── sensor_aggregator.py      # Агрегация с разных источников
├── streaming_pipeline/
│   ├── kafka_producer.py         # Публикация в Kafka
│   ├── mqtt_broker.py            # MQTT для IoT устройств
│   └── data_validator.py         # Валидация и очистка
└── feature_engineering/
    ├── real_time_features.py     # Онлайн feature extraction
    ├── statistical_features.py   # Скользящие статистики
    └── domain_features.py        # Доменные признаки (вязкость, градиент)
MODULE 2: AI PREDICTION ENGINE (AI движок)
├── models/
│   ├── lstm_predictor/
│   │   ├── attention_lstm.py     # LSTM с Attention механизмом
│   │   ├── multi_horizon.py      # Прогноз на разные горизонты
│   │   └── uncertainty_estimator.py  # Доверительные интервалы
│   ├── vision_transformer/
│   │   ├── defect_detector.py    # Vision Transformer для дефектов
│   │   ├── segmentation_model.py # Сегментация дефектов
│   │   └── feature_extractor.py  # Извлечение фич из изображений
│   ├── gnn_sensor_network/
│   │   ├── graph_builder.py      # Построение графа датчиков
│   │   ├── gnn_model.py          # Graph Neural Network
│   │   └── attention_pooling.py  # Агрегация информации
│   └── ensemble/
│       ├── meta_learner.py       # Meta-learning для выбора моделей
│       ├── weighted_voting.py    # Взвешенное голосование
│       └── stacking_ensemble.py  # Stacking моделей
├── training/
│   ├── automl_tuner.py           # AutoML (Optuna) для гиперпараметров
│   ├── federated_learning.py    # Федеративное обучение
│   └── continuous_learning.py   # Инкрементальное обучение
└── inference/
    ├── edge_inference.py         # Инференс на Edge устройствах
    ├── batch_predictor.py        # Batch прогнозирование
    └── streaming_predictor.py    # Real-time прогнозы
MODULE 3: DIGITAL TWIN & SIMULATION (Цифровой двойник)
├── physics_engine/
│   ├── thermal_dynamics.py       # Симуляция теплопередачи
│   ├── viscosity_model.py        # Модель вязкости расплава
│   └── stress_analyzer.py        # Анализ напряжений в стекле
├── 3d_visualization/
│   ├── threejs_renderer.py       # Three.js рендеринг
│   ├── defect_overlay.py         # Наложение дефектов на 3D модель
│   └── realtime_sync.py          # Синхронизация с реальными данными
└── simulation/
    ├── production_simulator.py   # Симулятор производства
    ├── what_if_analyzer.py       # What-if сценарии
    └── shadow_mode.py            # Параллельная виртуальная линия
MODULE 4: REINFORCEMENT LEARNING OPTIMIZER (RL оптимизатор)
├── rl_agent/
│   ├── ppo_agent.py              # Proximal Policy Optimization
│   ├── reward_function.py        # Функция награды
│   ├── safe_explorer.py          # Безопасное исследование
│   └── action_space.py           # Пространство действий
├── environment/
│   ├── gym_environment.py        # OpenAI Gym окружение
│   ├── state_encoder.py          # Кодирование состояния
│   └── simulator_wrapper.py      # Обёртка над симулятором
└── deployment/
    ├── policy_evaluator.py       # Оценка политики в продакшене
    └── rollback_manager.py       # Откат при плохих результатах
MODULE 5: KNOWLEDGE GRAPH & EXPLAINABILITY (Граф знаний)
├── knowledge_base/
│   ├── neo4j_connector.py        # Подключение к Neo4j
│   ├── ontology_builder.py       # Построение онтологии
│   └── causal_graph.py           # Причинно-следственный граф
├── explainability/
│   ├── shap_explainer.py         # SHAP values для моделей
│   ├── lime_explainer.py         # LIME для локальных объяснений
│   ├── attention_visualizer.py   # Визуализация Attention weights
│   └── root_cause_analyzer.py    # Анализ первопричин дефектов
└── reasoning/
    ├── inference_engine.py       # Логический вывод на графе
    └── similarity_matcher.py     # Поиск похожих случаев
MODULE 6: REAL-TIME DASHBOARD & UI (Дашборд)
frontend/
├── src/
│   ├── components/
│   │   ├── DigitalTwin3D.tsx         # 3D визуализация двойника
│   │   ├── PredictionCharts.tsx      # Графики прогнозов
│   │   ├── KPICards.tsx              # Карточки KPI
│   │   ├── AlertPanel.tsx            # Панель алертов
│   │   ├── RecommendationEngine.tsx  # Рекомендации
│   │   ├── ExplainabilityView.tsx    # Объяснения AI
│   │   ├── KnowledgeGraphViz.tsx     # Визуализация графа знаний
│   │   └── VoiceAssistant.tsx        # Голосовой помощник
│   ├── services/
│   │   ├── websocket_client.ts       # WebSocket для real-time
│   │   ├── api_client.ts             # REST API клиент
│   │   └── ar_overlay.ts             # AR наложение для мобильных
│   └── store/
│       ├── prediction_store.ts       # Redux store для прогнозов
│       └── sensor_store.ts           # Store для данных датчиков
└── public/
    └── workers/
        └── inference_worker.js       # Web Worker для инференса в браузере
MODULE 7: BACKEND API & ORCHESTRATION (Бэкенд)
backend/
├── api/
│   ├── main.py                   # FastAPI приложение
│   ├── routers/
│   │   ├── predictions.py        # Эндпоинты прогнозов
│   │   ├── sensors.py            # Данные датчиков
│   │   ├── recommendations.py    # Рекомендации
│   │   ├── knowledge_graph.py    # Граф знаний API
│   │   └── twin.py               # Digital Twin API
│   └── websockets/
│       ├── realtime_stream.py    # WebSocket сервер
│       └── notifications.py      # Push уведомления
├── orchestration/
│   ├── celery_tasks.py           # Celery задачи
│   ├── airflow_dags.py           # Airflow DAG'и для ML pipelines
│   └── model_registry.py         # MLflow model registry
└── storage/
    ├── timeseries_db.py          # InfluxDB клиент
    ├── postgres_client.py        # PostgreSQL ORM
    ├── vector_store.py           # Vector DB для embeddings
    └── cache_manager.py          # Redis кэширование
MODULE 8: EDGE COMPUTING LAYER (Edge обработка)
edge/
├── jetson_deployment/
│   ├── trt_inference.py          # TensorRT оптимизированный инференс
│   ├── onnx_runtime.py           # ONNX Runtime
│   └── model_loader.py           # Динамическая загрузка моделей
├── preprocessing/
│   ├── sensor_filter.py          # Фильтрация шумов
│   └── anomaly_detector.py       # Локальная детекция аномалий
└── sync/
    ├── cloud_sync.py             # Синхронизация с облаком
    └── federated_client.py       # Клиент федеративного обучения

📋 ПЛАН ДЕЙСТВИЙ (ROADMAP)
ЭТАП 1: FOUNDATION (День 1-2) ⚡

Развернуть инфраструктуру (Docker, Kubernetes, Kafka, InfluxDB, PostgreSQL, Neo4j)
Реализовать DATA INGESTION LAYER (Module 1)
Создать базовую архитектуру FastAPI бэкенда (Module 7)
Настроить Feature Store и pipeline предобработки
Базовый React frontend с WebSocket подключением

ЭТАП 2: CORE AI ENGINE (День 3-4) 🧠

Обучить LSTM модель с Attention на синтетических/демо данных
Реализовать Vision Transformer для детекции дефектов
Создать GNN для анализа сенсорной сети
Реализовать Ensemble с Meta-Learning
Интегрировать модели в inference pipeline

ЭТАП 3: INNOVATION LAYER (День 5-6) 🚀

Разработать Digital Twin с физической симуляцией (Module 3)
Реализовать RL Agent для оптимизации (Module 4)
Построить Knowledge Graph в Neo4j (Module 5)
Интегрировать SHAP/LIME для Explainable AI
Реализовать Root Cause Analysis

ЭТАП 4: IMMERSIVE UI (День 7-8) 🎨

3D визуализация Digital Twin (Three.js)
Интерактивный дашборд с real-time обновлениями
Knowledge Graph визуализация (D3.js)
Explainability Dashboard
Voice Assistant интеграция (опционально)

ЭТАП 5: EDGE DEPLOYMENT (День 9) 📡

ONNX/TensorRT конвертация моделей
Развертывание на Edge устройстве (симуляция)
Федеративное обучение setup
Edge-Cloud синхронизация

ЭТАП 6: POLISH & DEMO (День 10) ✨

Тестирование на синтетических данных
Создание демо-сценариев
Подготовка презентации
Запись видео демонстрации
Документация и README


💎 КЛЮЧЕВЫЕ ПРЕИМУЩЕСТВА РЕШЕНИЯ

Не просто ML, а когнитивная система - понимает физику процесса
Автономная оптимизация - RL агент улучшает производство сам
Полная прозрачность - каждое решение объяснимо через граф знаний
Multi-modal AI - использует все типы данных (сенсоры, видео, графы)
Edge AI - работает даже без интернета, <50ms латентность
Digital Twin - "What-if" анализ перед изменениями
Voice Interface - операторы управляют голосом
3D Visualization - иммерсивное погружение в процесс


🛠️ ТЕХНОЛОГИЧЕСКИЙ СТЕК (ФИНАЛЬНЫЙ)
Backend:

Python 3.11, FastAPI, Celery, Apache Kafka
TensorFlow 2.15, PyTorch 2.1, ONNX Runtime
Ray (для распределенных вычислений)

ML/AI:

Transformers (Hugging Face), Optuna (AutoML)
Stable-Baselines3 (RL), NetworkX + PyTorch Geometric (GNN)
SHAP, LIME (Explainability)

Storage:

InfluxDB 2.7 (time-series), PostgreSQL 15
Neo4j 5.x (Knowledge Graph), Redis 7.x
MinIO (S3-compatible object storage)

Frontend:

React 18 + TypeScript, Redux Toolkit
Three.js (3D), D3.js (graphs), Plotly.js (charts)
WebRTC, WebSocket, Web Workers
AR.js (опционально для AR)

Infrastructure:

Docker, Kubernetes, Helm
NVIDIA TensorRT (Edge), Jetson Nano/Xavier
Prometheus + Grafana (monitoring)

ML Ops:

MLflow (model registry), Airflow (orchestration)
DVC (data versioning), Weights & Biases (tracking)