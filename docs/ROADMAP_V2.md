# CryptoIA v2 — Единая дорожная карта

Цель: перейти от реактивной модели к адаптивной AI‑платформе с точностью сигналов >70% за счёт улучшенного RT‑пайплайна, когнитивного слоя и многоагентной аналитики.

## 1) Фундамент: усиление Real‑Time

- Триггеры (services/pipeline/src/pipeline/rt/trigger_agent.py)
  - [СДЕЛАНО] L2‑стенки и дисбаланс: `L2_WALL`, `L2_IMBALANCE`.
  - [СДЕЛАНО] Всплеск объёма и дельты: `VOL_SPIKE`, `DELTA_SPIKE`.
  - [ДОБАВЛЕНО] Волатильность (ATR): `ATR_SPIKE` — ATR(1m) > avg(1h) × `TRIGGER_ATR_FACTOR`.
  - [ДОБАВЛЕНО] Импульс (5м): `MOMENTUM` — |ΔP(5m)| ≥ `TRIGGER_MOMENTUM_5M_PCT`.
  - [ДОБАВЛЕНО] Свечные паттерны 5м: `PATTERN_*` (поглощение, молот, падающая звезда).
  - [ДОБАВЛЕНО] Деривативы: `DERIV_OI_JUMP`, `DERIV_FUNDING` (open interest jump / |funding| ≥ порога).

- Реактор (services/pipeline/src/pipeline/rt/master_reactor.py)
  - [СДЕЛАНО] Быстрый контекст (features_calc → модели → ансамбль → карточка).
  - [ДОБАВЛЕНО] Композитная уверенность через модуль `trading/confidence.py` (конфиг через `agent_configurations`), факторы логируются в карточке (`confidence_factors`).
  - [ДОБАВЛЕНО] Мгновенная генерация графика и ссылка (reporting/charts.py) с уровнями SL/TP.
  - [ДОБАВЛЕНО] Иерархия алертов (композиция триггеров за окно ~3мин) в `rt/alert_priority.py`; приоритет выводится в алерте.

Параметры ENV ключевых триггеров:

- `TRIGGER_ATR_FACTOR` (по умолч. 3.0)
- `TRIGGER_MOMENTUM_5M_PCT` (по умолч. 0.005 = 0.5%)
- `TRIGGER_DERIV_OI_JUMP_PCT` (по умолч. 0.05 = 5%)
- `TRIGGER_DERIV_FUNDING_ABS` (по умолч. 0.01 = 1%)
- `RT_CHART_ENABLE` (1/0)
- `RT_ALERT_WINDOW_SEC` (окно агрегации триггеров, по умолчанию 180)

## 2) Мозг: самообучение

- [СДЕЛАНО] База техпаттернов + метрики открытия паттернов (migrations 025,027; agents/pattern_discovery.py).
- [PLANNED] Knowledge Core (RAG) с учебной теорией (книги/статьи/стратегии) для LLM‑агентов.
- [PLANNED] Ежемесячный «Когнитивный Архитектор»: анализ winrate и пересчёт весов факторов; запись уроков в `agent_lessons` (модель есть).

## 3) Результат: карточка сделки

- [СДЕЛАНО] Карточка с SL/TP на основе ATR, риск‑менеджмент, верификатор.
- [ДОБАВЛЕНО] Поле `confidence` теперь учитывает контекстные бонусы (триггеры/режим/TA‑сентимент).
- [PLANNED] Полная модель Confidence Score с весами из `agent_configurations` и факторной раскладкой.

## 4) Multi‑Agent слой (план)

- [СКЕЛЕТ] Агент «SMC Analyst» — `agents/smc_analyst.py` (HTF/LTF логика; сохраняет вердикт в S3; интеграция с БД требует миграции).
- [СКЕЛЕТ] Агент «Whale Watcher» — `agents/whale_watcher.py` (простые эвристики; сохраняет вердикт в S3; интеграция с БД требует миграции).
- [PLANNED] Интеграция выводов агентов в Reactor и Confidence Score.

## Текущий прогресс и следующие шаги

- Что сделано:
  - Исправлен баг в charts.py (опечатка импорта).
  - Реализованы новые RT‑триггеры: ATR, импульс 5м, свечные паттерны 5м, аномалии деривативов.
  - В Reactor добавлен график и композитное усиление confidence через отдельный модуль; логируем вклад факторов.
  - Иерархия алертов по композиции триггеров — приоритет в алерте.
  - Скелеты агентов SMC/Whale; месячный адаптер `ops/monthly_adaptation.py`.

- Что осталось:
  1) Доделать интеграцию SMC/Whale с БД (таблицы и upsert‑функции), запрос последних вердиктов в Reactor.
  2) Добавить правила компоновки алертов для «critical» (например, импульс+объём+пробой уровня/паттерн).
  3) Knowledge Core (RAG) — реализовать загрузчик/эмбеддер, поиск (pgvector) и подключение к reasoning.
  4) «Когнитивный Архитектор» — расширить анализ (winrate стратегий/режимов), авто‑обновление конфигов и записи уроков.

## Файлы и точки расширения

- RT триггеры: `services/pipeline/src/pipeline/rt/trigger_agent.py`
- RT реактор: `services/pipeline/src/pipeline/rt/master_reactor.py`
- Графики: `services/pipeline/src/pipeline/reporting/charts.py`
- Фичи и индикаторы: `services/pipeline/src/pipeline/features/features_calc.py`
- Паттерны: `services/pipeline/src/pipeline/features/features_patterns.py`
- ОИ/Фандинг: `services/pipeline/src/pipeline/data/ingest_futures.py`
- Конфиги агентов/веса: `services/pipeline/src/pipeline/infra/db.py` (agent_configurations)

## Краткие рекомендации (бизнес/продукт)

- Начать с закрытой беты RT‑сигналов с уровнем приоритета и ссылкой на RT‑график для повышения доверия.
- A/B‑эксперименты весов Confidence Score по сегментам рынка (тренд/флэт/высокая вола).
- Ежемесячный публичный отчёт точности и прозрачный разбор лучших/худших сетапов.

## Управление зависимостями

- Раз в квартал проводить обновление ключевых пакетов (`numpy`, `pandas`, `ccxt`, `python-telegram-bot`) с запуском полного `pytest` и smoke-тестов RT пайплайна.
- После успешного обновления фиксировать версии в `requirements.txt` / `requirements-dev.txt` (PR с пояснением изменений и найденных несовместимостей).
- Для критических обновлений безопасности (PSRT/OSV) выпускать hotfix вне очереди и документировать в changelog.
