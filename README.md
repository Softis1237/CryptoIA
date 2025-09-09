Crypto Forecast Agent Network v2 — Project Guide

Этот проект — многоагентная система прогнозирования BTC с оркестратором, сбором данных, фичами, моделями, ансамблями, reasoning, валидацией и торговыми компонентами. Документация разбита по файлам:

    docs/ARCHITECTURE.md — архитектура, главные модули, потоки данных
    docs/AGENTS.md — список агентов и флаги окружения
    docs/DATA_SOURCES.md — сбор данных, форматы, пути S3, ключи API
    docs/ENV.md — переменные окружения и значения по умолчанию
    docs/OPERATIONS.md — запуск, наблюдаемость, алерты, миграции
    docs/TRADING.md — live‑исполнение, оптимизатор и risk‑loop

Мини‑шпаргалка запуска:

    Скопируйте .env.example → .env, задайте ключи и PROM_PUSHGATEWAY_URL (если нужен Prometheus).
    Поднимите инфраструктуру: docker compose up -d --build
    Запустите планировщик: docker compose up -d scheduler (или вручную через agent_flow/Windmill).
    Дашборды: docker compose up -d prometheus grafana, Grafana на http://localhost:3001.

Подробнее — в соответствующих документах.
