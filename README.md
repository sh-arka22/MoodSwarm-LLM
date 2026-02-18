# MoodSwarm: LLM Twin & MLOps Platform

> **Goal:** Build an end-to-end LLM system that mimics a specific persona's writing style using the FTI (Feature, Training, Inference) architecture.

---

## 📅 Project Progress Log

### ✅ Week 1: Infrastructure Foundation
**Objective:** Set up a scalable, reproducible MLOps environment.
- **Tech Stack:** Python 3.11, Poetry, Docker, ZenML.
- **Achievements:**
    - Established **Domain-Driven Design (DDD)** folder structure (`domain/`, `application/`, `infrastructure/`).
    - Configured **Docker Compose** for persistence layer (MongoDB + Qdrant).
    - Initialized **ZenML** as the orchestration engine.
    - Implemented **Pydantic Settings** management for type-safe configuration.
    - Created a **Smoke Test Pipeline** to verify database connectivity automatically.

### ✅ Week 2: Digital Data ETL Pipeline
**Objective:** Build the ingestion engine to scrape and normalize unstructured data.
- **Architecture:** `Dispatcher` -> `Worker` pattern.
- **Achievements:**
    - **Domain Modeling:** Designed MongoDB ODM models (`User`, `Article`, `Repository`) with strict validation.
    - **Crawlers:** Built modular scrapers for:
        - **GitHub** (Clones repos, extracts code).
        - **Medium** (Selenium headless browser).
        - **Custom/Substack** (LangChain HTML parsing).
    - **Design Patterns Applied:**
        - **Strategy Pattern:** `BaseCrawler` interface for extensible workers.
        - **Factory Pattern:** `CrawlerDispatcher` for dynamic worker selection.
        - **Singleton Pattern:** `MongoDatabaseConnector` for efficient connection pooling.
    - **Pipeline:** Implemented `digital_data_etl` in ZenML to orchestrate user creation and crawling.
    - **Hardening:** Added **Exponential Backoff** (retry logic) and Deduplication checks.

### 📊 Visual Architecture

**1. High-Level System Architecture**
```mermaid
graph LR
    subgraph Sources
        GH[GitHub]
        MD[Medium/Substack]
    end

    subgraph "ETL Pipeline (Week 2)"
        Dispatcher[Crawler Dispatcher]
        Worker[Crawlers]
    end

    subgraph "Data Warehouse"
        Mongo[(MongoDB - NoSQL)]
        Qdrant[(Qdrant - Vectors)]
    end

    GH --> Dispatcher
    MD --> Dispatcher
    Dispatcher --> Worker
    Worker --> Mongo
    Mongo -.-> |Week 3| Qdrant
```

**2. ETL Pipeline Execution Flow**
```mermaid
sequenceDiagram
    participant CLI as tools.run
    participant ZenML as ZenML Pipeline
    participant Step1 as get_or_create_user
    participant Step2 as crawl_links
    participant DB as MongoDB

    CLI->>ZenML: Trigger --run-etl
    ZenML->>Step1: Execute (Paul Iusztin)
    Step1->>DB: Check User Exists?
    DB-->>Step1: Return User ID
    Step1->>ZenML: Return User Object
    
    ZenML->>Step2: Execute (User, Links)
    loop For each URL
        Step2->>Step2: Dispatcher finds Crawler
        Step2->>Step2: Download & Clean Data
        Step2->>DB: Save Article (deduplicated)
    end
    Step2-->>ZenML: Success Signal
```

---

## 🚀 How to Run

### 1. Start Infrastructure
```bash
docker-compose up -d
```

### 2. Run the ETL Pipeline
```bash
poetry run python -m tools.run --run-etl
```

### 3. Verification
Check the ZenML dashboard:
```bash
poetry run zenml login --local
```
