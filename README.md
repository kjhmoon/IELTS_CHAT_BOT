
---

# IELTS Academy Neuro-Symbolic AI Agent

## 1. 프로젝트 개요

본 프로젝트는 IELTS 어학원 입학 상담 및 수강 문의 처리를 위한 대화형 AI 에이전트 구축을 목표로 함. 단순 RAG(Retrieval-Augmented Generation) 방식을 넘어 LLM의 추론 능력과 명시적 프로그래밍 로직을 결합한 신경-심볼릭(Neuro-Symbolic) 아키텍처를 채택함. 이를 통해 할루시네이션을 억제하고 비즈니스 요구사항을 정확히 수행하는 신뢰성 높은 상담 시스템을 구현함.

## 2. 시스템 아키텍처

시스템은 모듈화된 4개의 핵심 컴포넌트로 구성되며 각 모듈은 독립적인 역할과 유기적인 상호작용을 수행함.

### 2.1 Semantic Router (의도 분류기)

* **역할:** 사용자 입력의 의미론적 분석을 통한 의도(Intent) 분류
* **분류 카테고리:**
* `TIMETABLE`: 수업 시간표 및 커리큘럼 문의
* `REVIEW`: 수강 후기 및 성공 사례 문의
* `FAQ`: 환불 및 주차 등 행정 정보 문의
* `CHIT_CHAT`: 단순 잡담 및 비즈니스 무관 질문


* **특이 사항:** 프롬프트 엔지니어링을 통해 사용자 입력 내 필수 정보(Slot) 추출 병행

### 2.2 Chat Memory (상태 관리자)

* **역할:** 대화 문맥 유지 및 사용자 프로필 관리
* **기능:**
* Sliding Window 방식의 대화 히스토리 관리
* 사용자 정보(현재 점수 및 목표 점수 등)의 지속적 갱신(Slot Filling)
* 휘발성 메모리 구조 채택



### 2.3 Hybrid Retriever (정보 검색기)

* **역할:** 벡터 유사도 기반 관련 문서 추출
* **기술:**
* ChromaDB 활용 벡터 인덱싱
* Kiwi 형태소 분석기 활용 전처리
* 사용자 프로필 정보를 포함한 쿼리 확장(Query Expansion) 수행



### 2.4 Consultant Agent (상담 실행기)

* **역할:** 최종 답변 생성 및 제어 로직 수행
* **제어 로직:**
* **Guardrails:** 잡담 및 보안 위협 질문 차단
* **Missing Slot Check:** 필수 정보 누락 시 검색 중단 및 역질문 생성
* **Fallback:** 검색 결과 부재 시 대안 검색 수행



## 3. 기술 스택

* **Language Model:** Google Gemini 2.0 Flash
* **Orchestration:** LangChain
* **Vector Database:** ChromaDB
* **NLP Tool:** Kiwi (Korean Morphological Analyzer)
* **Evaluation:** RAGAS Framework
* **Language:** Python 3.10+

## 4. 디렉토리 구조

```text
IELTS_CHAT_BOT/
├── 01_FAQ/             # FAQ 데이터 전처리 및 임베딩 스크립트
├── 02_REVIEW/          # 수강 후기 데이터 처리 및 벡터화
├── 03_TIMETABLE/       # 시간표 데이터 구조화 및 DB 적재
├── 04_RAG_ENGINE/      # 핵심 엔진 (Router, Retriever, Agent) 구현체
├── 05_EVALUATE/        # RAGAS 기반 평가 파이프라인 및 데이터셋
└── chroma_db/          # 벡터 데이터베이스 저장소

```

## 5. 핵심 구현 로직

### 5.1 슬롯 필링 및 되묻기 (Slot Filling & Ask More)

사용자가 시간표를 문의할 때 `현재 점수` 또는 `목표 점수` 등 필수 정보가 누락된 경우 DB 검색을 수행하지 않음. 대신 누락된 정보를 파악하여 사용자에게 역질문하는 로직을 우선 수행함. 이는 부정확한 정보 제공을 방지하고 상담 품질을 높이기 위함임.

### 5.2 잡담 및 이상 질문 제어 (Guardrails)

상담 범위를 벗어난 주제나 경쟁사 비방 유도 등의 질문이 입력되면 사전 정의된 Steering Prompt가 작동함. 이를 통해 에이전트는 정중하고 건조한 어조로 답변을 거부하며 AI 상담원으로서의 정체성을 유지함.

### 5.3 물리적 제약 조건 처리

사용자의 거주지나 요청 시간이 물리적으로 통학 불가능한 경우(예: 해외 거주 또는 심야 시간) 이를 Router 단계에서 감지함. 현장 강의 대신 온라인 강의(VOD)를 소극적으로 제안하거나 불가능함을 명시하는 로직이 적용됨.

## 6. 성능 평가 (Evaluation)

RAGAS 프레임워크를 도입하여 정량적 성능 평가를 수행함.

* **평가 데이터:** `test_dataset.json` (의도별 시나리오 정의)
* **주요 지표:**
* **Faithfulness:** 검색된 문서에 근거한 답변 생성 여부
* **Answer Relevancy:** 사용자 질문 의도와의 부합도


* **실행 방식:** `run_ragas.py` 스크립트를 통해 자동화된 평가 리포트(`ragas_report.xlsx`) 생성