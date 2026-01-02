import sys
import os
import json
import warnings
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# 경고 메시지 무시
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 1. 프로젝트 루트 경로 설정 및 모듈 import
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rag_engine_path = os.path.join(project_root, '04_RAG_ENGINE')
sys.path.append(rag_engine_path)

from rag_modules import ConsultantAgent

# 2. 환경변수 로드
load_dotenv(os.path.join(project_root, '.env'))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def run_evaluation():
    # 3. 데이터셋 로드
    with open('05_EVALUATE/test_dataset.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 4. 데이터 수집 (질문 -> 챗봇 -> 답변/맥락 저장)
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print("🚀 평가 데이터 생성 중...")
    
    for item in raw_data:
        q = item['question']
        print(f"Processing: {q}")
        
        # 매번 새로운 세션으로 평가하기 위해 Agent 재생성
        agent = ConsultantAgent()
        
        # with_context=True로 설정하여 검색된 문서 리스트까지 받음
        response, retrieved_docs = agent.run(q, with_context=True)
        
        questions.append(q)
        answers.append(response)
        contexts.append(retrieved_docs)
        
        # Ground Truth 처리
        gt = item.get('notes', "") + " " + " ".join(item.get('check_points', []))
        
        # [수정] 리스트([])를 제거하고 문자열(gt) 그대로 추가
        ground_truths.append(gt) 

    # 5. RAGAS 데이터셋 포맷 변환
    data_dict = {
        "user_input": questions,    # RAGAS 최신 버전 호환을 위해 키 이름 변경 권장 (question -> user_input)
        "response": answers,        # (answer -> response)
        "retrieved_contexts": contexts, # (contexts -> retrieved_contexts)
        "reference": ground_truths  # (ground_truth -> reference)
    }
    dataset = Dataset.from_dict(data_dict)

    # 6. 평가 모델 설정
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        temperature=0,
        google_api_key=GEMINI_API_KEY
    )
    gemini_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=GEMINI_API_KEY
    )

    # 7. 평가 실행
    print("\n📊 RAGAS 평가 시작 (시간이 좀 걸립니다)...")
    results = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),      # 할루시네이션 체크
            AnswerRelevancy(),   # 질문 관련성 체크
        ],
        llm=gemini_llm,
        embeddings=gemini_embeddings
    )

    # 8. 결과 저장
    print("\n✅ 평가 완료!")
    df = results.to_pandas()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ragas_report.xlsx")
    df.to_excel(output_path, index=False)
    print(f"결과가 '{output_path}'에 저장되었습니다.")

if __name__ == "__main__":
    run_evaluation()