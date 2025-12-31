import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ------------------------------------------------------------------
# [경로 설정]
# ------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
env_path = os.path.join(parent_dir, '.env')

load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(f"API Key not found. Checked path: {env_path}")

client = genai.Client(api_key=api_key)

EMBEDDING_MODEL = 'models/text-embedding-004'

# 입출력 파일
INPUT_FILE = os.path.join(current_dir, 'structured_reviews.json')
OUTPUT_FILE = os.path.join(current_dir, 'review_db_ready.json')

def create_embedding_payload(review_data):
    """
    구조화된 수강후기를 받아 ChromaDB용 [ID, Vector, Metadata]로 변환합니다.
    """
    data = review_data
    
    # 리스트 포장 벗기기
    if isinstance(data, list):
        data = data[0] if len(data) > 0 else None
    
    if not isinstance(data, dict):
        return None

    # 1. 임베딩할 텍스트 생성 (Serialize)
    criteria = data.get('search_criteria', {})
    display = data.get('display_info', {})
    facts = data.get('fact_sheet', {})
    
    text_to_embed = f"""
    상황(페르소나): {criteria.get('status', '')}
    가장 큰 고민: {criteria.get('pain_point', '')}
    수강 강좌: {criteria.get('solution_course', '')}
    달성 결과: {criteria.get('outcome', '')}
    기간: {facts.get('duration', '')}
    점수 변화: {facts.get('scores', '')}
    태그: {', '.join(display.get('tags', []))}
    """
    
    try:
        # 2. 임베딩 생성
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text_to_embed,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            )
        )
        vector = response.embeddings[0].values
        
    except Exception as e:
        print(f"⚠️ 임베딩 생성 실패 ({data['meta_data'].get('doc_id')}): {e}")
        return None
    
    # -----------------------------------------------------------
    # 3. 메타데이터 구성 (여기가 중요!)
    # -----------------------------------------------------------
    metadata = {
        "category": "수강후기",
        
        # ★★★ 핵심: 원본 링크 저장 ★★★
        # 나중에 챗봇이 "자세한 건 여기서 보세요" 하고 링크를 줄 수 있음
        "url": data['meta_data'].get('source_url', ''), 
        
        # 필터링용 데이터
        "status": criteria.get('status', ''),
        
        # UI 표시용 데이터 (JSON 문자열로 저장)
        "display_json": json.dumps(display, ensure_ascii=False),
        
        # 답변 생성 시 참고할 팩트
        "fact_json": json.dumps(facts, ensure_ascii=False)
    }
    
    return {
        "id": data['meta_data']['doc_id'],
        "values": vector,
        "metadata": metadata,
        "document": text_to_embed 
    }

def main():
    # 1. 데이터 로드
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            structured_data = json.load(f)
        print(f"📂 Loaded {len(structured_data)} reviews from {INPUT_FILE}")
    except FileNotFoundError:
        print(f"❌ File not found: {INPUT_FILE}")
        print("Please run '01_preprocess_reviews.py' first.")
        return

    # 2. 벡터화
    final_db_data = []
    
    print("🚀 Starting embedding process for Reviews...")
    
    for idx, item in enumerate(structured_data):
        payload = create_embedding_payload(item)
        
        if payload:
            final_db_data.append(payload)
            print(f"   [{idx+1}/{len(structured_data)}] Vectorized: {payload['id']}")
        
        # 임베딩 API는 속도 제한이 널널해서 0.5초면 충분합니다.
        time.sleep(0.5)

    # 3. 저장
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_db_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Successfully saved {len(final_db_data)} vectors to:")
    print(f"👉 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()