import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ------------------------------------------------------------------
# [경로 설정] .env 및 파일 경로 자동 인식
# ------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
env_path = os.path.join(parent_dir, '.env')

load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(f"API Key not found. Checked path: {env_path}")

# ------------------------------------------------------------------
# [최신 SDK] 클라이언트 초기화
# ------------------------------------------------------------------
client = genai.Client(api_key=api_key)

EMBEDDING_MODEL = 'models/text-embedding-004'

INPUT_FILE = os.path.join(current_dir, 'structured_faq.json')
OUTPUT_FILE = os.path.join(current_dir, 'faq_db_ready.json')

def create_embedding_payload(structured_json):
    """
    구조화된 JSON을 받아 ChromaDB에 넣을 [ID, Vector, Metadata] 형태로 변환합니다.
    """
    data = structured_json
    
    # -----------------------------------------------------------
    # ★★★ [FIX] 리스트로 감싸져 있는 경우 벗겨내는 로직 추가 ★★★
    # -----------------------------------------------------------
    if isinstance(data, list):
        if len(data) > 0:
            data = data[0] # 리스트의 첫 번째 요소를 진짜 데이터로 사용
        else:
            return None # 빈 리스트면 건너뜀
    
    # 데이터 유효성 검사 (필수 키가 없으면 에러 나므로 방어 코드 추가)
    if not isinstance(data, dict) or 'search_criteria' not in data:
        print(f"⚠️ Invalid data structure skipped: {type(data)}")
        return None
    # -----------------------------------------------------------

    # 1. 임베딩할 텍스트 생성 (Serialize)
    try:
        text_to_embed = f"""
        의도: {data['search_criteria'].get('intent', '')}
        대상: {data['search_criteria'].get('target_user', '')}
        키워드: {', '.join(data['search_criteria'].get('keywords', []))}
        질문: {data['faq_details'].get('question_summary', '')}
        핵심답변: {data['faq_details'].get('answer_summary', '')}
        상세내용: {' '.join(data['faq_details'].get('structured_points', []))}
        """
    except Exception as e:
        print(f"⚠️ Text serialization failed: {e}")
        return None
    
    try:
        # 2. 임베딩 생성 (Gemini 최신 SDK 사용)
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text_to_embed,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            )
        )
        vector = response.embeddings[0].values
        
    except Exception as e:
        # doc_id가 없는 경우를 대비해 안전하게 접근
        doc_id = data.get('meta_data', {}).get('doc_id', 'Unknown')
        print(f"⚠️ Error generating embedding for {doc_id}: {e}")
        return None
    
    # 3. 메타데이터 구성
    metadata = {
        "category": data['meta_data'].get('category', ''),
        "intent": data['search_criteria'].get('intent', ''),
        "display_json": json.dumps(data.get('display_info', {}), ensure_ascii=False),
        "full_context": json.dumps(data.get('faq_details', {}), ensure_ascii=False)
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
            structured_faqs = json.load(f)
        print(f"📂 Loaded {len(structured_faqs)} items from {INPUT_FILE}")
    except FileNotFoundError:
        print(f"❌ File not found: {INPUT_FILE}")
        print("Please run '01_preprocess_data.py' first.")
        return

    # 2. 전체 데이터 임베딩 처리
    final_db_data = []
    
    print("🚀 Starting embedding process...")
    for idx, item in enumerate(structured_faqs):
        payload = create_embedding_payload(item)
        
        if payload:
            final_db_data.append(payload)
            print(f"   [{idx+1}/{len(structured_faqs)}] Vectorized: {payload['id']}")
        else:
            print(f"   [{idx+1}/{len(structured_faqs)}] Skipped (Invalid Data)")
        
        # API 속도 제한 고려
        time.sleep(0.5)

    # 3. 결과 저장
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_db_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Successfully saved {len(final_db_data)} vectors to:")
    print(f"👉 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()