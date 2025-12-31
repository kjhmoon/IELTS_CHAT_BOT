import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ------------------------------------------------------------------
# [경로 설정] .env 및 파일 경로 자동 인식
# ------------------------------------------------------------------
# 1. 현재 파일의 폴더 경로 (03_TIMETABLE)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 프로젝트 루트 폴더
parent_dir = os.path.dirname(current_dir)

# 3. .env 파일 경로 조합
env_path = os.path.join(parent_dir, '.env')

# 4. 환경 변수 로드
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(f"API Key not found. Checked path: {env_path}")

# ------------------------------------------------------------------
# [최신 SDK] 클라이언트 초기화
# ------------------------------------------------------------------
client = genai.Client(api_key=api_key)

EMBEDDING_MODEL = 'models/text-embedding-004'

# 입출력 파일 (절대 경로)
INPUT_FILE = os.path.join(current_dir, 'structured_timetable.json')
OUTPUT_FILE = os.path.join(current_dir, 'timetable_db_ready.json')

def create_embedding_payload(course_data):
    """
    구조화된 시간표 데이터를 받아 ChromaDB용 [ID, Vector, Metadata]로 변환합니다.
    """
    data = course_data
    
    # -----------------------------------------------------------
    # [방어 코드] 리스트로 감싸져 있는 경우 벗겨내기
    # -----------------------------------------------------------
    if isinstance(data, list):
        if len(data) > 0:
            data = data[0]
        else:
            return None
            
    if not isinstance(data, dict):
        return None
    # -----------------------------------------------------------

    # 1. 임베딩할 텍스트 생성 (Serialize)
    # ★ 핵심: 지점, 강좌명, 특징, 그리고 '검색 키워드'를 모두 포함해야 함
    
    # 안전하게 필드 가져오기
    meta = data.get('meta_data', {})
    display = data.get('display_info', {})
    spec = data.get('course_spec', {})
    schedule = spec.get('schedule', {})
    keywords = data.get('search_keywords', [])

    # 요일/시간 정보 텍스트화
    days = ", ".join(schedule.get('days_exact') or [])
    time_str = ""
    if schedule.get('time_exact'):
        time_str = f"{schedule['time_exact'].get('start')}~{schedule['time_exact'].get('end')}"
    
    duration_str = schedule.get('duration_text') or ""

    # 임베딩용 텍스트 조립
    text_to_embed = f"""
    지점: {meta.get('branch', '')}
    강좌명: {display.get('title_main', '')}
    특징: {display.get('title_sub', '')}
    키워드: {', '.join(keywords)}
    요일 및 시간: {days} {time_str} {duration_str}
    """
    
    try:
        # 2. 임베딩 생성 (Gemini 최신 SDK)
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text_to_embed,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            )
        )
        vector = response.embeddings[0].values
        
    except Exception as e:
        print(f"⚠️ Error generating embedding for {meta.get('doc_id')}: {e}")
        return None
    
    # 3. 메타데이터 구성 (필터링 및 계산용)
    # ChromaDB는 List/Dict 저장이 안 되므로 json.dumps로 문자열 변환 필수
    metadata = {
        "branch": meta.get('branch', ''),          # 필터링용 (강남/종로)
        "course_type": meta.get('course_type', ''), # 필터링용 (online/offline)
        "display_json": json.dumps(display, ensure_ascii=False), # UI 카드용
        "price_json": json.dumps(spec.get('price_options', []), ensure_ascii=False), # 가격 계산용
        "keywords_str": ", ".join(keywords) # BM25 검색 보조용
    }
    
    return {
        "id": meta.get('doc_id', f"unknown_{int(time.time())}"),
        "values": vector,
        "metadata": metadata,
        "document": text_to_embed # 원본 텍스트 저장 (키워드 검색용)
    }

def main():
    # 1. 데이터 로드
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            structured_data = json.load(f)
        print(f"📂 Loaded {len(structured_data)} courses from {INPUT_FILE}")
    except FileNotFoundError:
        print(f"❌ File not found: {INPUT_FILE}")
        print("Please run '01_preprocess_timetable.py' first.")
        return

    # 2. 전체 데이터 임베딩 처리
    final_db_data = []
    
    print("🚀 Starting embedding process for Timetable...")
    
    for idx, item in enumerate(structured_data):
        payload = create_embedding_payload(item)
        
        if payload:
            final_db_data.append(payload)
            print(f"   [{idx+1}/{len(structured_data)}] Vectorized: {payload['id']}")
        else:
            print(f"   [{idx+1}/{len(structured_data)}] Skipped (Invalid Data)")
        
        # API 속도 제한 고려 (0.5초 대기)
        time.sleep(0.5)

    # 3. 결과 저장
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_db_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Successfully saved {len(final_db_data)} vectors to:")
    print(f"👉 {OUTPUT_FILE}")

    # 샘플 확인
    if final_db_data:
        print("\n--- Sample Metadata Check ---")
        print(f"Keys: {list(final_db_data[0]['metadata'].keys())}")
        print(f"Price Info (Serialized): {final_db_data[0]['metadata']['price_json'][:50]}...")

if __name__ == "__main__":
    main()