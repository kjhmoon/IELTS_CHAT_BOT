import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types





current_dir = os.path.dirname(os.path.abspath(__file__))


parent_dir = os.path.dirname(current_dir)


env_path = os.path.join(parent_dir, '.env')


load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(f"API Key not found. Checked path: {env_path}")




client = genai.Client(api_key=api_key)

EMBEDDING_MODEL = 'models/text-embedding-004'


INPUT_FILE = os.path.join(current_dir, 'structured_timetable.json')
OUTPUT_FILE = os.path.join(current_dir, 'timetable_db_ready.json')

def create_embedding_payload(course_data):
    """
    구조화된 시간표 데이터를 받아 ChromaDB용 [ID, Vector, Metadata]로 변환합니다.
    """
    data = course_data
    
    
    
    
    if isinstance(data, list):
        if len(data) > 0:
            data = data[0]
        else:
            return None
            
    if not isinstance(data, dict):
        return None
    

    
    
    
    
    meta = data.get('meta_data', {})
    display = data.get('display_info', {})
    spec = data.get('course_spec', {})
    schedule = spec.get('schedule', {})
    keywords = data.get('search_keywords', [])

    
    days = ", ".join(schedule.get('days_exact') or [])
    time_str = ""
    if schedule.get('time_exact'):
        time_str = f"{schedule['time_exact'].get('start')}~{schedule['time_exact'].get('end')}"
    
    duration_str = schedule.get('duration_text') or ""

    
    text_to_embed = f"""
    지점: {meta.get('branch', '')}
    강좌명: {display.get('title_main', '')}
    특징: {display.get('title_sub', '')}
    키워드: {', '.join(keywords)}
    요일 및 시간: {days} {time_str} {duration_str}
    """
    
    try:
        
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
    
    
    
    metadata = {
        "branch": meta.get('branch', ''),          
        "course_type": meta.get('course_type', ''), 
        "display_json": json.dumps(display, ensure_ascii=False), 
        "price_json": json.dumps(spec.get('price_options', []), ensure_ascii=False), 
        "keywords_str": ", ".join(keywords) 
    }
    
    return {
        "id": meta.get('doc_id', f"unknown_{int(time.time())}"),
        "values": vector,
        "metadata": metadata,
        "document": text_to_embed 
    }

def main():
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            structured_data = json.load(f)
        print(f"📂 Loaded {len(structured_data)} courses from {INPUT_FILE}")
    except FileNotFoundError:
        print(f"❌ File not found: {INPUT_FILE}")
        print("Please run '01_preprocess_timetable.py' first.")
        return

    
    final_db_data = []
    
    print("🚀 Starting embedding process for Timetable...")
    
    for idx, item in enumerate(structured_data):
        payload = create_embedding_payload(item)
        
        if payload:
            final_db_data.append(payload)
            print(f"   [{idx+1}/{len(structured_data)}] Vectorized: {payload['id']}")
        else:
            print(f"   [{idx+1}/{len(structured_data)}] Skipped (Invalid Data)")
        
        
        time.sleep(0.5)

    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_db_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Successfully saved {len(final_db_data)} vectors to:")
    print(f"👉 {OUTPUT_FILE}")

    
    if final_db_data:
        print("\n--- Sample Metadata Check ---")
        print(f"Keys: {list(final_db_data[0]['metadata'].keys())}")
        print(f"Price Info (Serialized): {final_db_data[0]['metadata']['price_json'][:50]}...")

if __name__ == "__main__":
    main()