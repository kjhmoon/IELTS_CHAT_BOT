import os
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from kiwipiepy import Kiwi
import chromadb
from google import genai
from google.genai import types




current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

load_dotenv(os.path.join(project_root, '.env'))
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("API Key not found in .env")

client = genai.Client(api_key=api_key)
kiwi = Kiwi()

CHROMA_DB_PATH = os.path.join(project_root, 'chroma_db')
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
EMBEDDING_MODEL = 'models/text-embedding-004'




def recursive_flatten(data):
    if isinstance(data, list):
        if len(data) == 0:
            return []
        if isinstance(data[0], list):
            flat = []
            for item in data:
                flat.extend(recursive_flatten(item))
            return flat
        return data
    return []




def clean_metadata(meta: Dict) -> Dict:
    """
    ChromaDB는 metadata 값으로 None(Null)을 허용하지 않습니다.
    모든 None 값을 빈 문자열 ""로 변환하고, 
    리스트나 딕셔너리가 있다면 문자열로 변환합니다.
    """
    cleaned = {}
    for k, v in meta.items():
        if v is None:
            cleaned[k] = ""  
        elif isinstance(v, (list, dict)):
            cleaned[k] = json.dumps(v, ensure_ascii=False) 
        else:
            cleaned[k] = v
    return cleaned




def generate_bm25_tokens(tags: List[str], content_text: str) -> str:
    tokens = []
    
    
    if tags:
        if isinstance(tags, str):
            tags = [tags]
        clean_tags = [str(t).replace('#', '').strip() for t in tags if t]
        tokens.extend(clean_tags) 
        tokens.extend(clean_tags) 
    
    
    if content_text:
        try:
            result = kiwi.analyze(str(content_text))
            for token, pos, _, _ in result[0][0]:
                if pos.startswith('N') and len(token) > 1:
                    tokens.append(token)
        except:
            pass 

    return " ".join(tokens)




def process_and_insert(collection_name: str, structured_path: str, ready_path: str, type_config: Dict):
    
    
    try:
        chroma_client.delete_collection(collection_name)
        print(f"🗑️  기존 컬렉션 삭제 완료: {collection_name}")
    except:
        pass
    
    collection = chroma_client.create_collection(name=collection_name)
    
    
    
    
    if os.path.exists(ready_path):
        print(f"♻️  [재활용 모드] 벡터 파일 발견! ({os.path.basename(ready_path)}) - API 호출을 생략합니다.")
        with open(ready_path, 'r', encoding='utf-8') as f:
            ready_data = json.load(f)
            ready_data = recursive_flatten(ready_data)
        
        print(f"✨ 적재 대상: {len(ready_data)}건")
        
        BATCH_SIZE = 100 
        total_processed = 0

        for i in range(0, len(ready_data), BATCH_SIZE):
            batch = ready_data[i : i + BATCH_SIZE]
            
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for idx_in_batch, item in enumerate(batch):
                try:
                    if not isinstance(item, dict): continue

                    
                    raw_id = item.get('id', 'unknown')
                    unique_doc_id = f"{raw_id}_{total_processed + idx_in_batch}"

                    vector = item.get('values')
                    meta = item.get('metadata', {})
                    text_content = item.get('document', '')
                    
                    
                    tags = []
                    if 'display_json' in meta:
                        try:
                            display_obj = json.loads(meta['display_json'])
                            tags = display_obj.get('tags', [])
                        except:
                            pass
                    
                    bm25_text = generate_bm25_tokens(tags, text_content)
                    meta['bm25_tokens'] = bm25_text 
                    
                    
                    clean_meta = clean_metadata(meta)

                    ids.append(unique_doc_id)
                    embeddings.append(vector)
                    documents.append(text_content)
                    metadatas.append(clean_meta) 
                    
                except Exception as e:
                    print(f"⚠️ 데이터 처리 에러: {e}")
                    continue
            
            if ids:
                try:
                    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
                    total_processed += len(ids)
                    print(f"   Saved {total_processed} items (No API calls)")
                except Exception as e:
                    print(f"❌ DB 저장 실패: {e}")
                
        return 

    
    
    
    print(f"🆕 [신규 생성 모드] 벡터 파일이 없습니다. {os.path.basename(structured_path)}에서 임베딩을 생성합니다.")
    
    if not os.path.exists(structured_path):
        print(f"❌ 파일 없음: {structured_path}")
        return

    with open(structured_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        data_list = recursive_flatten(raw_data)

    print(f"✨ 적재 대상: {len(data_list)}건")
    BATCH_SIZE = 50
    total_processed = 0

    for i in range(0, len(data_list), BATCH_SIZE):
        batch = data_list[i : i + BATCH_SIZE]
        ids, documents, metadatas, texts_for_embedding = [], [], [], []

        for idx_in_batch, item in enumerate(batch):
            try:
                if not isinstance(item, dict): continue
                
                meta = item.get('meta_data', {})
                raw_id = meta.get('doc_id', 'unknown')
                unique_doc_id = f"{raw_id}_{total_processed + idx_in_batch}"
                
                if type_config['type'] == 'faq':
                    tags = item.get('search_criteria', {}).get('keywords', [])
                    details = item.get('faq_details', {})
                    content_for_noun = f"{details.get('question_summary', '')} {details.get('answer_summary', '')}"
                    text_to_embed = f"Q: {details.get('question_summary', '')}\nA: {details.get('answer_summary', '')}"
                    metadata_payload = {"category": str(meta.get('category', 'General')), "source": "faq"}

                elif type_config['type'] == 'review':
                    tags = item.get('display_info', {}).get('tags', [])
                    criteria = item.get('search_criteria', {})
                    content_for_noun = f"{criteria.get('pain_point', '')} {criteria.get('outcome', '')}"
                    text_to_embed = f"상황: {criteria.get('status')}, 고민: {criteria.get('pain_point')}, 해결: {criteria.get('outcome')}"
                    metadata_payload = {"status": str(criteria.get('status', '')), "score": str(item.get('fact_sheet', {}).get('scores', '')), "source_url": str(meta.get('source_url', '')), "source": "review", "display_json": json.dumps(item.get('display_info', {}), ensure_ascii=False)}

                elif type_config['type'] == 'timetable':
                    tags = item.get('search_keywords', [])
                    display = item.get('display_info', {})
                    content_for_noun = f"{display.get('title_main', '')} {display.get('title_sub', '')}"
                    text_to_embed = f"{display.get('title_main')} - {display.get('title_sub')}"
                    metadata_payload = {"day": "Unknown", "level": "Unknown", "full_json": json.dumps(item, ensure_ascii=False), "source": "timetable"}

                bm25_text = generate_bm25_tokens(tags, content_for_noun)
                metadata_payload['bm25_tokens'] = bm25_text

                
                clean_meta = clean_metadata(metadata_payload)

                ids.append(unique_doc_id)
                documents.append(text_to_embed)
                metadatas.append(clean_meta)
                texts_for_embedding.append(text_to_embed)
            except: continue

        if texts_for_embedding:
            try:
                response = client.models.embed_content(
                    model=EMBEDDING_MODEL, contents=texts_for_embedding,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                embeddings = [e.values for e in response.embeddings]
                collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
                total_processed += len(ids)
                print(f"   Generated & Saved {total_processed} items...")
            except Exception as e:
                print(f"❌ API 에러: {e}")
                time.sleep(10)




def main():
    print("🚀 RAG Vector DB 구축 시작 (Null Cleaning Applied)")

    
    process_and_insert(
        'faq', 
        os.path.join(project_root, '01_FAQ', 'structured_faq.json'),
        os.path.join(project_root, '01_FAQ', 'faq_db_ready.json'), 
        {'type': 'faq'}
    )

    
    process_and_insert(
        'review', 
        os.path.join(project_root, '02_REVIEW', 'structured_reviews.json'),
        os.path.join(project_root, '02_REVIEW', 'review_db_ready.json'),
        {'type': 'review'}
    )

    
    process_and_insert(
        'timetable', 
        os.path.join(project_root, '03_TIMETABLE', 'structured_timetable.json'),
        os.path.join(project_root, '03_TIMETABLE', 'timetable_db_ready.json'),
        {'type': 'timetable'}
    )

    print("\n🎉 모든 데이터 적재 완료! (./chroma_db)")

if __name__ == "__main__":
    main()