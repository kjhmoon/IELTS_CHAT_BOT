import json
import time
import os
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

# ------------------------------------------------------------------
# [최신 SDK] 클라이언트 초기화
# ------------------------------------------------------------------
client = genai.Client(api_key=api_key)

# ------------------------------------------------------------------
# [데이터 로드]
# ------------------------------------------------------------------
input_file_path = os.path.join(current_dir, 'raw_faq.json')

try:
    with open(input_file_path, 'r', encoding='utf-8') as f:
        raw_faqs = json.load(f)
    print(f" '{input_file_path}' 로드 성공! 총 {len(raw_faqs)}개의 데이터를 찾았습니다.")
except FileNotFoundError:
    print(f" 오류: '{input_file_path}' 파일을 찾을 수 없습니다.")
    raw_faqs = []

# ------------------------------------------------------------------
# [프롬프트 템플릿]
# ------------------------------------------------------------------
PROMPT_TEMPLATE = """
당신은 IELTS 학원의 전문 상담 데이터를 관리하는 AI입니다.
아래 제공되는 [Raw Data]를 분석하여, 지정된 [Target JSON Schema] 형식으로 완벽하게 변환하세요.

[Raw Data]
카테고리: {category}
질문: {subject}
답변내용: {contents}

[Target JSON Schema]
{{
  "meta_data": {{
    "doc_id": "faq_{{category}}_001 (카테고리 영문변환+난수)",
    "category": "{category}",
    "source_url": null,
    "last_updated": "2025-12-30"
  }},
  "search_criteria": {{
    "intent": "질문의 핵심 의도를 1문장으로 요약",
    "target_user": "이 정보가 필요한 대상 (예: 중급반 수료생)",
    "keywords": ["검색용 키워드1", "키워드2", "키워드3"]
  }},
  "display_info": {{
    "link_text": "클릭을 유도하는 매력적인 짧은 제목 (카피라이팅)",
    "tags": ["#태그1", "#태그2"]
  }},
  "faq_details": {{
    "question_summary": "질문 내용을 깔끔하게 요약",
    "answer_summary": "답변 핵심을 1-2문장으로 요약",
    "structured_points": ["단계별/항목별 핵심 내용을 리스트로 분리"],
    "related_action": "행동 유도 문구 (예: 신청하기)"
  }}
}}
"""

# ------------------------------------------------------------------
# [핵심 로직] 변환 함수
# ------------------------------------------------------------------
def transform_raw_to_structured(raw_item):
    prompt = PROMPT_TEMPLATE.format(
        category=raw_item.get('category', '기타'),
        subject=raw_item.get('subject', '제목없음'),
        contents=raw_item.get('contents', '내용없음')
    )
    
    try:
        # 모델: gemini-2.0-flash-exp (속도 제한 10 RPM)
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json'
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f" 변환 실패 (제목: {raw_item.get('subject')}): {e}")
        return None

# ------------------------------------------------------------------
# [실행부]
# ------------------------------------------------------------------
if __name__ == "__main__":
    if not raw_faqs:
        print("처리할 데이터가 없습니다. 종료합니다.")
    else:
        structured_faqs = []
        total_count = len(raw_faqs)

        print(f"🔄 데이터 변환 시작... (안전 모드: 10초 간격)")
        
        for idx, item in enumerate(raw_faqs):
            start_time = time.time() # 시작 시간 기록
            
            result = transform_raw_to_structured(item)
            if result:
                structured_faqs.append(result)
                print(f"   [{idx+1}/{total_count}] 성공: {item.get('subject')[:15]}...")
            
            # -----------------------------------------------------------
            # [속도 조절 구간]
            # -----------------------------------------------------------
            # gemini-2.0-flash-exp 제한: 분당 10회 (6초에 1회)
            # 안전하게 10초 대기 설정 (확실히 에러 안 남)
            wait_time = 10 
            
            print(f"      ㄴ ⏳ 다음 요청 대기 중... ({wait_time}초)")
            time.sleep(wait_time) 

        # 파일 저장
        output_path = os.path.join(current_dir, 'structured_faq.json')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structured_faqs, f, indent=2, ensure_ascii=False)

        print(f"\n 전체 변환 완료! 총 {len(structured_faqs)}건 저장됨.")
        print(f"파일 위치: {output_path}")