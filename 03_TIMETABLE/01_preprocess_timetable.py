import json
import time
import os
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




input_file_path = os.path.join(current_dir, 'raw_timetable.json')

try:
    with open(input_file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    print(f"📂 '{input_file_path}' 로드 성공! 총 {len(raw_data)}개의 강의 데이터를 찾았습니다.")
except FileNotFoundError:
    print(f"❌ 오류: '{input_file_path}' 파일을 찾을 수 없습니다.")
    raw_data = []




PROMPT_TEMPLATE = """
당신은 IELTS 학원 시간표 데이터를 구조화하는 AI 데이터 엔지니어입니다.
아래 [Raw Data]를 분석하여, [Target JSON Schema]에 맞춰 완벽하게 변환하세요.

[핵심 지침]
1. **search_keywords (중요)**: 사용자가 검색할만한 동의어, 상황 태그를 5개 이상 생성하세요.
   - 예: "토, 일" -> ["주말반", "직장인", "단기완성"]
   - 예: "19:00" -> ["저녁반", "직장인반", "퇴근후"]
   - 예: "ON" -> ["인강", "온라인", "집에서", "무제한"]
2. **price_options**: 가격 정보 텍스트를 분석하여 숫자(Integer)로 변환하고 배열에 담으세요.
3. **Branch 구분**: m_jiyuk 값이 'ON'이면 course_type은 'online', 나머지는 'offline'입니다.

[Raw Data]
지점(m_jiyuk): {m_jiyuk}
강좌명(m_name): {m_name}
요일(m_yoil): {m_yoil}
시간/기간(m_sigan): {m_sigan}
가격정보(m_priceinfo): {m_priceinfo}
기본가격(m_cashprice): {m_cashprice}

[Target JSON Schema]
{{
  "meta_data": {{
    "doc_id": "course_{{지점영문}}_{{난수}}",
    "branch": "{m_jiyuk}", 
    "course_type": "offline 또는 online (m_jiyuk이 ON이면 online)",
    "last_updated": "2025-12-30",
    "is_active": true
  }},
  
  "display_info": {{
    "title_main": "{m_name}",
    "title_sub": "강좌 특징 요약 (예: 1개월 38만 / 2개월 59만)",
    "status_badge": "모집중",
    "link_url": ""
  }},

  "search_keywords": ["키워드1", "키워드2", "주중반/주말반", "오전/오후/저녁", "타겟대상"],

  "course_spec": {{
    "schedule": {{
      "days_exact": ["월", "수", "금"] (또는 null),
      "time_exact": {{ "start": "HH:mm", "end": "HH:mm" }} (또는 null),
      "duration_text": "{m_sigan}"
    }},
    "price_options": [
      {{ "option_name": "1개월 등록", "amount": 380000 }},
      {{ "option_name": "2개월 등록 (할인)", "amount": 590000 }}
    ],
    "materials": "교재 정보 요약"
  }}
}}
"""




def transform_timetable_data(raw_item):
    
    prompt = PROMPT_TEMPLATE.format(
        m_jiyuk=raw_item.get('m_jiyuk', 'Unknown'),
        m_name=raw_item.get('m_name', '제목없음'),
        m_yoil=raw_item.get('m_yoil', ''),
        m_sigan=raw_item.get('m_sigan', ''),
        m_priceinfo=raw_item.get('m_priceinfo', ''),
        m_cashprice=raw_item.get('m_cashprice', 0)
    )
    
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json'
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"❌ 변환 실패 (강좌명: {raw_item.get('m_name')}): {e}")
        return None




if __name__ == "__main__":
    if not raw_data:
        print("처리할 데이터가 없습니다. 종료합니다.")
    else:
        structured_timetable = []
        total_count = len(raw_data)

        print(f"🔄 시간표 데이터 구조화 시작... (안전 모드: 10초 간격)")
        
        for idx, item in enumerate(raw_data):
            
            result = transform_timetable_data(item)
            
            if result:
                structured_timetable.append(result)
                print(f"   [{idx+1}/{total_count}] 성공: {item.get('m_name')[:20]}...")
            
            
            
            
            wait_time = 10
            print(f"      ㄴ ⏳ API 대기 중... ({wait_time}초)")
            time.sleep(wait_time) 

        
        output_path = os.path.join(current_dir, 'structured_timetable.json')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structured_timetable, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 전체 변환 완료! 총 {len(structured_timetable)}건 저장됨.")
        print(f"파일 위치: {output_path}")