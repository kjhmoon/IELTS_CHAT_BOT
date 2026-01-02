import json
import time
import os
import re
import pandas as pd
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




def clean_sensitive_patterns(text):
    if not isinstance(text, str):
        return ""
    
    
    phone_pattern = r'01[016789]-?\d{3,4}-?\d{4}'
    text = re.sub(phone_pattern, "(전화번호 삭제됨)", text)
    
    
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    text = re.sub(email_pattern, "(이메일 삭제됨)", text)
    
    
    ssn_pattern = r'\d{6}-[1-4]\d{6}'
    text = re.sub(ssn_pattern, "(주민번호 삭제됨)", text)
    
    return text




PROMPT_TEMPLATE = """
당신은 IELTS 학원의 '수강후기 데이터'를 정제하는 AI 전문가입니다.
제공된 [Raw Review]는 '수강생의 원본 후기'와 그 밑에 달린 '학원 직원/선생님의 답글'이 섞여 있을 수 있습니다.

아래 지침에 따라 데이터를 완벽하게 분리하고 구조화하세요.

[핵심 지침]
1. **화자 구분(중요):**
   - 본문 내용 중 "안녕하세요 회원님", "축하드립니다" 등으로 시작하거나, 날짜와 함께 달린 코멘트(예: 부원장 (2016...))는 **학원 측의 답글**입니다.
   - 분석할 때는 **오직 '수강생이 쓴 본문' 내용만 사용**하세요. 직원의 칭찬 멘트를 수강생의 후기로 착각하면 안 됩니다.

2. **개인정보 비식별화:**
   - **수강생 이름:** 본문이나 작성자란에 있는 이름은 "수강생", "직장인", "학생" 등으로 변경하세요.
   - **강사 이름:** "미쉘 선생님", "김화경 강사님" 등 **수업을 가르친 강사의 이름은 유지**하세요. (마케팅 포인트임)
   - **직원 이름:** 답글에 있는 직원 이름(예: 조세영, 부원장)은 분석에서 제외하거나 삭제하세요.

3. **팩트 추출:**
   - 수강생이 언급한 점수 변화(예: 5.5 -> 7.0)와 수강 기간만 추출하세요. (직원이 "7.0 받으셨네요"라고 재언급한 내용이 아니라, 학생이 쓴 내용 기준)

[Raw Review]
제목: {title}
내용: {content}

[Target JSON Schema]
{{
  "meta_data": {{
    "doc_id": "review_{{난수ID}}",
    "source_url": "{source_url}", 
    "privacy_check": "passed" 
  }},
  "search_criteria": {{
    "status": "수강생의 상황 (예: 직장인, N수생)",
    "pain_point": "학생이 겪었던 어려움 (예: 라이팅 과락)",
    "solution_course": "수강한 강의명",
    "outcome": "최종 결과"
  }},
  "display_info": {{
    "link_text": "클릭을 유도하는 매력적인 요약 문구",
    "tags": ["#태그1", "#태그2"]
  }},
  "fact_sheet": {{
    "duration": "수강 기간",
    "scores": "점수 변화"
  }}
}}
"""

def process_review_item(row):
    
    title_raw = row.get('Title', '')
    content_raw = row.get('Content', '')
    link_raw = row.get('Link', '') 

    
    title_clean = clean_sensitive_patterns(str(title_raw))
    content_clean = clean_sensitive_patterns(str(content_raw))
    
    
    if len(content_clean) < 10:
        return None

    
    prompt = PROMPT_TEMPLATE.format(
        title=title_clean,
        content=content_clean,
        source_url=str(link_raw) if pd.notna(link_raw) else ""
    )
    
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json'
            )
        )
        
        
        parsed_data = json.loads(response.text)
        
        
        
        
        if isinstance(parsed_data, list):
            if len(parsed_data) > 0:
                parsed_data = parsed_data[0] 
            else:
                return None 
        
        return parsed_data
        
    except Exception as e:
        print(f"❌ 변환 API 에러: {e}")
        return None




if __name__ == "__main__":
    input_file = os.path.join(current_dir, 'raw_reviews.xlsx')
    output_file = os.path.join(current_dir, 'structured_reviews.json')
    
    try:
        
        df = pd.read_excel(input_file, engine='openpyxl')
        print(f"📂 엑셀 로드 성공! 총 {len(df)}건")
        
        
        expected_cols = ['Title', 'MetaInfo', 'Content', 'Link']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️ 경고: 다음 컬럼을 찾을 수 없습니다 -> {missing_cols}")
            print(f"   현재 엑셀 컬럼: {df.columns.tolist()}")

    except Exception as e:
        print(f"❌ 엑셀 로드 실패: {e}")
        exit()

    structured_data = []
    
    target_df = df
    
    print(f"🔄 데이터 처리 시작 (상위 {len(target_df)}건)...")
    
    for idx, row in target_df.iterrows():
        print(f"\n--- [{idx+1}/{len(target_df)}] 처리 중 ---")
        
        result = process_review_item(row)
        
        if result:
            structured_data.append(result)
            print(f"✅ 변환 성공 (ID: {result['meta_data']['doc_id']})")
        else:
            print(f"🚫 스킵됨 (내용 부족 등)")
        
        
        time.sleep(10)

    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ 작업 완료! 결과 파일: {output_file}")