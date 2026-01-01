import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
import chromadb
from google import genai
from google.genai import types
from kiwipiepy import Kiwi

# 환경 변수 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
load_dotenv(os.path.join(project_root, '.env'))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# ChromaDB 경로 설정
CHROMA_DB_PATH = os.path.join(project_root, 'chroma_db')

# ★ 수정된 부분: 모델 이름을 변수로 관리 (안정적인 버전 사용)
MODEL_NAME = "gemini-2.0-flash"

# -----------------------------------------------------------------------------
# 1. ChatMemory: 대화 기억 및 사용자 프로필 관리
# -----------------------------------------------------------------------------
class ChatMemory:
    def __init__(self):
        self.history = []  # 대화 기록 [{"role": "user", "content": "..."}, ...]
        self.user_profile = {
            "current_score": None, # 현재 점수/실력
            "target_score": None,  # 목표 점수
            "target_period": None, # 목표 기간
            "preferred_time": None # 선호 시간대
        }

    def add_turn(self, role: str, content: str):
        """대화 턴 추가"""
        self.history.append({"role": role, "content": content})
        # 메모리 무한 증식 방지 (최근 10턴 유지)
        if len(self.history) > 10:
            self.history = self.history[-10:]

    def update_profile(self, new_slots: Dict):
        """라우터가 추출한 정보로 프로필 업데이트"""
        for k, v in new_slots.items():
            if v is not None and v != "":
                self.user_profile[k] = v

    def get_context_string(self) -> str:
        """LLM에게 던져줄 대화 요약 문자열"""
        context = "--- [Conversation History] ---\n"
        for msg in self.history:
            context += f"{msg['role']}: {msg['content']}\n"
        
        context += "\n--- [User Profile (Known Info)] ---\n"
        for k, v in self.user_profile.items():
            val = v if v else "(Unknown)"
            context += f"- {k}: {val}\n"
        return context

# -----------------------------------------------------------------------------
# 2. HybridRetriever: 하이브리드 검색기 (Vector + Filter)
# -----------------------------------------------------------------------------
class HybridRetriever:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.kiwi = Kiwi()
        self.embedding_model = 'models/text-embedding-004'

    def search(self, collection_name: str, query: str, top_k: int = 10) -> str:
        """
        ChromaDB 검색 수행
        """
        try:
            collection = self.chroma_client.get_collection(collection_name)
            
            # 1. 쿼리 임베딩
            resp = client.models.embed_content(
                model=self.embedding_model,
                contents=query,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            query_embedding = resp.embeddings[0].values

            # 2. 검색 (Vector Search)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

            # 3. 결과 포맷팅
            formatted_results = ""
            if not results['documents'] or not results['documents'][0]:
                return "검색 결과가 없습니다."

            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                formatted_results += f"[Result {i+1}]\nContent: {doc}\nSource: {meta.get('source', 'unknown')}\n\n"
            
            return formatted_results

        except Exception as e:
            return f"검색 중 오류 발생: {str(e)}"

# -----------------------------------------------------------------------------
# 3. SemanticRouter: 의도 분류 및 슬롯 필링
# -----------------------------------------------------------------------------
ROUTER_SYSTEM_PROMPT = """
You are the 'Intent Classifier' for an IELTS Academy Chatbot.
Analyze the user's input based on the conversation history and user profile.

[Role]
1. Identify User Intent:
   - TIMETABLE: Questions about class schedules, curriculum, prices.
   - REVIEW: Asking for success stories, student reviews, difficulty concerns.
   - FAQ: Administrative questions (refund, location, parking, login).
   - CHIT_CHAT: Greetings, small talk, insults, or off-topic.

2. Slot Filling (Crucial for TIMETABLE):
   - Extract info for: current_score, target_score, target_period, preferred_time.
   - If user provides info, fill 'slots_to_update'.
   - Identify 'missing_slots' ONLY IF intent is TIMETABLE.

3. Output Format (JSON Only):
{
  "intent": "TIMETABLE" | "REVIEW" | "FAQ" | "CHIT_CHAT",
  "reason": "Short explanation",
  "slots_to_update": {
      "current_score": "...",
      "target_score": "...",
      "target_period": "...",
      "preferred_time": "..."
  },
  "missing_slots": ["current_score", "target_score", ...] (List missing critical info),
  "search_query": "Refined search query for DB"
}
"""

class SemanticRouter:
    def __init__(self):
        self.model_name = MODEL_NAME # ★ 수정됨

    def analyze(self, user_input: str, context: str) -> Dict:
        """사용자 의도 및 슬롯 분석"""
        prompt = f"""
        {ROUTER_SYSTEM_PROMPT}

        [Context]
        {context}

        [User Input]
        {user_input}
        
        Generate JSON response:
        """
        
        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"Router Error: {e}")
            return {"intent": "CHIT_CHAT", "reason": "Error", "slots_to_update": {}, "missing_slots": [], "search_query": user_input}

# -----------------------------------------------------------------------------
# 4. ConsultantAgent: 총괄 매니저 (Main Brain)
# -----------------------------------------------------------------------------
CONSULTANT_SYSTEM_PROMPT = """
당신은 이알피어학원의 **'AI 입학 상담 실장 아이린'**입니다.
[대화 기억]과 [검색 결과]를 바탕으로, 전문적이고 신뢰감 있는 태도로 상담을 진행하세요.

[상담 가이드라인 - 엄격 준수]
1. **톤앤매너(No Emoji):**
   - **이모티콘이나 이모지를 절대 사용하지 마세요.** (예: 😊, 👍 금지)
   - 가볍거나 유치한 표현을 삼가고, "하십시오", "합니다" 체의 정중하고 전문적인 비즈니스 화법을 구사하세요.

2. **현장 강의(Offline) 최우선 추천:**
   - 학원의 주력 상품은 **강남/종로 캠퍼스의 현장 강의**입니다.
   - 검색 결과에 현장 강의가 있다면 그것을 가장 매력적으로 포장하여 추천하세요.
   - **온라인 강의(VOD)는 언급을 자제하세요.** 수요가 적은 사양 콘텐츠이므로, 사용자가 "지방에 산다", "도저히 시간이 안 맞다"라고 명시적으로 말하기 전까지는 먼저 권하지 마세요.

3. **맥락 연결 및 논리적 제안:**
   - 사용자의 상황(직장인, 점수 등)을 언급하며 공감대를 형성하되, 과한 감탄사는 배제하세요.
   - 예: "직장 생활과 병행하시느라 시간 내기가 어려우시겠지만, 목표 달성을 위해 최적의 반을 찾아보겠습니다."

4. **Action 유도:**
   - 상담의 마무리는 항상 **"정확한 반 배정을 위한 무료 레벨테스트"** 권유입니다.
   - 기계적으로 반복하지 말고 자연스럽게 연결하세요.
"""

class ConsultantAgent:
    def __init__(self):
        self.memory = ChatMemory()
        self.router = SemanticRouter()
        self.retriever = HybridRetriever()

    def run(self, user_input: str) -> str:
        # 1. 메모리에 사용자 질문 기록
        self.memory.add_turn("user", user_input)
        context = self.memory.get_context_string()

        # 2. 라우터 분석 (CoT: 생각 단계)
        analysis = self.router.analyze(user_input, context)
        intent = analysis.get("intent")
        slots = analysis.get("slots_to_update", {})
        missing = analysis.get("missing_slots", [])
        search_query = analysis.get("search_query")

        print(f"🧐 [Analysis] Intent: {intent} | Missing: {missing}")

        # 3. 프로필 업데이트
        self.memory.update_profile(slots)

        final_response = ""

        # 4. 시나리오 분기 (Logic Flow)
        
        # [CASE 1] 잡담 (CHIT_CHAT)
        if intent == "CHIT_CHAT":
            final_response = self._generate_chit_chat(user_input)

        # [CASE 2] 시간표 질문인데 필수 정보 부족 (Slot Filling) ★ 수정됨
        # 개수(len)로 세지 않고, 핵심 필드(Time, Score)가 비어있으면 무조건 되묻기
        elif intent == "TIMETABLE" and (not self.memory.user_profile.get("preferred_time") or not self.memory.user_profile.get("current_score")):
             # 로그 확인용 프린트
             print(f"🛑 필수 정보 누락! 되묻기 실행 (Time: {self.memory.user_profile.get('preferred_time')}, Score: {self.memory.user_profile.get('current_score')})")
             final_response = self._generate_ask_more(missing)
        
        # [CASE 3] 검색 필요 (FAQ, REVIEW, 또는 정보 충분한 TIMETABLE)
        else:
            collection_map = {
                "TIMETABLE": "timetable",
                "REVIEW": "review",
                "FAQ": "faq"
            }
            collection_name = collection_map.get(intent, "faq")
            
            enhanced_query = f"{search_query} {self._profile_to_string()}"
            search_results = self.retriever.search(collection_name, enhanced_query, top_k=10)
            
            if "검색 결과가 없습니다" in search_results:
                print("⚠️ 검색 결과 0건 -> Fallback 실행")
                fallback_query = "아이엘츠 온라인 강의 인강 추천"
                search_results = self.retriever.search("timetable", fallback_query)
                search_results = f"[알림: 원하시는 조건의 강의가 없어 온라인 강의 정보를 가져왔습니다.]\n{search_results}"

            final_response = self._generate_final_answer(user_input, search_results)

        # 5. 메모리에 봇 답변 기록
        self.memory.add_turn("assistant", final_response)
        return final_response

    def _profile_to_string(self):
        """프로필 정보를 검색어용 문자열로 변환"""
        p = self.memory.user_profile
        text = ""
        if p['preferred_time']: text += f"{p['preferred_time']} "
        if p['target_score']: text += f"목표{p['target_score']} "
        return text

    def _generate_chit_chat(self, user_input):
        """가벼운 대화 생성"""
        prompt = f"당신은 친절한 아이엘츠 AI 상담원입니다. 다음 말에 자연스럽게 대답하세요: {user_input}"
        # ★ 수정됨: 변수 사용
        resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        return resp.text

    def _generate_ask_more(self, missing_slots):
        """부족한 정보 되묻기"""
        prompt = f"""
        사용자가 아이엘츠 수업을 찾고 있는데, 다음 정보가 부족합니다: {missing_slots}.
        AI 상담원으로서, 정확한 추천을 위해 이 정보들을 자연스럽게 물어보는 문장을 작성하세요.
        (예: "목표 점수가 어떻게 되시나요?", "수업 가능한 시간대가 있으신가요?")
        """
        # ★ 수정됨: 변수 사용
        resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        return resp.text

    def _generate_final_answer(self, user_input, search_results):
        """RAG 최종 답변 생성 (전문성 강화 & VOD 억제 버전)"""
        
        # 사용자의 제약 조건을 강조하는 텍스트 생성
        constraints = ""
        p = self.memory.user_profile
        if p.get('preferred_time') == 'Weekend':
            constraints += "- 사용자 제약: 주말 선호 (평일 불가능 가능성 높음)\n"
        if "직장인" in self.memory.get_context_string():
            constraints += "- 사용자 제약: 직장인 (효율적인 커리큘럼 선호)\n"

        prompt = f"""
        {CONSULTANT_SYSTEM_PROMPT}

        [User Profile & Constraints]
        {self.memory.get_context_string()}
        {constraints}

        [User Question]
        {user_input}

        [Search Results (Database)]
        {search_results}

        위 정보를 종합하여 답변을 작성하세요.
        주의: [Search Results]에 온라인 강의가 포함되어 있더라도, 현장 강의(강남/종로)가 있다면 현장 강의 위주로만 설명하세요.
        온라인 강의는 사용자가 도저히 통학할 수 없는 상황일 때만 '참고용'으로 짧게 언급하십시오.
        """
        
        resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        return resp.text

if __name__ == "__main__":
    agent = ConsultantAgent()
    print(" 아이린 상담원과 연결되었습니다. (종료: q)")
    
    while True:
        try:
            user_text = input("\nUser: ")
            if user_text.lower() == 'q':
                break
            
            response = agent.run(user_text)
            print(f"Irene: {response}")
        except Exception as e:
            print(f"❌ 에러 발생: {e}")