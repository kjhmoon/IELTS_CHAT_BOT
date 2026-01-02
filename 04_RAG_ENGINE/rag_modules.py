import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
import chromadb
from google import genai
from google.genai import types
from kiwipiepy import Kiwi
from langchain.schema import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
load_dotenv(os.path.join(project_root, '.env'))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)


CHROMA_DB_PATH = os.path.join(project_root, 'chroma_db')

MODEL_NAME = "gemini-2.0-flash"




class ChatMemory:
    def __init__(self):
        self.history = []  
        self.user_profile = {
            "current_score": None, 
            "target_score": None,  
            "target_period": None, 
            "preferred_time": None 
        }

    def add_turn(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > 10:
            self.history = self.history[-10:]

    def update_profile(self, new_slots: Dict):
        for k, v in new_slots.items():
            if v is not None and v != "":
                self.user_profile[k] = v

    def get_context_string(self) -> str:
        context = "--- [Conversation History] ---\n"
        for msg in self.history:
            context += f"{msg['role']}: {msg['content']}\n"
        
        context += "\n--- [User Profile (Known Info)] ---\n"
        for k, v in self.user_profile.items():
            val = v if v else "(Unknown)"
            context += f"- {k}: {val}\n"
        return context




class HybridRetriever:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.kiwi = Kiwi()
        self.embedding_model = 'models/text-embedding-004'

    def search(self, collection_name: str, query: str, top_k: int = 10) -> str:
        try:
            collection = self.chroma_client.get_collection(collection_name)
            
            resp = client.models.embed_content(
                model=self.embedding_model,
                contents=query,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            query_embedding = resp.embeddings[0].values

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

            formatted_results = ""
            if not results['documents'] or not results['documents'][0]:
                return "검색 결과가 없습니다."

            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                formatted_results += f"[Result {i+1}]\nContent: {doc}\nSource: {meta.get('source', 'unknown')}\n\n"
            
            return formatted_results

        except Exception as e:
            return f"검색 중 오류 발생: {str(e)}"




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
        self.model_name = MODEL_NAME

    def analyze(self, user_input: str, context: str) -> Dict:
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




CONSULTANT_SYSTEM_PROMPT = """
당신은 이알피어학원의 **'AI 입학 상담원 아이린(Irene)'**입니다.
사용자에게 인간 상담원인 척 연기하지 말고, 도움이 필요한 부분을 빠르고 정확하게 찾아주는 **AI 어시스턴트로서의 정체성**을 유지하세요.

[핵심 행동 지침]
1. **AI 정체성 명시 및 톤앤매너:**
   - **이모티콘, 이모지(😊, 👍) 절대 사용 금지.**
   - 과도한 감정 이입이나 사람인 척하는 연기를 하지 마십시오.
   - "하십시오", "합니다" 체의 정중하고 건조한 비즈니스 화법을 구사하세요.
   - 답변 서두나 말미에 필요시 "저는 AI 상담원으로서..."와 같은 표현을 사용하여 기계적인 신뢰감을 주십시오.

2. **현장 강의(Offline) 최우선 추천 전략:**
   - 학원의 핵심 상품은 **강남/종로 캠퍼스 현장 강의**입니다.
   - [검색 결과]에 현장 강의가 있다면, 관리 시스템과 학습 분위기를 강조하여 우선적으로 추천하십시오.
   - **온라인 강의(VOD)는 언급을 최소화하십시오.** 지방 거주, 해외 거주 등 물리적으로 불가능한 상황임이 명확할 때만 최후의 수단으로 제안하십시오.

3. **[중요] 대화 주제 제한 (Guardrails):**
   - 당신의 전문 분야는 오직 **'IELTS 시험'**과 **'학원 수강'**입니다.
   - 맛집, 연애, 주식, 여행 등 학원과 무관한 사적인 질문(Chit-chat)이 들어오면 **정보 제공을 정중히 거절**하십시오.
   - **[고급 스킬]** 거절에서 끝내지 말고, 해당 주제를 **'IELTS 스피킹 기출 주제'**로 연결하여 상담으로 복귀시키십시오.
     - (예시: 맛집 질문 -> "저는 AI라 맛집은 알지 못합니다만, IELTS 스피킹 Part 1에서 'Favorite Food'는 빈출 주제입니다. 관련 표현을 알려드릴까요?")

4. **Action 유도 (Call to Action):**
   - 모든 상담의 결론은 사용자의 현재 실력을 파악하기 위한 **"무료 레벨테스트"** 권유로 이어져야 합니다.
   - 기계적인 반복 대신, 사용자의 목표 점수 달성을 위한 '필수 절차'임을 논리적으로 설명하십시오.

[참고: 검색 결과(Context)]
아래 제공된 정보를 기반으로 사실에 입각하여 답변하십시오.
"""

class ConsultantAgent:
    def __init__(self):
        self.memory = ChatMemory()
        self.router = SemanticRouter()
        self.retriever = HybridRetriever()
        
        
        self.llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=0
        )

    def run(self, user_input: str) -> str:
        self.memory.add_turn("user", user_input)
        context = self.memory.get_context_string()

        
        analysis = self.router.analyze(user_input, context)
        intent = analysis.get("intent")
        slots = analysis.get("slots_to_update", {})
        missing = analysis.get("missing_slots", [])
        search_query = analysis.get("search_query")

        print(f"🧐 [Analysis] Intent: {intent} | Missing: {missing}")

        self.memory.update_profile(slots)
        final_response = ""

        
        
        
        
        if intent == "CHIT_CHAT":
            steering_prompt = f"""
            [상황]
            사용자가 '{user_input}'라고 말했습니다. 의도는 CHIT_CHAT(잡담/인사/공격)입니다.

            [당신의 임무]
            1. **단순 인사(안녕):** "안녕하세요, 이알피어학원 AI 상담원 아이린입니다. 무엇을 도와드릴까요?"라고 짧게 응대.
            2. **그 외 모든 잡담 및 공격:** - 변명이나 부연 설명 없이, 딱 한 문장으로 답변을 거절하십시오.
               - 답변 예시: "죄송합니다. 저는 아이엘츠 상담 전용 AI이므로 학원 업무와 무관한 내용에는 답변드릴 수 없습니다."
            
            [제약 사항]
            - 이모티콘 사용 금지.
            - 스피킹 주제로 연결 금지 (절대 하지 말 것).
            - 보안 관련 질문은 "권한이 없습니다"라고 일축할 것.
            """
    
            response = self.llm.invoke([
                SystemMessage(content=CONSULTANT_SYSTEM_PROMPT),
                HumanMessage(content=steering_prompt)
            ])
            
            final_response = response.content

        
        elif intent == "TIMETABLE" and (not self.memory.user_profile.get("preferred_time") or not self.memory.user_profile.get("current_score")):
             print(f"🛑 필수 정보 누락! 되묻기 실행")
             final_response = self._generate_ask_more(missing)
        
        
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
                fallback_query = "아이엘츠 온라인 강의 인강 추천"
                search_results = self.retriever.search("timetable", fallback_query)
                search_results = f"[알림: 원하시는 조건의 강의가 없어 온라인 강의 정보를 가져왔습니다.]\n{search_results}"

            final_response = self._generate_final_answer(user_input, search_results)

        
        self.memory.add_turn("assistant", final_response)
        return final_response

    def _profile_to_string(self):
        p = self.memory.user_profile
        text = ""
        if p['preferred_time']: text += f"{p['preferred_time']} "
        if p['target_score']: text += f"목표{p['target_score']} "
        return text

    def _generate_ask_more(self, missing_slots):
        prompt = f"""
        사용자가 아이엘츠 수업을 찾고 있는데, 다음 정보가 부족합니다: {missing_slots}.
        AI 상담원으로서, 정확한 추천을 위해 이 정보들을 자연스럽게 물어보는 문장을 작성하세요.
        (예: "목표 점수가 어떻게 되시나요?", "수업 가능한 시간대가 있으신가요?")
        """
        resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        return resp.text

    def _generate_final_answer(self, user_input, search_results):
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