from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage
# AnyMessage : 여러 xxMessage 타입을 하나로 통합하여 표현하는 타입.
#       다양한 Message 타입을 모두 받을수 있다.

# AnyMessage <=
    # AIMessage
    # HumanMessage
    # SystemMessage
    # ToolMessage

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from typing_extensions import TypedDict
from typing import List

from datetime import datetime
import os

from utils import *
from models import *
from tools import retrieve, web_search, add_web_pages_json_to_chroma

# 현재 폴더 경로
filename = os.path.basename(__file__)  # 현재 파일명
absolute_path = os.path.abspath(__file__)  # 현재 파일의 절대경로
current_path = os.path.dirname(absolute_path)   # 현재 .py 파일이 있는 경로

# 모델 초기화
llm = ChatOpenAI(model="gpt-4o")


# 🟦 상태 정의
class State(TypedDict):
    messages: List[AnyMessage | str]
    
    # Task 목록 을 저장.  현재 작업하는 내용을 list 로 추가해 나갈거다
    task_history: List[Task]
    references: dict   # 벡터 검색 결과를 State에 보관
    user_request: str  # 사용자의 요구사항을 저장 <- business_analyst가 사용자와 대화한 내용과 현재의 진행상황 (목차, 참고자료..) 를 바탕으로 분석하여 채울거다.
    ai_recommendation: str  # AI 의 조언

    supervisor_call_count: int  # supervisor 호출 횟수 저장 <- 무한루프 방지 목적


# 🟦 business_analyst 노드 agent
def business_analyst(state: State):
    print("\n\n🤍====== BUSINESS ANALYST ======")

    # 시스템 프롬프트
    business_analyst_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 비즈니스 애널리스트로서,
        AI팀의 진행상황과 "사용자 요구사항"을 토대로,
        현 시점에서 'ai_recommendation'과 최근 사용자의 발언을 바탕으로 요구사항이 무엇인지 판단한다.
        지난 요청사항이 달성되었는지 판단하고, 현 시점에서 어떤 작업을 해야 하는지 결정한다.

        다음과 같은 템플릿 형태로 반환한다.
        ```
        - 목표: OOOO \n 방법: OOOO
        ```

        ------------------------------------
        *AI 추천(ai_recommendation)* : {ai_recommendation}
        ------------------------------------
        사용자 최근 발언: {user_last_comment}
        ------------------------------------
        참고자료: {references}
        ------------------------------------
        목차 (outline): {outline}
        ------------------------------------
        "messages": {messages}
        """
    )

    # 체인
    ba_chain = business_analyst_system_prompt | llm | StrOutputParser()

    # 체인 입력값들 준비
    messages = state['messages']

    # 위 messages 에서 사용자의 마지막 발언 가져오기
    user_last_comment = None
    for m in messages[::-1]:
        if isinstance(m, HumanMessage):
            user_last_comment = m.content
            break

    # inputs 준비
    inputs = {
        "ai_recommendation": state.get("ai_recommendation", None),
        "user_last_comment": user_last_comment,
        "references": state.get("references", {"queries": [], "docs": []}), 
        "outline": get_outline(current_path),
        "messages": messages,
    }

    # 체인호출:  '목표' 와 '방법'을 생성
    user_request = ba_chain.invoke(inputs)

    # 결과는 메세지에 추가
    business_analyst_message = f"[Business Analyst] {user_request}"
    print(business_analyst_message) # 확인
    messages.append(AIMessage(business_analyst_message))

    # 현재 state 내용 저장
    # 이전에는 사용자가 메세지 입력한 직후에 저장했지만,  앞으로는 사용자가 입력하지 않아도
    # 알아서 루프를 여러번 돌수 있으므로.. state 를 여기서 저장
    save_state(current_path, state)

    print("\n\n🤍====== BUSINESS ANALYST 종료 ======")

    return {
        "messages": messages,
        "user_request": user_request,
        "ai_recommendation": "",   # ai_recommendation 은 business_analyst 에만 필요.. 다른 부분에 실수로 영향주지 않기 위해 처리
    }



# 🟦 목차를 작성하는 노드 agent: content_strategist
def content_strategist(state: State):
    print("\n\n💛====== CONTENT STRATEGIST ======")

    task_history = state.get("task_history", [])
    task = task_history[-1]
    if task.agent != "content_strategist":
        raise ValueError(f"💥Content Strategist가 아닌 agent가 목차 작성을 시도하고 있습니다.\n {task}")


    # 지난 목차(outline)와 이전 대화 내용(messages)이 주어지면 이전 대화 내용을 바탕으로
    # 새로운 목차를 생성하라는 문구를 추가합니다.
    content_strategist_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 콘텐츠 전략가(Content Strategist)로서,
        이전 대화 내용을 바탕으로 사용자의 요구사항을 분석하고, AI팀이 쓸 책의 세부 목차를 결정한다.

        지난 목차가 있다면 그 버전을 사용자의 요구에 맞게 수정하고, 없다면 새로운 목차를 제안한다.
        목차를 작성하는데 필요한 정보는 "참고 자료" 에 있으므로 활용한다.

        다음 정보를 활용하여 목차를 작성하라.
        - 사용자 요구사항(user_request)
        - 작업(task)
        - 검색 자료 (references)
        - 기존 목차 (previous_outline)
        - 이전 대화 내용(messages)

        너의 작업 목표는 다음과 같다:
        1. 만약 "기존 목차 구조 (previous_outline)"이 존재한다면, 사용자의 요구사항을 토대로 "기존 목차 구조"에서 어떤 부분을 수정하거나 추가할지 결정한다.
        - "이번 목차 작성의 주안점"에 사용자 요구사항(user_request)을 충족시키는 것을 명시해야 한다.
        2. 책의 전반적인 구조(chapter, section)를 설계하고, 각 chpater와 section의 제목을 정한다.
        3. 책의 전반적인 세부구조(chapter, section, sub-section)를 설계하고, sub-section 하부의 주요 내용을 리스트 형태로 정리한다.
        4. 목차의 논리적인 흐름이 사용자 요구를 충족시키는지 확인한다.
        5. 참고자료 (references)를 적극 활용하여 근거에 기반한 목차를 작성한다.
        6. 참고문헌은 반드시 참고자료(references) 자료를 근거로 작성해야 하며, 최대한 풍부하게 준비한다. URL은 전체 주소를 적어야 한다.
        7. 추가 자료나 리서치가 필요한 부분을 파악하여 supervisor에게 요청한다.

        사용자 요구사항(user_request)을 최우선으로 반영하는 목차로 만들어야 한다.

        --------------------------------
        - 사용자 요구사항(user_request):
        {user_request}
        --------------------------------
        - 작업(task):
        {task}
        --------------------------------
        - 참고 자료 (references)
        {references}
        --------------------------------
        - 기존 목차 (previous_outline)
        {outline}
        --------------------------------
        - 이전 대화 내용(messages)
        {messages}
        --------------------------------

        작성 형식 아래 양식을 지키되 하부 항목으로 더 세분화해도 좋다. 목차(outline) 양식의 챕터, 섹션 등 항목의 갯수는 필요한만큼 추가하라.
        섹션 갯수는 최소 2개 이상이어야 하며, 더 많으면 좋다.

        outline_template은 예시로 앞부분만 제시한 것이다. 각 장은 ':---CHAPTER DIVIDER---:'로 구분한다.
        outline_template:
        {outline_template}

        사용자가 추가 피드백을 제공할 수 있도록 논리적인 흐름과 주요 목차 아이디어를 제안하라.    

        """
    )

    content_strategist_chain = content_strategist_system_prompt | llm | StrOutputParser()

    user_request = state.get("user_request", "")  # business_analyst 가 생성한 사용자 요구사항
    messages = state["messages"]  # 이전까지의 Message들 
    outline = get_outline(current_path)   # 저장된 목차 가져오기

    # 템플릿 이용
    with open(f"{current_path}/templates/outline_template.md", "r", encoding="utf-8") as f:
        outline_template = f.read() 

    inputs = {
        "user_request": user_request,
        "task": task,
        "messages": messages,
        "outline": outline,
        "references": state.get("references", {"queries":[], "docs":[]}),
        "outline_template": outline_template,
    }

    # 목차 작성
    gathered = ''
    # ↓ 여기서 chunk 는 str 이다 <- StrOutputParser()
    for chunk in content_strategist_chain.stream(inputs):  
        gathered += chunk
        print(chunk, end='')

    print()

    save_outline(current_path, gathered)  # 체인을 이용해 만든 목차 저장.

    # 템플릿 을 활용한 작업 후기 메세지 찾기
    if '-----: DONE :-----' in gathered:
        review = gathered.split('-----: DONE :-----')[1]
    else:
        review = gathered[-200:]

    # AIMesage 추가
    # 목차작성 템플릿 파일에는 목차를 작성하고 그 후기를 적으라는 내용이 포함되어 있다.
    # 이를 가져와서 작업후기를 messages 에 추가.
    # 이를 통해 content_strategist 가 작업한 후기를 보고 다른 에이전트들이 지금 무슨 일이 어떻게 진행되는지 파악할수 있다
    content_strategist_message = f"[Content Strategist] 목차 작성 완료: outline 작성 완료\n {review}"
    print(content_strategist_message)
    messages.append(AIMessage(content_strategist_message)) 

    # ■ 최근 task 작업 완료(done) 처리하기

    task_history[-1].done = True
    task_history[-1].done_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ■ 새 task 추가.   다음작업이 communicator
    # new_task = Task(
    #     agent="communicator",
    #     done=False,
    #     description="AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다",
    #     done_at="",
    # )
    # task_history.append(new_task)

    # print(new_task) # 확인용

    print("\n\n💛====== CONTENT STRATEGIST 종료======")
    
    return {
        "messages": messages,
        "task_history": task_history,
        } 

# 🟦 목차를 검토하는 노드 agent: 
def outline_reviewer(state: State):
    print("\n\n💖====== OUTLINE REVIEWER ======")

    # 시스템 프롬프트 정의
    outline_reviewer_system_promprt = PromptTemplate.from_template(
        """
        너는 AI팀의 목차 리뷰어로서, AI팀이 작성한 목차(outline)를 검토하고 문제점을 지적한다.

        - outline이 사용자의 요구사항을 충족시키는지 여부
        - outline의 논리적인 흐름이 적절한지 여부
        - 근거에 기반하지 않은 내용이 있는지 여부
        - 주어진 참고자료(references)를 충분히 활용했는지 여부
        - 참고자료가 충분한지, 혹은 잘못된 참고자료가 있는지 여부
        - example.com 같은 더미 URL이 있는지 여부:
        - 실제 페이지 URL이 아닌 대표 URL로 되어 있는 경우 삭제 해야함: 어떤 URL이 삭제되어야 하는지 명시하라.
        - 기타 리뷰 사항

        그 분석 결과를 설명하고, 다음 어떤 작업을 하면 좋을지 제안하라.
       
        - 분석결과: outline이 사용자의 요구사항을 충족시키는지 여부
        - 제안사항: (vector_search_agent, communicator 중 어떤 agent를 호출할지)

        ------------------------------------------
        user_request: {user_request}
        ------------------------------------------
        references: {references}
        ------------------------------------------
        outline: {outline}
        ------------------------------------------
        messages: {messages}

        """
    )

    # inputs 준비
    user_request = state.get("user_request", None)
    outline = get_outline(current_path)
    references = state.get("references", {"queries": [], "docs": []})
    messages = state.get("messages", [])

    inputs = {
        "user_request": user_request,
        "outline": outline,
        "references": references,
        "messages": messages
    }

    # 체인
    outline_reviewer_chain = outline_reviewer_system_promprt | llm

    # 체인호출. 
    # 목차리뷰가 길어질수 있다. 스트림호출
    review = outline_reviewer_chain.stream(inputs)

    gathered = None

    for chunk in review:
        print(chunk.content, end='')  # 확인용

        if gathered is None:
            gathered = chunk  # 첫 chunk Message
        else:
            gathered += chunk  # chunk Message 객체는 덧셈 가능

    # 작업후기를 messages 에 추가
    if '[OUTLINE REVIEW AGENT]' not in gathered.content:
        gathered.content = f"[OUTLINE REVIEW AGENT] {gathered.content}"
    print('🟡', gathered.content)  
    messages.append(gathered)

    # 리뷰 결과는 ai_recommendation 의 값으로도 활용
    ai_recommendation = gathered.content

    print("\n\n💖====== OUTLINE REVIEWER 종료 ======")

    # 상태 업데이트
    return {"messages": messages, "ai_recommendation": ai_recommendation}


# 🟦 웹 검색을 하는 노드 agent: 
def web_search_agent(state: State):
    print("\n\n🤎====== WEB SEARCH AGENT ======")

    tasks = state.get("task_history", [])
    task = tasks[-1]
    
    if task.agent != "web_search_agent":
        raise ValueError(f"Web Search Agent가 아닌 agent가 Web Search Agent를 시도하고 있습니다.\n {task}")

    web_search_system_prompt = PromptTemplate.from_template(
        """
        너는 다른 AI Agent 들이 수행한 작업을 바탕으로,
        목차(outline) 작성에 필요한 정보를 웹 검색을 통해 찾아내는 Web Search Agent이다.

        현재 부족한 정보를 검색하고, 복합적인 질문은 나눠서 검색하라.

        - 검색 목적: {mission}
        --------------------------------
        - 과거 검색 내용: {references}
        --------------------------------
        - 이전 대화 내용: {messages}
        --------------------------------
        - 목차(outline): {outline}
        --------------------------------
        - 현재 시각 : {current_time}
        """
    )

    # 기존 대화 내용
    messages = state.get("messages", [])

    # input 자료 준비
    inputs = {
        "mission": task.description,
        "references": state.get("references", {"queries": [], "docs": []}),
        "messages": messages,
        "outline": get_outline(current_path),
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # LLM 과 웹검색도구 연결
    llm_with_web_search = llm.bind_tools([web_search])

    # 체인
    web_search_chain = web_search_system_prompt | llm_with_web_search

    # llm 이 판단해 검색할 리스트 받아온다. (웹 검색 tool_calls 가져오기)
    search_plans = web_search_chain.invoke(inputs)

    # web_search_agent 에서 어떤 작업을 했는지 메세지로 남기기 위해
    # 빈 리스트 준비
    queries = []

    for tool_call in search_plans.tool_calls:
        print('------ web search ------', tool_call)
        args = tool_call['args']
        queries.append(args['query'])

        # 검색결과 -> json
        _, json_path = web_search.invoke(args)
        print('json_path:', json_path)

        # json 파일을 chroma 에 추가
        add_web_pages_json_to_chroma(json_path)

    # task 완료
    tasks[-1].done = True
    tasks[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 새로운 task 추가
    task_desc = "AI팀이 쓸 책의 세부 목차를 결정하기 위한 정보를 벡터 검색을 통해 찾아낸다."
    task_desc += f" 다음 항목이 새로 추가되었다\n: {queries}"

    new_task = Task(
        agent = "vector_search_agent",
        done = False,
        description=task_desc,
        done_at = "",
    )

    tasks.append(new_task)

    # 작업후 메세지
    msg_str = f"[WEB SEARCH AGENT] 다음 질문에 대한 검색 완료: {queries}"
    messages.append(AIMessage(msg_str))

    print("\n\n🤎====== WEB SEARCH AGENT 종료======")

    # state 업데이트
    return {
        "messages": messages,
        "task_history": tasks,
    }


# 🟦 사용자와 대화할 노드 agent: communicator
# 사용자와 상호 작용하며 대화하는 임무 수행
def communicator(state: State):
    print("\n\n🧡====== COMMUNICATOR ======")

    communicator_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 커뮤니케이터로서,
        AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다.

        사용자도 outline(목차)를 이미 보고 있으므로, 다시 출력할 필요는 없다.

        outline: {outline}
        ----------------------------------------
        messages: {messages}
        """     # ↑ 기존 대화 내용 받아서 답변 할수 있도록 한다.
    )

    system_chain = communicator_system_prompt | llm

    # 상태에서 messages 가져오기
    messages = state['messages']

    inputs = {
        "messages": messages,
        "outline": get_outline(current_path),
    }

    # 스트림 되는 메세지를 모으기
    gathered = None

    print("\nAI\t: ", end='')

    # stream() 의 결과는 Chunk 객체 (AIMessageChunk 등..)
    for chunk in system_chain.stream(inputs):
        print(chunk.content, end='')   # 스트림 되는 내용을 화면에 출력

        if gathered is None:
            gathered = chunk   
        else:
            gathered += chunk  # Chunk 객체끼리 덧셈 가능. 스트리밍에서 토큰 조각을 이어붙일수 있다!

    messages.append(gathered)


    # ■ communicator task 완료 처리
    task_history = state.get("task_history", [])
    if task_history[-1].agent != "communicator":
        raise ValueError(f"💥Communicator가 아닌 agent가 대화를 시도하고 있습니다.\n {task_history[-1]}")
    task_history[-1].done = True
    task_history[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    #  ■ 추가할 Task 는 없다!

    print("\n🧡====== COMMUNICATOR 종료 ======")

    return {
        "messages": messages,
        "task_history": task_history,
        # ↓ communicator 가 사용자 보고 마치면 0으로 리셋하며 다시 supervisor 호출이 2회 초과할때까지 에이전트들끼리 작업할수 있도록 함. 
        "supervisor_call_count": 0,
    }  # 상태 업데이트

# 🟦 supervisor agent 
def supervisor(state: State):
    print("\n\n💚====== SUPERVISOR ======")

    supervisor_system_prompt = PromptTemplate.from_template(
        """
        너는 AI 팀의 supervisor로서 AI 팀의 작업을 관리하고 지도한다.
        사용자가 원하는 책을 써야 한다는 최종 목표를 염두에 두고,
        사용자의 요구를 달성하기 위해 현재 해야할 일이 무엇인지 결정한다.

        supervisor가 활용할 수 있는 agent는 다음과 같다.    
        - content_strategist: 사용자의 요구사항이 명확해졌을 때 사용한다. AI 팀의 콘텐츠 전략을 결정하고, 전체 책의 목차(outline)를 작성한다.
        - communicator: AI 팀에서 해야 할 일을 스스로 판단할 수 없을 때 사용한다. 사용자에게 진행상황을 사용자에게 보고하고, 다음 지시를 물어본다.
        - web_search_agent: vector_search_agent를 시도하고, 검색 결과(references) 에 필요한 정보가 부족한 경우 사용한다.  웹 검색을 통해 해당 정보를 벡터 DB에 보강한다.
        - vector_search_agent: 벡터 DB 검색을 통해 목차(outline) 작성에 필요한 정보를 확보한다.

        아래 내용을 고려하여, 현재 해야할 일이 무엇인지, 사용할 수 있는 agent를 단답으로 말하라.

        ------------------------------------------
        previous_outline: {outline}
        ------------------------------------------
        messages:
        {messages}
        """
    )

    # llm 의 출력을 Task 필드에 맞춰 내용을 생성하도록 체인 생성!
    supervisor_chain = supervisor_system_prompt | llm.with_structured_output(Task)
    
    messages = state.get('messages', [])

    inputs = {
        "messages": messages,
        "outline": get_outline(current_path)
    }

    # supervisor 호출 횟수 확인
    supervisor_call_count = state.get("supervisor_call_count", 0)

    if supervisor_call_count > 2:
        print("📣Supervisor 호출 횟수 초과: Communicator 호출")
        task = Task(
            agent="communicator",
            done=False,
            description="supervisor 호출 횟수 초과했으므로, 현재까지의 진행상황을 사용자에게 보고한다.",
            done_at = "",
        )

    else:
        # 체인을 invoke 한 결과는 Task 다!  
        task = supervisor_chain.invoke(inputs)

    # 이를 state 의 task_history 에 추가.  해야될 일 추가!
    task_history = state.get("task_history", [])
    task_history.append(task)

    # AIMessage 추가
    supervisor_message = AIMessage(f"[Supervisor] {task}")
    messages.append(supervisor_message)
    print(supervisor_message.content)  # 확인

    print("\n\n💚====== SUPERVISOR 종료======")

    return {
        "messages": messages, 
        "task_history": task_history,
        "supervisor_call_count": supervisor_call_count + 1,
    }

# 위 task 에 따라 라우팅
def supervisor_router(state: State):
    task = state['task_history'][-1]  # 가장 최근에 추가된 Task
    return task.agent  # "content_strategist" 혹은 "communicator"


def vector_search_agent(state: State):
    print("\n\n💙====== VECTOR SEARCH AGENT ======")

    # 가장 최근의 작업을 가져와 해당 작업을 vector_search_agent 에서 처리할수 있는지 확인
    tasks = state.get("task_history", [])
    task = tasks[-1]
    if task.agent != "vector_search_agent":
        raise ValueError(f"Vector Search Agent가 아닌 agent가 Vector Search Agent를 시도하고 있습니다.\n {task}")


    vector_search_system_prompt = PromptTemplate.from_template(
        """
        너는 다른 AI Agent 들이 수행한 작업을 바탕으로,
        목차(outline) 작성에 필요한 정보를 벡터 검색을 통해 찾아내는 Agent이다.

        현재 목차(outline)을 작성하는데 필요한 정보를 확보하기 위해,
        다음 내용을 활용해 적절한 벡터 검색을 수행하라.

        - 검색 목적: {mission}
        --------------------------------
        - 과거 검색 내용: {references}
        --------------------------------
        - 이전 대화 내용: {messages}
        --------------------------------
        - 목차(outline): {outline}
        """
    )
    
    mission = task.description
    references = state.get("references", {"queries": [], "docs": []})
    messages = state["messages"]
    outline = get_outline(current_path)

    inputs = {
        "mission": mission,
        "references": references,
        "messages": messages,
        "outline": outline,
    }

    llm_with_retriever = llm.bind_tools([retrieve])
    vector_search_chain = vector_search_system_prompt | llm_with_retriever

    search_plans = vector_search_chain.invoke(inputs)

    # 검색할 내용 출력
    for tool_call in search_plans.tool_calls:
        print('-------------------------', tool_call)

        args = tool_call['args']
        query = args['query']

        retrieved_docs = retrieve.invoke(query)

        references['queries'].append(query)
        references['docs'] += retrieved_docs

    # 벡터스토어에 저장된 문서의 종류가 많지 않으면...
    # 혹은 하나의 문서에 여러 질문에 대한 답을 모두 포함하고 있다면...
    # 같은 문서가 반복해서 나올수 있다..
    # 같은 문서가 서로 다른 query 에 대해 반복해서 나오더라도 중복없이 처리하기 위해 page_content 를 set을 사용해
    # 담아보자
    unique_docs = []
    unique_page_contents = set()  

    for doc in references['docs']:
        if doc.page_content not in unique_page_contents:
            unique_docs.append(doc)
            unique_page_contents.add(doc.page_content)

    # 이렇게 중복없이 정리된 문서들만 references['docs'] 에 담는다
    references['docs'] = unique_docs

    # ③ 검색 결과 출력 – 쿼리 출력
    # 검색한 질의들 (queries)과 검색된 청크 문서들을 출력해보기.
    print('Queries:--------------------------')
    queries = references["queries"]
    for query in queries:
        print(query)
   
    # 검색 결과 출력 – 문서 청크 출력
    print('References:--------------------------')
    for doc in references["docs"]:
        print(doc.page_content[:100])
        print('--------------------------')    

    # 현재 task 완료
    tasks[-1].done = True
    tasks[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # communicator 로 보내는 Task 삭제, business_analyst 에게 조언할수 있게 수정
    #  -> ai_recommendation 을 작성하도록 수정

    # new_task = Task(
    #     agent="communicator",
    #     done=False,
    #     description="AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다",
    #     done_at=""
    # )
    # tasks.append(new_task)    


    msg_str = f"[VECTOR SEARCH AGENT] 다음 질문에 대한 검색 완료: {queries}"
    message = AIMessage(msg_str)
    print(msg_str)
    messages.append(message)

    ai_recommendation = "현재 참고자료(references)가 목차(outline)를 개선하는데 충분한지 확인하라. 충분하다면 content_strategist로 목차 작성을 하라."

    print("\n\n💙====== VECTOR SEARCH AGENT 종료 ======")

    # state 업데이트
    return {
        "messages": messages,
        "task_history": tasks,
        "references": references,
        "ai_recommendation": ai_recommendation,
    }


# 🟦 그래프 정의
graph_builder = StateGraph(State)

# Nodes
graph_builder.add_node("business_analyst", business_analyst)
graph_builder.add_node("outline_reviewer", outline_reviewer)
graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("communicator", communicator)
graph_builder.add_node("content_strategist", content_strategist)
graph_builder.add_node("vector_search_agent", vector_search_agent)
graph_builder.add_node("web_search_agent", web_search_agent)

# Edges
graph_builder.add_edge(START, "business_analyst")
graph_builder.add_edge("business_analyst", "supervisor")
graph_builder.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        "content_strategist": "content_strategist",
        "communicator": "communicator", 
        "vector_search_agent": "vector_search_agent",
        "web_search_agent": "web_search_agent",
    }
)

graph_builder.add_edge("content_strategist", "outline_reviewer")
graph_builder.add_edge("outline_reviewer", "business_analyst")
graph_builder.add_edge("web_search_agent", "vector_search_agent")
graph_builder.add_edge("vector_search_agent", "business_analyst")
graph_builder.add_edge("communicator", END)

graph = graph_builder.compile()

# 🟦 상태초기화
# 전체 워크플로의 시스템 메세지 작성
state = State(
    messages = [
        SystemMessage(
                f"""
            너희 AI들은 사용자의 요구에 맞는 책을 쓰는 작가팀이다.
            사용자가 사용하는 언어로 대화하라.

            현재시각은 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}이다.

            """            
            # 초기 시스템 프롬프트에 '현재 시간' 을 명시하는 이유
            # 향후 책/보고서 작성시 LLM 이 만들어진 시점 기준으로 판단하는 오류를 방지하게 하기 위함.
        )
    ],

    task_history= [],
    references={"queries": [], "docs": []},
    user_request="",
)

# 사용자 입력을 받아 graph 실행하는 코드
while True:
    user_input = input("\nUser\t: ").strip()

    if user_input.lower() in ['exit', 'quit', 'q']:
        print('Goodbye!')
        break

    state['messages'].append(HumanMessage(user_input))
    state = graph.invoke(state)

    print('\n-------------------------- MESSAGE COUNT\t', len(state['messages']))

    save_state(current_path, state)  # 현재 state 내용 저장.



