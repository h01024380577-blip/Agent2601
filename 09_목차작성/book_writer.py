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


# 🟦 목차를 작성하는 노드 agent: content_strategist
def content_strategist(state: State):
    print("\n\n💛====== CONTENT STRATEGIST ======")

    # 지난 목차(outline)와 이전 대화 내용(messages)이 주어지면 이전 대화 내용을 바탕으로
    # 새로운 목차를 생성하라는 문구를 추가합니다.
    content_strategist_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 콘텐츠 전략가(Content Strategist)로서,
        이전 대화 내용을 바탕으로 사용자의 요구사항을 분석하고, AI팀이 쓸 책의 세부 목차를 결정한다.

        지난 목차가 있다면 그 버전을 사용자의 요구에 맞게 수정하고, 없다면 새로운 목차를 제안한다.

        --------------------------------
        - 지난 목차: {outline}
        --------------------------------
        - 이전 대화 내용: {messages}
        """
    )

    content_strategist_chain = content_strategist_system_prompt | llm | StrOutputParser()

    messages = state["messages"]  # 이전까지의 Message들 
    outline = get_outline(current_path)   # 저장된 목차 가져오기

    inputs = {
        "messages": messages,
        "outline": outline,
    }

    # 목차 작성
    gathered = ''
    # ↓ 여기서 chunk 는 str 이다 <- StrOutputParser()
    for chunk in content_strategist_chain.stream(inputs):  
        gathered += chunk
        print(chunk, end='')

    print()

    save_outline(current_path, gathered)  # 체인을 이용해 만든 목차 저장.

    # AIMesage 추가
    content_strategist_message = f"[Content Strategist] 목차 작성 완료:"
    print(content_strategist_message)
    messages.append(AIMessage(content_strategist_message))  # communicator 에 전달
    # 대화 히스토리인 messages에 결과가 있으므로 communicator는 업데이트된 messages로
    # 임무를 수행하게 됩니다. 따라서 현재의 진행 상황을 사용자에게 제대로 보고할 수 있습니다.

    # ■ 최근 task 작업 완료(done) 처리하기
    task_history = state.get("task_history", [])
    if task_history[-1].agent != "content_strategist":
        raise ValueError(f"💥Content Strategist가 아닌 agent가 목차 작성을 시도하고 있습니다.\n {task_history[-1]}")

    task_history[-1].done = True
    task_history[-1].done_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ■ 새 task 추가.   다음작업이 communicator
    new_task = Task(
        agent="communicator",
        done=False,
        description="AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다",
        done_at="",
    )
    task_history.append(new_task)

    print(new_task) # 확인용

    print("\n\n💛====== CONTENT STRATEGIST 종료======")
    
    return {
        "messages": messages,
        "task_history": task_history,
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
    }

# 위 task 에 따라 라우팅
def supervisor_router(state: State):
    task = state['task_history'][-1]  # 가장 최근에 추가된 Task
    return task.agent  # "content_strategist" 혹은 "communicator"



# 🟦 그래프 정의
graph_builder = StateGraph(State)

# Nodes
graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("communicator", communicator)
graph_builder.add_node("content_strategist", content_strategist)

# Edges
graph_builder.add_edge(START, "supervisor")
graph_builder.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        "content_strategist": "content_strategist",
        "communicator": "communicator", 
    }
)

graph_builder.add_edge("content_strategist", "communicator")
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



