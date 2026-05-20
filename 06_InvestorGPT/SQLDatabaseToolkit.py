# movies.sqlite
# https://www.dropbox.com/scl/fi/ja3xxtojs4is9oteovd6j/movies.sqlite?rlkey=hcomrb2yez52147ocs2o4vmyi&st=s8mr314l&dl=1


from dotenv import load_dotenv
load_dotenv()

import os
from langchain_openai.chat_models.base import ChatOpenAI

# sql agent 생성함수
from langchain_community.agent_toolkits.sql.base import create_sql_agent

# tool kit
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

from langchain_community.utilities.sql_database import SQLDatabase

file_dir = os.path.dirname(os.path.realpath(__file__))
db_path = os.path.join(file_dir, 'movies.sqlite')

db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

# print(db) # 확인

# toolkit 생성
llm = ChatOpenAI(temperature=0.1)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# agent 생성
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    agent_type="tool-calling",     # LangChain v1.x 방식
    verbose=True, 
)

# agent 호출
# agent.invoke("Give me 5 directors that have the highest grossing films.")


# agent.invoke("Give me the movies that have the highest votes but the lowest budgets and give me the name of their directors.")

print(toolkit.get_tools())















