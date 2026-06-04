import os
import json

def save_state(current_path, state):
    """state 를 json 파일로 저장"""
    if not os.path.exists(f"{current_path}/data"):
        os.makedirs(f"{current_path}/data")
   
    state_dict = {}

    # List[(Message타입, Message내용)] 형태로 저장
    state_dict['messages'] = [(m.__class__.__name__, m.content) for m in state['messages']]

    # 작업이력도 함께 저장.
    state_dict['task_history'] = [task.to_dict() for task in state.get("task_history", [])]
   
    with open(f"{current_path}/data/state.json", "w", encoding='utf-8') as f:
        json.dump(state_dict, f, indent=4, ensure_ascii=False)

def get_outline(current_path):
    outline = '아직 작성된 목차가 없습니다.'

    if os.path.exists(f"{current_path}/data/outline.md"):
        with open(f"{current_path}/data/outline.md", "r", encoding='utf-8') as f:
            outline = f.read()  
    return outline

def save_outline(current_path, outline):
    if not os.path.exists(f"{current_path}/data"):
        os.makedirs(f"{current_path}/data")
   
    with open(f"{current_path}/data/outline.md", "w", encoding='utf-8') as f:
        f.write(outline)
    return outline