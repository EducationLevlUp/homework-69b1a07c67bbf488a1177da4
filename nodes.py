"""Узлы графа: генерация завязки, прерывание, генерация концовки."""

import re
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from state import StoryState

# ---------------------------------------------------------------------------
# Промпты
# ---------------------------------------------------------------------------

INTRO_PROMPT_TEMPLATE = """Ты — автор интерактивной мини-истории.
Тема: {topic}

Придумай короткую завязку (2–3 предложения) и ровно 3 варианта поступка героя.
Ответь строго в следующем формате:

ЗАВЯЗКА: <текст завязки>
ВАРИАНТЫ:
1) <первый вариант>
2) <второй вариант>
3) <третий вариант>

Не добавляй ничего лишнего."""

ENDING_PROMPT_TEMPLATE = """Ты — автор интерактивной мини-истории.
Завязка: {scene_text}
Выбор пользователя: {user_choice}

Допиши короткую концовку (2–3 предложения) с учётом выбора героя.
Ответь только текстом концовки, без пояснений."""

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _parse_intro_response(response_text: str) -> tuple[str, list[str]]:
    """Разобрать ответ LLM на завязку и список вариантов."""
    lines = response_text.strip().splitlines()

    scene = ""
    choices: list[str] = []

    in_choices = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("ЗАВЯЗКА:"):
            scene = stripped.replace("ЗАВЯЗКА:", "", 1).strip()
            in_choices = False
        elif stripped.startswith("ВАРИАНТЫ:"):
            in_choices = True
            continue
        elif in_choices and re.match(r"^\d+\)", stripped):
            # Извлекаем текст после "1) "
            choice_text = re.sub(r"^\d+\)\s*", "", stripped)
            choices.append(choice_text)
        elif in_choices and stripped and not re.match(r"^\d+\)", stripped):
            # Если строка без номера, но мы в секции вариантов — игнорируем
            continue

    # Fallback: если парсинг не дал 3 варианта, используем заглушки
    if len(choices) < 3:
        choices = [
            "Пойти вперёд",
            "Осмотреться",
            "Подождать",
        ]

    if not scene:
        scene = "Герой стоит перед выбором."

    return scene, choices


def _get_llm() -> ChatOpenAI:
    """Создать экземпляр LLM."""
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# ---------------------------------------------------------------------------
# Узел графа
# ---------------------------------------------------------------------------

def story_node(state: StoryState) -> dict[str, Any]:
    """
    Единый узел, который:
    1. Вызывает LLM для генерации завязки и вариантов.
    2. Делает interrupt с вопросом и вариантами.
    3. После resume получает выбор пользователя.
    4. Вызывает LLM для генерации концовки.
    5. Возвращает обновлённое состояние.
    """
    topic = state.get("topic", "космический кот")
    llm = _get_llm()

    # --- Шаг 1: Генерация завязки ---
    intro_prompt = INTRO_PROMPT_TEMPLATE.format(topic=topic)
    intro_response = llm.invoke(intro_prompt)
    scene_text, choices = _parse_intro_response(intro_response.content)

    # --- Шаг 2: Прерывание для выбора пользователя ---
    interrupt_payload = {
        "type": "choice",
        "question": f"{scene_text}\n\nЧто делаем?",
        "choices": choices,
    }

    # interrupt() блокирует выполнение узла до resume
    resumed_payload = interrupt(interrupt_payload)

    # После resume resumed_payload — это обновлённый словарь с полем user_choice
    user_choice = resumed_payload.get("user_choice", choices[0])

    # --- Шаг 3: Генерация концовки ---
    ending_prompt = ENDING_PROMPT_TEMPLATE.format(
        scene_text=scene_text,
        user_choice=user_choice,
    )
    ending_response = llm.invoke(ending_prompt)
    ending = ending_response.content.strip()

    # --- Возврат обновлённого состояния ---
    return {
        "scene_text": scene_text,
        "choices": choices,
        "user_choice": user_choice,
        "ending": ending,
    }
