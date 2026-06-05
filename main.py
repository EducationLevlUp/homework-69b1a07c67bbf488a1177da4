"""
Интерактивная мини-игра «выбери свою историю» на LangGraph.

Использует:
- StateGraph с TypedDict-состоянием
- InMemorySaver как чекпоинтер
- interrupt / Command(resume=...) для human-in-the-loop
- questionary.select для выбора в консоли
- Два вызова LLM: завязка + концовка
"""

import sys

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command
from nodes import story_node
from state import StoryState

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

DEFAULT_TOPIC = "космический кот"


def get_topic() -> str:
    """Получить тему из аргументов командной строки или использовать дефолтную."""
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    return DEFAULT_TOPIC


# ---------------------------------------------------------------------------
# Сборка графа
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Собрать и скомпилировать граф с чекпоинтером."""
    checkpointer = InMemorySaver()

    graph = StateGraph(StoryState)
    graph.add_node("story", story_node)
    graph.add_edge(START, "story")

    compiled = graph.compile(checkpointer=checkpointer)
    return compiled


# ---------------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------------

def main() -> None:
    """Запустить интерактивную историю."""
    topic = get_topic()
    graph = build_graph()

    config = {"configurable": {"thread_id": "story_session_1"}}
    initial_state: StoryState = {
        "topic": topic,
        "intro_text": "",
        "choices": [],
        "user_choice": "",
        "ending_text": "",
    }

    print(f"Тема: {topic}\n")

    # --- Первый проход: до прерывания ---
    for chunk in graph.stream(initial_state, config):
        if "__interrupt__" in chunk:
            # Извлекаем payload из прерывания
            interrupt_value = chunk["__interrupt__"][0].value

            question = interrupt_value.get("question", "Что делаем?")
            choices = interrupt_value.get("choices", [])

            print(f"[LLM] {question.split(chr(10))[0]}")
            print()

            # --- Выбор пользователя через questionary ---
            try:
                import questionary
            except ImportError:
                print("Ошибка: questionary не установлен. Выполните: pip install questionary")
                sys.exit(1)

            choice_text = questionary.select(
                "Что делаем?",
                choices=choices,
            ).ask()

            if not choice_text:
                choice_text = choices[0] if choices else "Пойти вперёд"

            # Добавляем выбор в payload
            interrupt_value["user_choice"] = choice_text

            # --- Возобновление графа ---
            print(f"\n> {choice_text}\n")

            for chunk in graph.stream(
                Command(resume=interrupt_value), config
            ):
                # После возобновления узел вернёт обновлённое состояние
                if "__interrupt__" not in chunk:
                    # chunk — это словарь вида {"node_name": updated_state}
                    for node_name, node_output in chunk.items():
                        if node_name != "__interrupt__" and isinstance(node_output, dict):
                            intro = node_output.get("intro_text", "")
                            ending = node_output.get("ending_text", "")
                            user_choice = node_output.get("user_choice", "")

                            if ending:
                                print(f"[LLM] {ending}\n")

                            # --- Итоговый вывод ---
                            print("=" * 50)
                            print("ИТОГОВАЯ ИСТОРИЯ")
                            print("=" * 50)
                            print(f"Тема: {topic}")
                            print(f"\nЗавязка:\n{intro}")
                            print(f"\nВыбор: {user_choice}")
                            print(f"\nКонцовка:\n{ending}")
                            print("=" * 50)


if __name__ == "__main__":
    main()
