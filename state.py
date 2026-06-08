"""Определение состояния графа интерактивной истории.

Используем TypedDict для строгой типизации полей состояния.
Каждое поле соответствует этапу генерации истории:
  - topic: тема, заданная пользователем
  - scene_text: завязка, сгенерированная LLM
  - choices: список вариантов поступка
  - user_choice: выбор пользователя после interrupt
  - ending: концовка, сгенерированная LLM на основе выбора
"""

from typing import TypedDict


class StoryState(TypedDict):
    """Состояние графа с полями для темы, текста сцены, вариантов, выбора и концовки."""

    topic: str
    scene_text: str
    choices: list[str]
    user_choice: str
    ending: str
