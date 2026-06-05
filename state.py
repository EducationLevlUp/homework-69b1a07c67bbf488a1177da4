"""Определение состояния графа интерактивной истории."""

from typing import TypedDict


class StoryState(TypedDict):
    """Состояние графа с полями для завязки, выбора и концовки."""

    topic: str
    intro_text: str
    choices: list[str]
    user_choice: str
    ending_text: str
