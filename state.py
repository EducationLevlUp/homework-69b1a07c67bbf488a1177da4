"""Определение состояния графа интерактивной истории."""

from typing import TypedDict


class StoryState(TypedDict):
    """Состояние графа с полями для темы, текста сцены, вариантов, выбора и концовки."""

    topic: str
    scene_text: str
    choices: list[str]
    user_choice: str
    ending: str
