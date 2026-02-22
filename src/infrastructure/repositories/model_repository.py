from typing import Dict, Any, Tuple
from src.domain.interfaces import IModelRepository


class InMemoryModelRepository(IModelRepository):
    """
    Реализация репозитория моделей, хранящая данные в оперативной памяти.

    Модели сохраняются во внутреннем словаре, где ключом является строковый идентификатор модели,
    а значением — кортеж (бинарные данные модели, словарь метаданных).
    """

    def __init__(self):
        """Инициализирует пустое хранилище моделей."""
        self._storage: Dict[str, Tuple[bytes, Dict[str, Any]]] = {}

    def save(self, model_id: str, model_data: bytes, metadata: Dict[str, Any]) -> None:
        """
        Сохраняет модель и её метаданные в хранилище.

        Параметры
        ----------
        model_id : str
            Уникальный идентификатор модели.
        model_data : bytes
            Бинарные данные обученной модели
        metadata : Dict[str, Any]
            Словарь с метаданными модели (серия, горизонт, лаги, стратегия, метрики и т.д.).
        """
        self._storage[model_id] = (model_data, metadata)

    def load(self, model_id: str) -> Tuple[bytes, Dict[str, Any]]:
        """
        Загружает модель и её метаданные по идентификатору.

        Параметры
        ----------
        model_id : str
            Идентификатор модели, которую необходимо загрузить.

        Возвращает
        -------
        Tuple[bytes, Dict[str, Any]]
            Кортеж (бинарные данные модели, метаданные).

        Исключения
        ----------
        KeyError
            Если модель с указанным model_id отсутствует в хранилище.
        """
        return self._storage[model_id]
