from typing import *

from pydantic import BaseModel, Field

from .ItemInfo import ItemInfo


class ItemList(BaseModel):
    """
    None model

    """

    list: Optional[List[Optional[ItemInfo]]] = Field(alias="list", default=None)
