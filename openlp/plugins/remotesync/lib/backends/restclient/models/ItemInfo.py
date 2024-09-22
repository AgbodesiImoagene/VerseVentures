from typing import *

from pydantic import BaseModel, Field


class ItemInfo(BaseModel):
    """
    None model

    """

    uuid: Optional[str] = Field(alias="uuid", default=None)

    version: Optional[int] = Field(alias="version", default=None)

    user: Optional[str] = Field(alias="user", default=None)

    timestamp: Optional[str] = Field(alias="timestamp", default=None)

    title: Optional[str] = Field(alias="title", default=None)
