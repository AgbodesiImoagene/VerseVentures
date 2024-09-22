from typing import *

from pydantic import BaseModel, Field


class ApiResponse(BaseModel):
    """
    None model

    """

    code: Optional[int] = Field(alias="code", default=None)

    type: Optional[str] = Field(alias="type", default=None)

    message: Optional[str] = Field(alias="message", default=None)
