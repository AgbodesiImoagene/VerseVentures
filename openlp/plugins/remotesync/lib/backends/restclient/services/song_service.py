import json
from typing import *

import requests

from ..api_config import APIConfig, HTTPException
from ..models import *


def getSongList(api_config_override: Optional[APIConfig] = None) -> ItemList:
    api_config = api_config_override if api_config_override else APIConfig()

    base_path = api_config.base_path
    path = f"/song-list"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer { api_config.get_access_token() }",
    }
    query_params: Dict[str, Any] = {}

    query_params = {key: value for (key, value) in query_params.items() if value is not None}

    response = requests.request(
        "get",
        f"{base_path}{path}",
        headers=headers,
        params=query_params,
        verify=api_config.verify,
    )
    if response.status_code != 200:
        raise HTTPException(response.status_code, f" failed with status code: {response.status_code}")

    return ItemList(**response.json()) if response.json() is not None else ItemList()


def getSongListChangesSince(sinceDateTime: str, api_config_override: Optional[APIConfig] = None) -> ItemList:
    api_config = api_config_override if api_config_override else APIConfig()

    base_path = api_config.base_path
    path = f"/song-list/changes/{sinceDateTime}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer { api_config.get_access_token() }",
    }
    query_params: Dict[str, Any] = {}

    query_params = {key: value for (key, value) in query_params.items() if value is not None}

    response = requests.request(
        "get",
        f"{base_path}{path}",
        headers=headers,
        params=query_params,
        verify=api_config.verify,
    )
    if response.status_code != 200:
        raise HTTPException(response.status_code, f" failed with status code: {response.status_code}")

    return ItemList(**response.json()) if response.json() is not None else ItemList()


def getSong(uuid: str, api_config_override: Optional[APIConfig] = None) -> TextItem:
    api_config = api_config_override if api_config_override else APIConfig()

    base_path = api_config.base_path
    path = f"/song/{uuid}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer { api_config.get_access_token() }",
    }
    query_params: Dict[str, Any] = {}

    query_params = {key: value for (key, value) in query_params.items() if value is not None}

    response = requests.request(
        "get",
        f"{base_path}{path}",
        headers=headers,
        params=query_params,
        verify=api_config.verify,
    )
    if response.status_code != 200:
        raise HTTPException(response.status_code, f" failed with status code: {response.status_code}")

    return TextItem(**response.json()) if response.json() is not None else TextItem()


def updateSong(uuid: str, data: TextItem, api_config_override: Optional[APIConfig] = None) -> None:
    api_config = api_config_override if api_config_override else APIConfig()

    base_path = api_config.base_path
    path = f"/song/{uuid}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer { api_config.get_access_token() }",
    }
    query_params: Dict[str, Any] = {}

    query_params = {key: value for (key, value) in query_params.items() if value is not None}

    response = requests.request(
        "put", f"{base_path}{path}", headers=headers, params=query_params, verify=api_config.verify, json=data.dict()
    )
    if response.status_code != 200:
        raise HTTPException(response.status_code, f" failed with status code: {response.status_code}")

    return None


def deleteSong(uuid: str, api_config_override: Optional[APIConfig] = None) -> None:
    api_config = api_config_override if api_config_override else APIConfig()

    base_path = api_config.base_path
    path = f"/song/{uuid}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer { api_config.get_access_token() }",
    }
    query_params: Dict[str, Any] = {}

    query_params = {key: value for (key, value) in query_params.items() if value is not None}

    response = requests.request(
        "delete",
        f"{base_path}{path}",
        headers=headers,
        params=query_params,
        verify=api_config.verify,
    )
    if response.status_code != 200:
        raise HTTPException(response.status_code, f" failed with status code: {response.status_code}")

    return None


def getSongVersion(uuid: str, version: int, api_config_override: Optional[APIConfig] = None) -> TextItem:
    api_config = api_config_override if api_config_override else APIConfig()

    base_path = api_config.base_path
    path = f"/song/{uuid}/{version}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer { api_config.get_access_token() }",
    }
    query_params: Dict[str, Any] = {}

    query_params = {key: value for (key, value) in query_params.items() if value is not None}

    response = requests.request(
        "get",
        f"{base_path}{path}",
        headers=headers,
        params=query_params,
        verify=api_config.verify,
    )
    if response.status_code != 200:
        raise HTTPException(response.status_code, f" failed with status code: {response.status_code}")

    return TextItem(**response.json()) if response.json() is not None else TextItem()


def getSongHistory(uuid: str, api_config_override: Optional[APIConfig] = None) -> TextItem:
    api_config = api_config_override if api_config_override else APIConfig()

    base_path = api_config.base_path
    path = f"/song-history/{uuid}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer { api_config.get_access_token() }",
    }
    query_params: Dict[str, Any] = {}

    query_params = {key: value for (key, value) in query_params.items() if value is not None}

    response = requests.request(
        "get",
        f"{base_path}{path}",
        headers=headers,
        params=query_params,
        verify=api_config.verify,
    )
    if response.status_code != 200:
        raise HTTPException(response.status_code, f" failed with status code: {response.status_code}")

    return TextItem(**response.json()) if response.json() is not None else TextItem()


def getCustomSlideListChangesSince(sinceDateTime: str, api_config_override: Optional[APIConfig] = None) -> ItemList:
    api_config = api_config_override if api_config_override else APIConfig()

    base_path = api_config.base_path
    path = f"/custom-list/changes/{sinceDateTime}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer { api_config.get_access_token() }",
    }
    query_params: Dict[str, Any] = {}

    query_params = {key: value for (key, value) in query_params.items() if value is not None}

    response = requests.request(
        "get",
        f"{base_path}{path}",
        headers=headers,
        params=query_params,
        verify=api_config.verify,
    )
    if response.status_code != 200:
        raise HTTPException(response.status_code, f" failed with status code: {response.status_code}")

    return ItemList(**response.json()) if response.json() is not None else ItemList()
