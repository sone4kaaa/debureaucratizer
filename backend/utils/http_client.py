"""Per-loop httpx.AsyncClient: один клиент на event loop процесса."""

import asyncio

import httpx

_clients: dict[int, httpx.AsyncClient] = {}


def get_http_client(timeout: float = 300.0) -> httpx.AsyncClient:
    try:
        loop_id = id(asyncio.get_running_loop())
    except RuntimeError:
        loop_id = 0
    client = _clients.get(loop_id)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(timeout=timeout)
        _clients[loop_id] = client
    return client


async def close_all_clients() -> None:
    for client in _clients.values():
        if not client.is_closed:
            await client.aclose()
    _clients.clear()
