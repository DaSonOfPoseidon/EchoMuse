"""
The two middlewares every dashboard and API request passes through.

`_ingress_only_middleware` is the add-on's only network protection: with
host networking the dashboard port is on the LAN, so anything not from
Supervisor's gateway is refused while INGRESS_ONLY is set, and nothing is
refused while it is not. `_error_middleware` gives every failure the API's
error shape: an AuthError its own status, an aiohttp HTTP exception passed
through untouched, anything else a 500 that does not carry the exception's
text to the browser.
"""

import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

import em_api
import em_auth
import em_ingressauth


def _request(remote="192.168.1.20"):
    return make_mocked_request("GET", "/api/devices").clone(remote=remote)


def _run(middleware, request, handler):
    return asyncio.run(middleware(request, handler))


async def _ok(request):
    return web.Response(text="ok")


# ── Ingress only ─────────────────────────────────────────────────────────────


def test_ingress_off_lets_any_address_through(monkeypatch):
    monkeypatch.setattr(em_ingressauth, "INGRESS_ONLY", False)
    resp = _run(em_api._ingress_only_middleware, _request(), _ok)
    assert resp.text == "ok"


def test_ingress_on_admits_the_supervisor_gateway(monkeypatch):
    monkeypatch.setattr(em_ingressauth, "INGRESS_ONLY", True)
    request = _request(remote=em_ingressauth.INGRESS_GATEWAY_IP)
    resp = _run(em_api._ingress_only_middleware, request, _ok)
    assert resp.text == "ok"


def test_ingress_on_refuses_any_other_address(monkeypatch):
    monkeypatch.setattr(em_ingressauth, "INGRESS_ONLY", True)
    called = []

    async def handler(request):
        called.append(request)
        return web.Response(text="ok")

    with pytest.raises(web.HTTPForbidden):
        _run(em_api._ingress_only_middleware, _request(), handler)
    assert called == []


# ── Error shape ──────────────────────────────────────────────────────────────


def test_a_good_response_is_returned_untouched():
    resp = _run(em_api._error_middleware, _request(), _ok)
    assert resp.status == 200
    assert resp.text == "ok"


def test_an_auth_error_keeps_its_own_status_and_code():
    async def handler(request):
        raise em_auth.AuthError("forbidden", "Admin only", 403)

    resp = _run(em_api._error_middleware, _request(), handler)
    assert resp.status == 403
    assert json.loads(resp.text)["code"] == "forbidden"


def test_an_http_exception_passes_through():
    async def handler(request):
        raise web.HTTPNotFound()

    with pytest.raises(web.HTTPNotFound):
        _run(em_api._error_middleware, _request(), handler)


def test_anything_else_is_a_500_without_the_exception_text():
    async def handler(request):
        raise RuntimeError("secret detail from inside the handler")

    resp = _run(em_api._error_middleware, _request(), handler)
    assert resp.status == 500
    body = json.loads(resp.text)
    assert body["code"] == "internal_error"
    assert "secret detail" not in resp.text
