"""Part of scripts/docker_smoke.sh: get a personal access token for Folio from the MCP Gate container of the stack.

Signs in to MCP Gate's console API as the owner created from ADMIN_EMAIL / ADMIN_PASSWORD, registers Folio as a
service (host `folio`, port 5039, the given audience) unless it is already there, issues a personal access token for
it and prints the token. Standard library only. Arguments: MCP Gate URL (the issuer), email, password, audience.
"""

import http.cookiejar
import json
import sys
import urllib.request

BASE, EMAIL, PASSWORD, AUDIENCE = sys.argv[1:5]
SERVICE = "Folio"

opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
headers = {"Content-Type": "application/json", "Origin": BASE}


def call(method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    with opener.open(request, timeout=30) as response:
        return json.load(response)


session = call("POST", "/api/session/login", {"email": EMAIL, "password": PASSWORD})
headers["X-Session-Binding"] = session["session_binding"]

service = next((s for s in call("GET", "/api/services")["services"] if s["name"] == SERVICE), None)
if service is None:
    service = call("POST", "/api/services", {"name": SERVICE, "host": "folio", "port": 5039, "mcp_path": "/mcp",
                                             "oauth_audience": AUDIENCE})
token = call("POST", "/api/oauth/tokens/personal",
             {"service_id": service["id"], "expires_days": 1, "label": "docker smoke"})["access_token"]
print(token)
