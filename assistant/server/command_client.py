import http.client  # shaves 100ms off command-issue app runs compared to requests

_SIMPLE_POST_COMMANDS = {
    "toggle_terminal",
    "confirm",
    "stop",
    "toggle_recording",
    "snippet",
}


def port_is_taken(port: int) -> bool:
    try:
        conn = http.client.HTTPConnection("localhost", port, timeout=0.5)
        conn.request("GET", "/health")
        conn.getresponse()
        conn.close()
        return True
    except (ConnectionRefusedError, OSError, http.client.HTTPException):
        return False


def send_command(port: int, command_name: str) -> bool:
    try:
        if command_name in _SIMPLE_POST_COMMANDS:
            conn = http.client.HTTPConnection("localhost", port, timeout=2)
            conn.request("POST", f"/{command_name}")
            response = conn.getresponse()
            conn.close()
            return response.status == 200
        else:
            print(f"Unknown command: {command_name}")
            return False
    except (ConnectionRefusedError, OSError, http.client.HTTPException):
        print(f"Could not connect to server at localhost:{port}")
        return False
