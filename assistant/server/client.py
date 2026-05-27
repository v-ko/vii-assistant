import requests

LOCALHOST = "http://localhost"

_SIMPLE_POST_COMMANDS = {
    "toggle_terminal",
    "confirm",
    "stop",
    "toggle_recording",
    "snippet",
}


def port_is_taken(port: int) -> bool:
    try:
        requests.get(f"{LOCALHOST}:{port}/health", timeout=0.5)
        return True
    except requests.ConnectionError:
        return False


def send_command(port: int, command_name: str) -> bool:
    try:
        if command_name in _SIMPLE_POST_COMMANDS:
            response = requests.post(f"{LOCALHOST}:{port}/{command_name}")
            return response.status_code == 200
        else:
            print(f"Unknown command: {command_name}")
            return False
    except requests.ConnectionError:
        print(f"Could not connect to server at {LOCALHOST}:{port}")
        return False
