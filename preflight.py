"""Check the real local model's tool calling and sandbox lifecycle before a long run."""
from model import make_model
from sandbox import download, open_sandbox, upload
from langchain_core.tools import tool
from agents import configure_model


@tool
def lab_echo(value: str) -> str:
    """Echo a value for a local tool-calling connectivity test."""
    return value


def main():
    model = configure_model(make_model())
    response = model.bind_tools([lab_echo]).invoke("Call lab_echo with value 'ready'. Use the tool, do not answer in prose. /no_think")
    if not response.tool_calls or response.tool_calls[0]["name"] != "lab_echo":
        raise RuntimeError("Configured model failed the tool-calling test")
    print("OK: configured model tool calling", flush=True)
    with open_sandbox() as backend:
        payload = {"/tmp/work/preflight.txt": b"ready"}
        upload(backend, payload)
        if download(backend, list(payload)) != payload:
            raise RuntimeError("sandbox upload/download failed")
        result = backend.execute("python3 -c 'print(2 + 2)'")
        if result.exit_code or result.output.strip() != "4":
            raise RuntimeError("sandbox execution failed")
    print("OK: sandbox execution, upload/download and cleanup", flush=True)


if __name__ == "__main__":
    main()
