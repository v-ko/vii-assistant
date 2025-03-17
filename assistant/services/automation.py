class AutomationService:
    """Service for automatic querying."""

    def __init__(self):
        self._running = False
        self._system_prompt = ""

    def start(self):
        """Start the automation service."""
        if self._running:
            print("Error: Automation service already running")
            return
        self._running = True
        print("Automation service started")

    def stop(self):
        """Stop the automation service."""
        if not self._running:
            print("Error: Automation service not running")
            return
        self._running = False
        print("Automation service stopped")

    def is_running(self) -> bool:
        """Check if the automation service is running."""
        return self._running

    def set_system_prompt(self, prompt: str):
        """Set the system prompt."""
        self._system_prompt = prompt
        print(f"System prompt set: {prompt}")

    def get_system_prompt(self) -> str:
        """Get the system prompt."""
        return self._system_prompt
