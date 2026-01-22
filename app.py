import asyncio
import datetime
from abc import ABC, abstractmethod
from typing import Any
import click
from google import genai
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Textual Imports for the modern UI
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Header, Footer, Input, Markdown, Static

# --- App Configuration ---
class AppSettings(BaseSettings):
    gemini_api_key: SecretStr = Field(..., validation_alias='GEMINI_API_KEY')
    heavy_model: str = Field("gemini-2-pro-preview")
    light_model: str = Field("gemini-2.5-flash")

    model_config = SettingsConfigDict(
        env_file='.env', 
        env_file_encoding='utf-8', 
        extra='ignore'
    )

# --- The Interface & Workers ---
class LLMInterface(ABC):
    @abstractmethod
    async def process_task(self, prompt_text: str) -> str:
        pass

class FlashImplementation(LLMInterface):
    def __init__(self, external_client: Any, model_id: str):
        self._scribe_tool, self._target = external_client, model_id

    async def process_task(self, prompt_text: str) -> str:
        response = await self._scribe_tool.aio.models.generate_content(model=self._target, contents=prompt_text)
        return response.text

class HighReasoningImplementation(LLMInterface):
    def __init__(self, external_client: Any, model_id: str):
        self._scribe_tool, self._target = external_client, model_id

    async def process_task(self, prompt_text: str) -> str:
        response = await self._scribe_tool.aio.models.generate_content(model=self._target, contents=prompt_text)
        return response.text

def route_logic(user_input: str, client: genai.Client, settings: AppSettings) -> tuple[LLMInterface, str]:
    if len(user_input) < 100:
        return FlashImplementation(client, settings.light_model), "Gemini Flash (Fast)"
    return HighReasoningImplementation(client, settings.heavy_model), "Gemini Pro (Reasoning)"

# --- Textual UI Implementation ---
class RouterTUI(App):
    """A modern terminal interface for the LLM Router."""
    
    CSS = """
    Screen {
        align: center middle;
    }
    #results-container {
        width: 90%;
        height: 70%;
        border: solid round $accent;
        padding: 1;
        margin: 1;
    }
    Input {
        width: 90%;
        margin: 1;
    }
    """

    BINDINGS = [("q", "quit", "Quit App"), ("c", "clear", "Clear Screen")]

    def __init__(self, client, settings, initial_prompt=None):
        super().__init__()
        self.client = client
        self.settings = settings
        self.initial_prompt = initial_prompt

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll(id="results-container"):
            yield Markdown("# Welcome Scribe\nSubmit a prompt below to begin.", id="md_viewer")
        yield Input(placeholder="Enter your prompt here...", id="user_input")
        yield Footer()

    async def on_mount(self) -> None:
        """Handle prompt passed via CMD argument on startup."""
        if self.initial_prompt:
            input_widget = self.query_one("#user_input", Input)
            input_widget.value = self.initial_prompt
            await self.handle_llm_request(self.initial_prompt)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        await self.handle_llm_request(event.value)

    async def handle_llm_request(self, user_text: str) -> None:
        md_viewer = self.query_one("#md_viewer", Markdown)
        input_widget = self.query_one("#user_input", Input)
        
        # UI Feedback
        input_widget.value = ""
        await md_viewer.update("## ⏳ Thinking...")

        # 1. Routing
        worker, model_name = route_logic(user_text, self.client, self.settings)
        
        # 2. Execution
        try:
            raw_response = await worker.process_task(user_text)
            
            # b) The Markdown Template
            formatted_md = f"""
# 🤖 Assistant Response
**Model Used:** `{model_name}`
**Timestamp:** *{datetime.datetime.now().strftime('%H:%M:%S')}*

---

{raw_response}

---
**Prompt Sent:**
> {user_text}
"""
            # c) Rendering the Markdown
            await md_viewer.update(formatted_md)
            
        except Exception as e:
            await md_viewer.update(f"# ❌ Error\n{str(e)}")

    def action_clear(self):
        self.query_one("#md_viewer", Markdown).update("# Cleared")

# --- Orchestration ---
@click.command()
@click.argument('prompt', required=False)
def cli(prompt: str):
    """Launch the Intelligent Router TUI."""
    settings = AppSettings()
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    
    # Launch Textual App
    app = RouterTUI(client, settings, initial_prompt=prompt)
    app.run()

if __name__ == "__main__":
    cli()