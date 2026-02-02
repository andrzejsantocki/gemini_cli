from abc import ABC, abstractmethod
from datetime import datetime
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr, Field
from typing import Any
from google import genai

# Textual imports
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Header, Footer, Input, RichLog, Label
from textual import work

import asyncio
import os
# Rich import
from rich.text import Text

# ==========================================
# PART 1: LOGIC
# ==========================================

class AppSettings(BaseSettings):
    gemini_api_key: SecretStr = Field(..., alias='GEMINI_API_KEY')
    heavy_model: str = Field("gemini-2.5-flash") 
    light_model: str = Field("gemini-flash-lite-latest") 
    picker_model: str = Field("gemini-2.0-flash-lite-preview-02-05")

    model_config = SettingsConfigDict(
        env_file='.env',
        extra='ignore'
    )


class LLMInterface(ABC):
    @abstractmethod
    async def process_task(self, prompt_text: str) -> str:
        pass


class FlashImplementation(LLMInterface):
    def __init__(self, external_client: Any, model_name: str):
        self._external_client = external_client
        self._model_name = model_name
        self._chat_session = None

    async def process_task(self, prompt_text: str):
        if self._chat_session is None:
            self._chat_session = self._external_client.aio.chats.create(
                model=self._model_name
            )
        response = await self._chat_session.send_message(prompt_text)
        return response.text

class HighReasoningImplementation(LLMInterface):
    def __init__(self, external_client: Any, model_name: str):
        self._external_client = external_client
        self._model_name = model_name
        self._chat_session = None

    async def process_task(self, prompt_text: str):
        if self._chat_session is None:
            self._chat_session = self._external_client.aio.chats.create(
                model=self._model_name
            )
        response = await self._chat_session.send_message(prompt_text)
        return response.text

async def route_logic(client: genai.Client, settings: AppSettings, user_prompt: str) -> tuple[LLMInterface, str]:
    # 1. Ask the "Picker" to classify the work
    try:
        # We use a distinct chat session just for routing
        picker = client.aio.chats.create(model=settings.picker_model)
        
        # System instruction inside the prompt for the router
        router_prompt = (
            f"Analyze this prompt: '{user_prompt}'\n"
            "If it requires complex reasoning, coding, or creative writing, reply 'HEAVY'.\n"
            "If it is a simple greeting, factual lookup, or short question, reply 'LIGHT'.\n"
            "Reply ONLY with the word HEAVY or LIGHT."
        )
        
        response = await picker.send_message(router_prompt)
        decision = response.text.strip().upper()
    except Exception:
        # Fallback if router fails: assume Heavy to be safe
        decision = "HEAVY"

    # 2. Dispatch based on the Concierge's decision
    if "HEAVY" in decision:
        return HighReasoningImplementation(client, settings.heavy_model), settings.heavy_model
    else:
        return FlashImplementation(client, settings.light_model), settings.light_model

# ==========================================
# PART 2: TEXTUAL APP
# ==========================================

class GeminiTui(App):

    ENABLE_COMMAND_PALETTE = False
    SCROLLBAR_GUTTER = 0

    CSS = """
    Screen {
        layout: vertical;
        overflow: hidden;
        height: 100%;
        scrollbar-gutter: stable;
    }
    
    #input-container {
        height: 3;
        dock: top;
        background: $boost;
    }

    Input {
        dock: left;
        width: 4fr;
    }

    #model-status {
        width: 1fr;
        content-align: center middle;
        background: $primary;
        color: auto;
    }

    RichLog {
        height: 1fr;
        border: solid green;
        background: $surface;

        /* Make the vertical scrollbar easier to hit with a mouse */
        scrollbar-size-vertical: 2;
        
        /* Optional: High contrast colors for the "handle" (thumb) */
        scrollbar-color: $success;
        scrollbar-color-hover: $warning;        
        scrollbar-background: $surface-lighten-1;

        overflow-y: auto;
        overflow-x: hidden;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("c", "new_session", "New Chat")

    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_worker = None
        self.current_model_name = None

    def on_mount(self) -> None:
        try:
            self.settings = AppSettings()
            self.client = genai.Client(api_key=self.settings.gemini_api_key.get_secret_value())
            self.query_one(RichLog).write(f"🟢 App initialized: {datetime.now()}")
            self.query_one("#user_input").focus()

            self.current_worker = None
        except Exception as e:
            self.query_one(RichLog).write(f"🔴 Error loading settings: {e}")

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog(id="chat_log", highlight=True, markup=True, wrap=True)
        with Horizontal(id="input-container"):
            yield Input(placeholder="Ask Gemini something...", id="user_input")
            yield Label("WAITING", id="model-status")
        yield Footer()

    # FIX 1: Removed thread=True. Since we use async/await, we stay on the main loop.
    @work(exclusive=True, exit_on_error=False) 
    async def process_user_request(self, prompt: str):
        log_widget = self.query_one(RichLog)
        status_label = self.query_one('#model-status')

        if self.current_worker is None:
            log_widget.write(Text("🤔 Routing...", style="dim yellow"))
            
            # CHECK: Added 'await'. This executes the coroutine and returns the tuple.
            worker, model_name = await route_logic(self.client, self.settings, prompt) # 🟩
            
            self.current_worker = worker 
            self.current_model_name = model_name 
            log_widget.write(Text(f"🚀 Routed to: {model_name}", style="bold magenta"))
        else:
            worker = self.current_worker
            model_name = self.current_model_name
        
        # Retry Logic Block to handle 503 Unavailable Errors
        max_retries = 5
        base_delay = 1
        result = None

        try:
            # FIX 2: Removed self.call_from_thread wrappers.
            # Since we are not in a thread anymore, we update UI directly.

            for attempt in range(max_retries):
                try:
                    result = await worker.process_task(prompt)
                    break # Success! Exit the loop
                except Exception as e:
                    # CHECK: Is this a 503 (Server Overloaded)?
                    error_str = str(e)
                    if "503" in error_str and attempt < max_retries - 1:
                        # Calculate delay: 1s, 2s, 4s...
                        wait_time = base_delay * (2 ** attempt) 
                        
                        # Visual Feedback
                        dots = "." * (attempt + 1)
                        retry_msg = Text(f"Server overloaded. Retrying in {wait_time}s{dots}", style="dim red")
                        log_widget.write(retry_msg)
                        
                        # CHECK: Using asyncio.sleep, NOT time.sleep!
                        await asyncio.sleep(wait_time) 
                    else:
                        # If it's not a 503, or we ran out of retries, crash for real.
                        raise e
            
            log_widget.write(f"📢 {result}")
            log_widget.write("-" * 50)
            
        except Exception as e:
            log_widget.write(f"Error: {e}")
        
        finally:
            status_label.update("WAITING")
            status_label.styles.background = None

    def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        
        if not user_text:
            return

        self.query_one("#user_input").value = ""
        log_widget = self.query_one(RichLog)
        
        user_msg = Text(f"You: {user_text}", style="bold green")
        log_widget.write(user_msg)
        
        self.process_user_request(user_text)

    def action_new_session(self) -> None:
        log_widget = self.query_one(RichLog)
        
        # 1. Reset the View (The Sponge)
        log_widget.clear()
        
        # 2. Reset the Logic (The School Bell)
        # CHECK: By setting this to None, the next message triggers 'route_logic' again.
        self.current_worker = None 
        self.current_model_name = None
        
        # 2. Focus input field
        self.query_one("#user_input").focus()

        # 4. Provide Feedback
        log_widget.write(Text("🟢 Session Reset. Memory cleared.", style="bold green"))

if __name__ == "__main__":

    # Before starting, we tell the OS to wipe the "Paper Roll" clean.
    # This prevents the "Scroll Up" action from revealing old text.

    app = GeminiTui()
    app.run()