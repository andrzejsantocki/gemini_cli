from abc import ABC, abstractmethod
from datetime import datetime
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr, Field
import asyncio
import click
from typing import Any 
from google import genai 

class AppSettings(BaseSettings):
    # TODO: Define API key and model names using Pydantic V2

    gemini_api_key:SecretStr = Field(..., alias = 'GEMINI_API_KEY') # alias is requred for BaseSettings to locate &validate var
    heavy_model: str = Field("gemini-2.5-flash")
    light_model: str = Field("gemini-2.5-flash")
    
    model_config = SettingsConfigDict(
        env_file = '.env',
        extra = 'ignore'

    )
    pass

class LLMInterface(ABC):
    # TODO: Define abstract async method for generation
    
    @abstractmethod
    async def process_task(self, prompt_text:str) -> str:
        pass
    

class FlashImplementation(LLMInterface):
    # TODO: Implement fast model logic (Flash 1.5)

    def __init__(self, external_client: Any, model_name: str):
        self._external_client, self._model_name = external_client, model_name 
    
    async def process_task(self, prompt_text:str):
        response =  await self._external_client.aio.models.generate_content(
            model = self._model_name, contents = prompt_text
        )
        return response.text

class HighReasoningImplementation(LLMInterface):
    # TODO: Implement complex model logic (Gemini 3)
   
    def __init__(self, external_client: Any, model_name: str):
        self._external_client, self._model_name = external_client, model_name 
    
    async def process_task(self, prompt_text:str):
        response = await  self._external_client.aio.models.generate_content(
            model = self._model_name, contents = prompt_text
        )
        return response.text


def route_logic(client: genai.Client, settings: AppSettings, user_prompt:str) -> tuple[LLMInterface, str]: #tuple is native unpackable in python
    # TODO: Implement complexity heuristic to return the correct implementation
    # This app actually decide which LLM to istantiate

    if len(user_prompt)> 20:
        return HighReasoningImplementation(client, settings.heavy_model), settings.heavy_model
    else:
        return FlashImplementation(client, settings.light_model), settings.light_model


@click.command()
@click.argument('prompt')
def cli(prompt: str):
    # TODO: Bridge synchronous click to async run_app
    click.echo(f"🟢 Cli is running: {datetime.now()}")
    settings = AppSettings()
    client = genai.Client(api_key = settings.gemini_api_key.get_secret_value())
    worker, model_name = route_logic(client, settings, prompt)
    click.echo(f"🤖 Selected model is {model_name}\n")
    result = asyncio.run(worker.process_task(prompt))
    click.echo(f"📢 {result}")


if __name__ == "__main__":
    cli()