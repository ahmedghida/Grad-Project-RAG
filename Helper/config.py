import os
from pydantic_settings import BaseSettings , SettingsConfigDict  


class Settings(BaseSettings):

    Llama_key :str
    gemini_key:str
    Chunk_size:int
    Chunk_overlay:int


    class Config:

        env_file='.env'




def get_config():

    return Settings()