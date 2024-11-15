import os
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
import anthropic
import google.generativeai as genai
from typing import Dict, Any, Optional, List

class LLMCaller:
    def __init__(self):
        load_dotenv()
        self.api_keys = {
            'GROQ': os.getenv('GROQ_API_KEY'),
            'OpenAI': os.getenv('OPENAI_API_KEY'),
            'Anthropic': os.getenv('ANTHROPIC_API_KEY'),
            'OpenRouter': os.getenv('OPEN_ROUTER_API_KEY'),
            'Google Generative AI': os.getenv('GOOGLE_API_KEY'),
        }
        
        # Updated configurations with temperature=0 and JSON mode
        self.default_configs = {
            'GROQ': {
                'model': "llama3-70b-8192",
                'max_tokens': 4096,
                'temperature': 0,
                'top_p': 1.0,
                'presence_penalty': 0,
                'frequency_penalty': 0,
                'system_prompt': "You are a helpful AI assistant.",
                'response_format': { "type": "json_object" }
            },
            'OpenAI': {
                'model': "gpt-4o-mini",
                'max_tokens': 2048,
                'temperature': 0,
                'top_p': 1.0,
                'presence_penalty': 0,
                'frequency_penalty': 0,
                'system_prompt': "You are a helpful AI assistant.",
                'response_format': { "type": "json_object" },
                'seed': None,
            },
            'Anthropic': {
                'model': "claude-3-sonnet-20240229",
                'max_tokens': 1024,
                'temperature': 0,
                'system_prompt': "You are Claude, a helpful AI assistant.",
                'metadata': None,
                'stop_sequences': None,
                'response_format': { "type": "json_object" }
            },
            'Google Generative AI': {
                'model': "gemini-1.5-flash",
                'temperature': 0,
                'top_p': 1.0,
                'top_k': 40,
                'candidate_count': 1,
                'stop_sequences': None,
                'system_prompt': "You are a helpful AI assistant.",
                'response_format': { "type": "json_object" }
            },
            'OpenRouter': {
                'model': "anthropic/claude-2",
                'max_tokens': 1024,
                'temperature': 0,
                'system_prompt': "You are a helpful AI assistant.",
                'presence_penalty': 0,
                'frequency_penalty': 0,
                'response_format': { "type": "json_object" }
            }
        }

    def generate_response(
        self, 
        api_name: str, 
        user_input: str, 
        config: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate response from specified AI model with custom configurations
        
        Args:
            api_name: Name of the API to use
            user_input: The prompt/question for the model
            config: Optional configuration overrides
            chat_history: Optional list of previous messages
        """
        if not self.api_keys.get(api_name):
            return f"No API key found for {api_name}"

        handlers = {
            'GROQ': self._handle_groq,
            'OpenAI': self._handle_openai,
            'Anthropic': self._handle_anthropic,
            'Google Generative AI': self._handle_google,
            'OpenRouter': self._handle_openrouter
        }

        handler = handlers.get(api_name)
        if not handler:
            return "Invalid API name"

        final_config = self.default_configs[api_name].copy()
        if config:
            final_config.update(config)

        try:
            return handler(user_input, final_config, chat_history)
        except Exception as err:
            return f"An error occurred with {api_name}: {str(err)}"

    def _prepare_messages(self, user_input: str, config: Dict[str, Any], chat_history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        messages = []
        if config.get('system_prompt'):
            messages.append({"role": "system", "content": config['system_prompt']})
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": user_input})
        return messages

    def _handle_groq(self, user_input: str, config: Dict[str, Any], chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        client = Groq(api_key=self.api_keys['GROQ'])
        messages = self._prepare_messages(user_input, config, chat_history)
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=config['model'],
            max_tokens=config.get('max_tokens'),
            temperature=config.get('temperature'),
            top_p=config.get('top_p'),
            presence_penalty=config.get('presence_penalty'),
            frequency_penalty=config.get('frequency_penalty')
        )
        return chat_completion.choices[0].message.content

    def _handle_openai(self, user_input: str, config: Dict[str, Any], chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        client = OpenAI(api_key=self.api_keys['OpenAI'])
        messages = self._prepare_messages(user_input, config, chat_history)
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=config['model'],
            max_tokens=config.get('max_tokens'),
            temperature=config.get('temperature'),
            top_p=config.get('top_p'),
            presence_penalty=config.get('presence_penalty'),
            frequency_penalty=config.get('frequency_penalty'),
            response_format=config.get('response_format'),
            seed=config.get('seed')
        )
        return chat_completion.choices[0].message.content

    def _handle_anthropic(self, user_input: str, config: Dict[str, Any], chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        client = anthropic.Anthropic(api_key=self.api_keys['Anthropic'])
        messages = self._prepare_messages(user_input, config, chat_history)
        message = client.messages.create(
            model=config['model'],
            max_tokens=config.get('max_tokens'),
            messages=messages,
            temperature=config.get('temperature'),
            system=config.get('system_prompt'),
            metadata=config.get('metadata'),
            stop_sequences=config.get('stop_sequences')
        )
        return message.content

    def _handle_google(self, user_input: str, config: Dict[str, Any], chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        genai.configure(api_key=self.api_keys['Google Generative AI'])
        model = genai.GenerativeModel(
            config['model'],
            generation_config={
                'temperature': config.get('temperature'),
                'top_p': config.get('top_p'),
                'top_k': config.get('top_k'),
                'candidate_count': config.get('candidate_count'),
                'stop_sequences': config.get('stop_sequences')
            }
        )
        
        chat = model.start_chat(history=[])
        if config.get('system_prompt'):
            chat.send_message(config['system_prompt'])
        if chat_history:
            for msg in chat_history:
                chat.send_message(msg['content'])
                
        response = chat.send_message(user_input)
        return response.text

    def _handle_openrouter(self, user_input: str, config: Dict[str, Any], chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_keys['OpenRouter'],
            default_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "API Test"
            }
        )
        messages = self._prepare_messages(user_input, config, chat_history)
        response = client.chat.completions.create(
            model=config['model'],
            messages=messages,
            max_tokens=config.get('max_tokens'),
            temperature=config.get('temperature'),
            presence_penalty=config.get('presence_penalty'),
            frequency_penalty=config.get('frequency_penalty')
        )
        return response.choices[0].message.content

def main():
    """Main function to run the API interaction loop."""
    handler = LLMCaller()
    while True:
        print("\nSelect an API to interact with:")
        for i, api_name in enumerate(handler.api_keys.keys(), 1):
            print(f"{i}. {api_name}")
        print("0. Exit")

        try:
            choice = int(input("Enter the number of your choice: "))
            if choice == 0:
                print("Exiting the program.")
                break
            elif 1 <= choice <= len(handler.api_keys):
                api_name = list(handler.api_keys.keys())[choice - 1]
                user_input = input(f"Enter input for {api_name}: ")
                result = handler.generate_response(api_name, user_input)
                print(f"Response from {api_name}: {result}")
            else:
                print("Invalid choice. Please select a valid option.")
        except ValueError:
            print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    main()
