from call_api import LLMCaller
from typing import Optional, Dict, Any, List

class Chat:
    def __init__(self):
        self.llm = LLMCaller()
        self.chat_history: List[Dict[str, str]] = []
        
        # Load system prompt from file
        try:
            with open('system_prompt.txt', 'r') as file:
                system_prompt = file.read()
        except FileNotFoundError:
            system_prompt = "You are a helpful AI assistant."
            print("System prompt not found, using default.")
        
        self.default_config = {
            'temperature': 0.7,
            'top_p': 0.8,
            'top_k': 40,
            'system_prompt': system_prompt,
            'candidate_count': 1,
        }

    def ask(self, prompt: str, custom_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Send a prompt to the model and get response
        
        Args:
            prompt: User's question or prompt
            custom_config: Optional configuration overrides
        """
        config = self.default_config.copy()
        if custom_config:
            config.update(custom_config)

        response = self.llm.generate_response(
            'Google Generative AI',
            prompt,
            config=config,
            chat_history=self.chat_history
        )

        # Update chat history
        self.chat_history.append({"role": "user", "content": prompt})
        self.chat_history.append({"role": "assistant", "content": response})

        return response

    def reset_chat(self):
        """Clear chat history"""
        self.chat_history = []

if __name__ == "__main__":
    # Initialize chat
    chat = Chat()

    # Example with custom configuration
    creative_config = {
        'temperature': 0.9,
        'top_p': 0.95,
    }
    
    print("\nChat - Type 'quit' to exit, 'reset' to clear history, 'config' to see current settings")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'reset':
            chat.reset_chat()
            print("Chat history cleared!")
            continue
        elif user_input.lower() == 'config':
            print("\nCurrent configuration:")
            for key, value in chat.default_config.items():
                print(f"{key}: {value}")
            continue
            
        print("\nAssistant:", end=" ")
        response = chat.ask(user_input, custom_config=creative_config)
        print(response) 