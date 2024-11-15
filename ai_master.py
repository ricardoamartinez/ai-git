import threading
from queue import Queue
import time
from call_api import LLMCaller

user_inputs = []
input_queue = Queue()
running = True
llm_handler = LLMCaller()

def input_thread():
    global running
    while running:
        user_input = input("Enter something (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            running = False
        else:
            input_queue.put(user_input)

def process_llm_input(text):
    # Using Groq as default, you can modify this
    response = llm_handler.generate_response('GROQ', text)
    print(f"\nAI Response: {response}\n")

def main():
    # Start input thread
    input_handler = threading.Thread(target=input_thread, daemon=True)
    input_handler.start()
    
    # Main thread processes the queue
    while running:
        if not input_queue.empty():
            item = input_queue.get()
            user_inputs.append(item)
            print(f"Processing: {item}")
            process_llm_input(item)
        time.sleep(0.1)

if __name__ == "__main__":
    main()
