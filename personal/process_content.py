from call_api import LLMCaller
import re
from typing import List, Optional, Dict
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
nltk.download('punkt', quiet=True)

class ContentProcessor:
    def __init__(self, input_file: str = 'content.txt', output_file: str = 'output.txt', max_concurrent: int = 5):
        self.llm = LLMCaller()
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.chunks: List[str] = []
        self.responses: Dict[int, str] = {}
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Load system prompt
        try:
            with open('system_prompt.txt', 'r', encoding='utf-8', errors='replace') as file:
                self.system_prompt = file.read().strip()
        except (FileNotFoundError, UnicodeError):
            self.system_prompt = "Process this text segment maintaining its original style and context."
            print("System prompt not found or error reading, using default.")

    def prepare_chunk_with_prompt(self, chunk: str) -> str:
        """Combine system prompt with chunk in a structured way"""
        return f"""System Instructions: {self.system_prompt}

Input Text to Process: {chunk}

Please process the above text according to the system instructions while maintaining context and flow."""

    def read_content(self) -> str:
        try:
            return self.input_file.read_text(encoding='utf-8', errors='replace')
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file {self.input_file} not found")
        except UnicodeError:
            raise UnicodeError(f"Error reading {self.input_file}. Please ensure it's a valid text file.")

    def split_into_chunks(self, sentences_per_chunk: int) -> List[str]:
        content = self.read_content()
        sentences = sent_tokenize(content)
        
        self.chunks = []
        current_chunk = []
        
        for sentence in sentences:
            current_chunk.append(sentence)
            if len(current_chunk) >= sentences_per_chunk:
                chunk_text = ' '.join(current_chunk)
                self.chunks.append(self.prepare_chunk_with_prompt(chunk_text))
                current_chunk = []
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            self.chunks.append(self.prepare_chunk_with_prompt(chunk_text))
            
        return self.chunks

    async def process_chunk(self, chunk: str, index: int, model_name: str, config: Dict) -> None:
        """Process a single chunk with rate limiting"""
        async with self.semaphore:
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    self.llm.generate_response,
                    model_name,
                    chunk,  # chunk already includes system prompt
                    config
                )
                self.responses[index] = response
                print(f"\rProcessed chunk {index + 1}/{len(self.chunks)}", end="", flush=True)
            except Exception as e:
                print(f"\nError processing chunk {index + 1}: {str(e)}")
                self.responses[index] = f"Error processing chunk: {str(e)}"
            finally:
                await asyncio.sleep(0.1)

    async def process_chunks_async(self, model_name: str, custom_config: Optional[dict] = None) -> List[str]:
        """Process chunks with controlled concurrency"""
        if not self.chunks:
            raise ValueError("No chunks to process. Run split_into_chunks first.")
            
        config = {
            'temperature': 0.7,
            'top_p': 0.95,
            'max_tokens': 4096
        }
        if custom_config:
            config.update(custom_config)

        tasks = []
        for i, chunk in enumerate(self.chunks):
            task = asyncio.create_task(self.process_chunk(chunk, i, model_name, config))
            tasks.append(task)

        await asyncio.gather(*tasks)
        return [self.responses[i] for i in range(len(self.chunks))]

    def save_output(self) -> None:
        """Save processed content to output file"""
        if not self.responses:
            raise ValueError("No responses to save. Process chunks first.")
            
        ordered_responses = [self.responses[i] for i in range(len(self.responses))]
        output_text = '\n\n'.join(ordered_responses)  # Removed separator
        self.output_file.write_text(output_text, encoding='utf-8')
        print(f"\nOutput saved to {self.output_file}")

async def main():
    processor = ContentProcessor(max_concurrent=30)
    
    available_models = list(processor.llm.api_keys.keys())
    
    print("Available models:")
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model}")
    
    while True:
        try:
            model_choice = int(input("\nSelect model number: ")) - 1
            if 0 <= model_choice < len(available_models):
                selected_model = available_models[model_choice]
                break
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")
    
    while True:
        try:
            sentences = int(input("\nHow many sentences per chunk? "))
            if sentences > 0:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")

    try:
        chunks = processor.split_into_chunks(sentences)
        print(f"\nSplit content into {len(chunks)} chunks")
        
        await processor.process_chunks_async(selected_model)
        processor.save_output()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 