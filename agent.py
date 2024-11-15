import asyncio
from call_api import LLMCaller
from pathlib import Path
import json
from typing import List, Dict, Optional
from queue import Queue
from datetime import datetime

class Agent:
    def __init__(self, name: str, responsible_files: List[str]):
        self.name = name
        self.responsible_files = responsible_files
        self.llm_handler = LLMCaller()
        self.message_queue = asyncio.Queue()
        self.running = True
        
        # Log creation
        self.log_action("Created", f"Responsible for files: {responsible_files}")

    def log_action(self, action: str, details: str):
        """Log agent actions to agent_logs.txt"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_path = Path.cwd() / "agent_logs.txt"
        
        log_entry = f"""
[{timestamp}] Agent: {self.name}
Action: {action}
Details: {details}
{'='*50}
"""
        with open(log_path, "a") as f:
            f.write(log_entry)

    async def send_message(self, target_agent: str, message: str):
        """Send message to another agent"""
        message_data = {
            "from": self.name,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        # In a real implementation, this would use a message broker or event system
        self.log_action("Message Sent", f"To: {target_agent}, Content: {message}")
        return message_data

    async def receive_message(self, message_data: Dict):
        """Handle incoming messages"""
        await self.message_queue.put(message_data)
        self.log_action("Message Received", f"From: {message_data['from']}, Content: {message_data['message']}")

    async def process_file(self, file_path: str, changes: Dict):
        """Process changes to a specific file"""
        if file_path not in self.responsible_files:
            self.log_action("Error", f"Not responsible for file: {file_path}")
            return False
            
        try:
            with open(file_path, 'r') as f:
                current_content = f.read()
            
            # Log the proposed changes
            self.log_action("Processing Changes", f"File: {file_path}, Changes: {json.dumps(changes, indent=2)}")
            
            # Implement changes
            if changes.get('replace_all'):
                new_content = changes['content']
            else:
                # Here you'd implement partial changes
                new_content = current_content  # Placeholder
            
            with open(file_path, 'w') as f:
                f.write(new_content)
                
            return True
            
        except Exception as e:
            self.log_action("Error", f"Failed to process {file_path}: {str(e)}")
            return False

    async def think(self) -> Dict:
        """Generate next action using LLM"""
        system_prompt = """You are an AI agent responsible for specific files in a project.
        Output Format:
        {
            "action": "file_change|send_message|analyze",
            "target_file": "path/to/file",
            "changes": {
                "replace_all": boolean,
                "content": "new content",
                "patches": [{"start": int, "end": int, "content": "string"}]
            },
            "message": {
                "target_agent": "agent_name",
                "content": "message content"
            },
            "analysis": {
                "files_analyzed": ["file1", "file2"],
                "findings": "analysis results"
            }
        }"""
        
        context = {
            "responsible_files": self.responsible_files,
            "name": self.name,
            "pending_messages": self.message_queue.qsize()
        }
        
        response = self.llm_handler.generate_response(
            'GROQ',
            json.dumps(context),
            {"system_prompt": system_prompt}
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.log_action("Error", "Failed to parse LLM response")
            return {}

    async def run(self):
        """Main agent loop"""
        while self.running:
            # Process any pending messages
            while not self.message_queue.empty():
                message = await self.message_queue.get()
                self.log_action("Processing Message", json.dumps(message))

            # Think about next action
            action_plan = await self.think()
            
            # Execute action
            if action_plan.get('action') == 'file_change':
                await self.process_file(
                    action_plan['target_file'],
                    action_plan['changes']
                )
            elif action_plan.get('action') == 'send_message':
                await self.send_message(
                    action_plan['message']['target_agent'],
                    action_plan['message']['content']
                )
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.1)

    def stop(self):
        """Stop the agent"""
        self.running = False
        self.log_action("Stopped", "Agent execution terminated") 