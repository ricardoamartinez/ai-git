from call_api import LLMCaller
import json
import os
from pathlib import Path
import re
import subprocess
from datetime import datetime

llm_handler = LLMCaller()

def log_git_action(project_path: Path, action: str, output: str):
    """Log git actions to git_logs.txt"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = Path.cwd() / "git_logs.txt"
    
    log_entry = f"""
[{timestamp}] {project_path.name}
Action: {action}
Output: {output}
{'='*50}
"""
    with open(log_path, "a") as f:
        f.write(log_entry)

def setup_git(project_path: Path, project_desc: str):
    """Initialize git repository and create first commit"""
    try:
        # Initialize git repository
        init_output = subprocess.run(
            ["git", "init"],
            cwd=project_path,
            capture_output=True,
            text=True
        )
        log_git_action(project_path, "git init", init_output.stdout + init_output.stderr)

        # Create .gitignore
        gitignore_content = """
__pycache__/
*.py[cod]
*$py.class
.env
.venv
env/
venv/
.idea/
.vscode/
"""
        with open(project_path / ".gitignore", "w") as f:
            f.write(gitignore_content)

        # Add all files
        add_output = subprocess.run(
            ["git", "add", "."],
            cwd=project_path,
            capture_output=True,
            text=True
        )
        log_git_action(project_path, "git add .", add_output.stdout + add_output.stderr)

        # Create initial commit
        commit_msg = f"Initial commit: {project_desc}"
        commit_output = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=project_path,
            capture_output=True,
            text=True
        )
        log_git_action(project_path, f"git commit -m '{commit_msg}'", 
                      commit_output.stdout + commit_output.stderr)

        print(f"\nGit repository initialized at: {project_path}")
        print("Check git_logs.txt for detailed git operations")
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Git operation failed: {str(e)}"
        log_git_action(project_path, "ERROR", error_msg)
        print(f"\nError: {error_msg}")

def get_system_prompt():
    with open("python_list_system_prompt.txt", "r") as f:
        return f.read()

def clean_json_response(response: str) -> str:
    """Remove markdown formatting and extract just the JSON"""
    # Remove markdown code blocks and any text before/after
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        return json_match.group()
    return response

def create_file_structure(structure, base_path):
    """Recursively create the file structure from JSON"""
    path = Path(base_path) / structure['name']
    
    if structure['type'] == 'directory':
        path.mkdir(exist_ok=True)
        if 'children' in structure:
            for child in structure['children']:
                create_file_structure(child, path)
    else:  # file
        with open(path, 'w') as f:
            f.write(structure.get('content', ''))

def process_llm_input(text):
    # Save the project description
    with open("project_prompt.txt", "w") as f:
        f.write(text)
    
    # Get system prompt and configure
    system_prompt = get_system_prompt()
    config = {
        'system_prompt': system_prompt
    }
    
    # Make LLM call
    response = llm_handler.generate_response('GROQ', text, config)
    
    try:
        # Clean the response
        cleaned_response = clean_json_response(response)
        
        # Parse the response to ensure it's valid JSON
        json_response = json.loads(cleaned_response)
        
        # Save the response as formatted JSON
        with open("project_structure.json", "w") as f:
            json.dump(json_response, f, indent=4)
            
        print(f"\nProject structure has been generated and saved to project_structure.json")
        
        # Create project directory
        project_name = json_response['name']
        project_path = Path.cwd() / 'projects' / project_name
        
        # Create the projects directory if it doesn't exist
        Path('projects').mkdir(exist_ok=True)
        
        # Create the project structure
        create_file_structure(json_response, project_path.parent)
        print(f"\nProject has been created at: {project_path}")
        
        # Initialize git repository
        setup_git(project_path, text)
        
    except json.JSONDecodeError as e:
        print(f"\nError: LLM response was not valid JSON: {e}")
        print("Raw response:", response)
    except Exception as e:
        print(f"\nError creating project structure: {e}")

def main():
    print("\nWhat project would you like to create? (Describe your project)")
    project_desc = input("> ")
    
    print(f"\nProcessing project request: {project_desc}")
    process_llm_input(project_desc)

if __name__ == "__main__":
    main() 