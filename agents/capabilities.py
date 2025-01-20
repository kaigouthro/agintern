import subprocess
import sys
from abc import ABC, abstractmethod
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from typing import Tuple, Dict, Any, List
import tempfile
import os
import difflib
import git


class Ability(ABC):
    @abstractmethod
    def execute(self, agent, **kwargs):
        pass


class CodeExecutionAbility(Ability):
    CODE_BLOCK_START = "```python"
    CODE_BLOCK_END = "```"
    MAX_RESPONSE_LENGTH = 8000
    RESPONSE_PREVIEW_LENGTH = 600

    def execute(self, agent, **kwargs):
        text_with_code = kwargs.get("text_with_code")
        if not text_with_code:
            return "No code provided for execution."

        code_to_execute = self._extract_code(text_with_code)
        stdout, stderr = self._execute_code(code_to_execute)
        return self._format_response(stdout, stderr)

    def _extract_code(self, text: str) -> str:
        return text.split(self.CODE_BLOCK_START)[1].split(self.CODE_BLOCK_END)[0]

    def _execute_code(self, code: str) -> Tuple[str, str]:
        exec_globals: Dict[str, Any] = {}
        with StringIO() as stdout_buffer, StringIO() as stderr_buffer:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, exec_globals)
            return stdout_buffer.getvalue(), stderr_buffer.getvalue()

    def _format_response(self, stdout: str, stderr: str) -> str:
        response = "Executed Python Code Successfully."
        if stdout:
            response += f"Standard Output:{stdout}"
        if stderr:
            response += f"Standard Error:{stderr}"

        if len(response) > self.MAX_RESPONSE_LENGTH:
            preview_end = self.RESPONSE_PREVIEW_LENGTH
            return f"{response[:preview_end]}...{response[-(self.MAX_RESPONSE_LENGTH - preview_end) :]}"
        return response


class FileSystemAbility(Ability):
    def __init__(self):
        self.base_dir = tempfile.mkdtemp()

    def execute(self, agent, **kwargs):
        filename = kwargs.get("filename") or None
        dirpath = kwargs.get("path") or None

        action = kwargs.get("action")
        if action == "write_file":
            if not filename:
                return "No filename specified."
            if "content" not in kwargs:
                return "No content specified."
            if content := kwargs.get("content") or None:
                return self._write_file(filename, content)
            else:
                return "No content specified."
        elif action == "read_file":
            return self._read_file(filename) if filename else "No filename specified."
        elif action == "list_contents":
            return self._list_contents(dirpath or ".")
        elif action == "make_dir":
            return self._make_dir(dirpath) if dirpath else "No path specified."
        elif action == "remove_dir":
            return self._remove_dir(dirpath) if dirpath else "No path specified."
        else:
            return "Invalid file system action specified."

    def _write_file(self, filename: str, content: str) -> str:
        filepath = os.path.join(self.base_dir, filename)
        try:
            with open(filepath, "w") as f:
                f.write(content)
            return f"File '{filename}' written successfully to '{filepath}'."
        except Exception as e:
            return f"Error writing to file '{filename}': {e}"

    def _read_file(self, filename: str) -> str:
        filepath = os.path.join(self.base_dir, filename)
        try:
            with open(filepath, "r") as f:
                content = f.read()
            return f"Content of '{filename}':{content}"
        except Exception as e:
            return f"Error reading file '{filename}': {e}"

    def _list_contents(self, path: str) -> str:
        """List the contents of a directory."""
        fullpath = os.path.join(self.base_dir, path)
        try:
            contents = os.listdir(fullpath)
            return f"Contents of '{path}': {', '.join(contents)}"
        except Exception as e:
            return f"Error listing contents of '{path}': {e}"

    def _make_dir(self, dirpath: str) -> str:
        """Create a new directory."""
        fullpath = os.path.join(self.base_dir, dirpath)
        try:
            os.makedirs(fullpath, exist_ok=True)
            return f"Directory '{dirpath}' created successfully at '{fullpath}'."
        except Exception as e:
            return f"Error creating directory '{dirpath}': {e}"

    def _remove_dir(self, dirpath: str) -> str:
        """Remove a directory."""
        fullpath = os.path.join(self.base_dir, dirpath)
        try:
            os.rmdir(fullpath)
            return f"Directory '{dirpath}' removed successfully."
        except Exception as e:
            return f"Error removing directory '{dirpath}': {e}"


class DiffAbility(Ability):
    def execute(self, agent, **kwargs):
        original_content = kwargs.get("original_content", "")
        modified_content = kwargs.get("modified_content", "")
        return self._calculate_diff(original_content, modified_content)

    def _calculate_diff(self, original_content: str, modified_content: str) -> str:
        diff = difflib.unified_diff(original_content.splitlines(keepends=True), modified_content.splitlines(keepends=True), fromfile="original", tofile="modified")
        return "".join(diff)


class GitAbility(Ability):
    def __init__(self):
        self.repo_path = tempfile.mkdtemp()
        self.repo = git.Repo.init(self.repo_path)

    def execute(self, agent, **kwargs):
        action = kwargs.get("action")
        # Add more git actions as needed, create checks for kwargs

        if action == "add":
            if filepath := kwargs.get("filepath") or None:
                return self._git_add(filepath)
            else:
                return "Filepath is required for 'add' action."

        elif action == "commit":
            message = kwargs.get("message") or None
            return self._git_commit(message) if message else "No message was Specified"
        elif action == "branch":
            if branch_name := kwargs.get("branch_name") or None:
                return self._git_branch(branch_name)
            else:
                return "No branch_name was Specified"
        elif action == "checkout":
            if branch_name := kwargs.get("branch_name") or None:
                return self._git_checkout(branch_name)
            else:
                return "No branch_name was Specified"
        else:
            return "Invalid git action specified."

    def _git_add(self, filepath: str) -> str:
        try:
            self.repo.index.add([filepath])
            return f"Successfully added '{filepath}' to the staging area."
        except Exception as e:
            return f"Error adding '{filepath}' to the staging area: {e}"

    def _git_commit(self, message: str) -> str:
        try:
            self.repo.index.commit(message)
            return f"Successfully committed changes with message: '{message}'."
        except Exception as e:
            return f"Error committing changes: {e}"

    def _git_branch(self, branch_name: str) -> str:
        try:
            self.repo.git.branch(branch_name)
            return f"Successfully created branch '{branch_name}'."
        except Exception as e:
            return f"Error creating branch '{branch_name}': {e}"

    def _git_checkout(self, branch_name: str) -> str:
        try:
            self.repo.git.checkout(branch_name)
            return f"Successfully checked out branch '{branch_name}'."
        except Exception as e:
            return f"Error checking out branch '{branch_name}': {e}"


class PythonFunctionAbility(Ability):
    def execute(self, agent, **kwargs):
        func = kwargs.get("function")
        args = kwargs.get("args", [])
        kwargs_inner = kwargs.get("kwargs", {})

        if not callable(func):
            return "Provided function is not callable."

        try:
            result = func(*args, **kwargs_inner)
            return f"Function executed successfully. Result: {result}"
        except Exception as e:
            return f"Error executing function: {e}"
