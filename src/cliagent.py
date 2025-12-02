"""
CLI-based coding agent that wraps provider commands (e.g., gemini, codex, qwen).

Usage:
    agent = CLICodingAgent(
        prompt="Write a Python script that prints 'Hello, World!'",
        provider="codex",
        model_name="",  # Optional, used only for Gemini
        provider_cmd=None,  # Optional override command list
        cwd=None  # Optional working directory
    )
    exit_code = agent.forward()
"""

import sys
import subprocess
from pathlib import Path
from shutil import which
from typing import Dict, List, Tuple, Optional

# Provider configuration table
PROVIDERS: Dict[str, Tuple[str, List[str]]] = {
    "gemini": ("arg",  ["gemini", "-y", "-p"]),
    "codex":  ("arg",  ["codex", "exec", "--full-auto", "-s", "workspace-write"]),
    "qwen":   ("stdin",["qwen", "-y", "-p"]),
}


class CLICodingAgent:
    """
    A CLI-based coding agent that wraps provider commands (e.g., gemini, codex, qwen).
    """

    def __init__(
        self,
        prompt: str,
        provider: str = "codex",
        model_name: str = "",
        provider_cmd: Optional[List[str]] = None,
        cwd: Optional[Path] = None,
    ):
        """
        Initialize the CLI agent.

        Args:
            prompt: Prompt text to send to the provider.
            provider: Name of provider (default: "codex").
            model_name: Optional model name (used only for Gemini).
            provider_cmd: Optional override command list (forces stdin mode).
            cwd: Optional working directory.
        """
        self.prompt = prompt
        self.provider = provider
        self.model_name = model_name
        self.provider_cmd = provider_cmd
        self.cwd = cwd

    def _build_command(self) -> Tuple[List[str], str]:
        """
        Build the provider command and determine input mode.
        Returns (command, mode).
        """
        if self.provider_cmd:
            cmd = list(self.provider_cmd)
            mode = "stdin"
        else:
            if self.provider not in PROVIDERS:
                print(f"Unknown provider '{self.provider}'", file=sys.stderr)
                return [], "error"

            mode, base_cmd = PROVIDERS[self.provider]
            cmd = list(base_cmd)

            # Best-effort model flag injection for Gemini only
            if self.model_name and self.provider == "gemini":
                cmd[1:1] = ["-m", self.model_name]

            # Check that executable exists
            exe = cmd[0]
            if which(exe) is None:
                print(f"Provider CLI '{exe}' not found on PATH", file=sys.stderr)
                return [], "error"

        return cmd, mode

    def forward(self, timeout: Optional[float] = None) -> int:
        """
        Execute the CLI command with the prompt.
        Returns the process exit code (int).
        """
        cmd, mode = self._build_command()
        print(f"Executing: {' '.join(cmd)}", file=sys.stderr)
        if mode == "error":
            return 2

        try:
            if mode == "stdin":
                proc = subprocess.run(
                    cmd,
                    input=self.prompt.encode("utf-8"),
                    cwd=str(self.cwd) if self.cwd else None,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    timeout=timeout,
                )
            else:  # mode == "arg"
                proc = subprocess.run(
                    cmd + [self.prompt],
                    cwd=str(self.cwd) if self.cwd else None,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    timeout=timeout,
                )
            return proc.returncode
        except subprocess.TimeoutExpired as exc:
            print(
                f"Provider command exceeded {timeout} seconds and was terminated.",
                file=sys.stderr,
            )
            raise TimeoutError(
                f"Provider command exceeded {timeout} seconds and was terminated"
            ) from exc


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CLI Coding Agent")
    parser.add_argument("--prompt", help="The prompt md file to send to the agent")
    parser.add_argument("--provider", default="codex", help="The provider to use (default: codex)")
    parser.add_argument("--model", default="", help="The model name (optional)")
    parser.add_argument("--cwd", default=None, help="The working directory (optional)")
    
    args = parser.parse_args()

    try:
        with open(args.prompt, "r") as f:
            prompt = f.read().strip()
    except FileNotFoundError:
        print(f"Error: Prompt file '{args.prompt}' not found.", file=sys.stderr)
        sys.exit(1)

    cwd_path = Path(args.cwd) if args.cwd else None

    agent = CLICodingAgent(
        prompt=args.prompt,
        provider=args.provider,
        model_name=args.model,
        cwd=cwd_path
    )
    
    exit_code = agent.forward()
    sys.exit(exit_code)
