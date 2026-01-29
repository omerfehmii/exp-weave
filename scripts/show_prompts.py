#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_DIR = ROOT / "prompts" / "active"
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from set_prompt_mode import list_modes, set_mode  # noqa: E402


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").rstrip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Show active prompts.")
    parser.add_argument("mode", nargs="?", help="Switch to mode before showing prompts")
    parser.add_argument("--list", action="store_true", help="List available modes")
    args = parser.parse_args()

    if args.list:
        modes = list_modes()
        if not modes:
            print("No modes found.")
            return 0
        print("Available modes:")
        for mode in modes:
            print(f"- {mode}")
        return 0

    if args.mode:
        set_mode(args.mode)

    mode_path = ACTIVE_DIR / "mode.txt"
    mode = read_text(mode_path) or "unknown"
    baskan = read_text(ACTIVE_DIR / "baskan.md")
    arastirmaci = read_text(ACTIVE_DIR / "arastirmaci.md")

    print(f"ACTIVE MODE: {mode}")
    print("\n=== BASKAN PROMPT ===\n")
    print(baskan)
    print("\n=== ARASTIRMACI PROMPT ===\n")
    print(arastirmaci)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
