#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES_DIR = ROOT / "prompts" / "modes"
ACTIVE_DIR = ROOT / "prompts" / "active"


def list_modes() -> list[str]:
    if not MODES_DIR.exists():
        return []
    return sorted([p.name for p in MODES_DIR.iterdir() if p.is_dir()])


def set_mode(mode: str) -> None:
    mode_dir = MODES_DIR / mode
    if not mode_dir.exists():
        raise SystemExit(f"Mode not found: {mode}")
    for name in ("baskan.md", "arastirmaci.md"):
        src = mode_dir / name
        if not src.exists():
            raise SystemExit(f"Missing {name} in {mode_dir}")
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("baskan.md", "arastirmaci.md"):
        shutil.copy2(mode_dir / name, ACTIVE_DIR / name)
    (ACTIVE_DIR / "mode.txt").write_text(f"{mode}\n", encoding="utf-8")
    print(f"Active prompt mode set to: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Switch active prompt mode.")
    parser.add_argument("mode", nargs="?", help="Mode name to activate")
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

    if not args.mode:
        parser.error("mode is required unless --list is used")

    set_mode(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
