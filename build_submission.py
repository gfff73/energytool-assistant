"""Build the platform ZIP from an explicit list of source files."""

import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FILES = ("solution.py", "search_kb.py", "russian_stemmer.py", "ask_kb.py", "requirements.txt")


def main():
    for name in FILES:
        text = (ROOT / name).read_text(encoding="utf-8")
        if name.endswith(".py"):
            compile(text, name, "exec")

    destination = ROOT / "dist/energytool-submission.zip"
    destination.parent.mkdir(exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in FILES:
                archive.write(ROOT / name, arcname=name)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None or archive.namelist() != list(FILES):
                raise ValueError("Не удалось проверить целостность архива.")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Архив готов: {destination}")
    print(f"Размер: {destination.stat().st_size} байт")


if __name__ == "__main__":
    main()
