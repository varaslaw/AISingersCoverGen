import argparse
import os
from pathlib import Path
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "dist" / "aisingers_updated.zip"
DEFAULT_SKIPS = {
    ".git",
    "__pycache__",
    "mdxnet_models",
    "rvc_models",
    "song_output",
    "uploads",
    "images",
    "dist",
    ".pytest_cache",
}


def create_archive(output_path: Path, skip_dirs: set[str]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for root, dirs, files in os.walk(REPO_ROOT):
            root_path = Path(root)
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]

            for file_name in files:
                if file_name.startswith("."):
                    continue

                file_path = root_path / file_name
                if any(part in skip_dirs for part in file_path.parts):
                    continue

                archive_path = file_path.relative_to(REPO_ROOT)
                archive.write(file_path, arcname=archive_path)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Создает архив с обновленными файлами репозитория. По умолчанию пропускает "
            "тяжелые модели и выходные данные, но может включить их по флагам."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Путь для сохранения архива (по умолчанию {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--include-models",
        action="store_true",
        help="Включить каталоги моделей rvc_models и mdxnet_models в архив.",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Собрать архив без пропуска модельных и выходных директорий.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    skip_dirs = set(DEFAULT_SKIPS)

    if args.include_all:
        skip_dirs = {".git", "__pycache__", "dist", ".pytest_cache"}
    elif args.include_models:
        skip_dirs.discard("rvc_models")
        skip_dirs.discard("mdxnet_models")

    archive_path = create_archive(args.output, skip_dirs)
    print(f"Архив сохранен: {archive_path}")


if __name__ == "__main__":
    main()
