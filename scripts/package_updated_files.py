import argparse
import os
from pathlib import Path
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "dist" / "aisingers_updated.zip"
LIGHTWEIGHT_SKIPS = {
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
FULL_SKIPS = {".git", "__pycache__", "dist", ".pytest_cache"}


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
            "Создает архив с обновленными файлами репозитория. По умолчанию сохраняет "
            "полную копию кода (без .git и кэшей), а флаги позволяют собрать лёгкую "
            "версию без тяжёлых директорий."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["full", "light"],
        default="full",
        help=(
            "Режим архивации: full — полная копия кода без .git и кэшей; "
            "light — пропустить модели, выходные данные и изображения."
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
        help="Добавить rvc_models и mdxnet_models даже в лёгком режиме.",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Синоним режима full (для обратной совместимости).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # В полном режиме оставляем только служебные исключения, в лёгком — пропускаем
    # тяжёлые каталоги; include-all перекрывает выбор режима для совместимости.
    if args.include_all:
        args.mode = "full"

    skip_dirs = set(FULL_SKIPS if args.mode == "full" else LIGHTWEIGHT_SKIPS)

    if args.mode == "light" and args.include_models:
        skip_dirs.discard("rvc_models")
        skip_dirs.discard("mdxnet_models")

    archive_path = create_archive(args.output, skip_dirs)
    print(f"Архив сохранен: {archive_path}")


if __name__ == "__main__":
    main()
