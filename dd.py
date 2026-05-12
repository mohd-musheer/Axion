import os

# Root folder to scan
ROOT_DIR = "."

# Output file
OUTPUT_FILE = "data.txt"

# Extensions to include
TEXT_EXTENSIONS = {
    ".py", ".cpp", ".hpp", ".h", ".c",
    ".txt", ".md", ".json", ".yaml", ".yml",
    ".sh", ".bat", ".ps1", ".js", ".ts",
    ".html", ".css", ".xml", ".toml",
    ".ini", ".cfg", ".sql", ".proto",
    ".dockerfile",".yml",".ipynb",".jsonl",".json",""
}

# Folders to ignore
IGNORE_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "build",
    "dist"
}


def is_text_file(filename):
    ext = os.path.splitext(filename)[1].lower()

    if filename.lower() == "dockerfile":
        return True

    return ext in TEXT_EXTENSIONS


with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    for root, dirs, files in os.walk(ROOT_DIR):

        # Remove ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:

            if file == OUTPUT_FILE:
                continue

            if not is_text_file(file):
                continue

            filepath = os.path.join(root, file)

            try:
                relative_path = os.path.relpath(filepath, ROOT_DIR)

                outfile.write("\n")
                outfile.write("=" * 80 + "\n")
                outfile.write(f"FILE: {relative_path}\n")
                outfile.write("=" * 80 + "\n\n")

                with open(filepath, "r", encoding="utf-8", errors="ignore") as infile:
                    content = infile.read()

                outfile.write(content)
                outfile.write("\n\n")

            except Exception as e:
                print(f"Skipped {filepath}: {e}")

print(f"\nDone. All text files saved into {OUTPUT_FILE}")