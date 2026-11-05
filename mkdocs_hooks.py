"""MkDocs hook that publishes the example scripts as doc pages.

Each example in `examples/` that opens with a `# %% [markdown]` cell is a jupytext
percent script. This hook adds it to the build at the top level, so
`examples/regression_example.py` becomes the page `/regression_example/`. The
`mkdocs-jupyter` plugin then renders it.
"""

from pathlib import Path

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.plugins import event_priority
from mkdocs.structure.files import File, Files

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
MARKDOWN_CELL = "# %% [markdown]"


# Run before mkdocs-jupyter, which only converts the files it already sees.
@event_priority(100)
def on_files(files: Files, config: MkDocsConfig) -> Files:
    """Add every rendered example script to the files of the build."""
    for script in sorted(EXAMPLES_DIR.glob("*.py")):
        if not script.read_text().startswith(MARKDOWN_CELL):
            continue

        files.append(
            File(
                script.name,
                src_dir=str(EXAMPLES_DIR),
                dest_dir=config.site_dir,
                use_directory_urls=config.use_directory_urls,
            )
        )

    return files
