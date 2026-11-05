"""Checks that the project is set up correctly.

These tests contain no engineering. They only prove that pytest runs and that
Python can find our code in src/pipeline.
"""

from pathlib import Path

import pipeline


def test_pipeline_package_is_loaded_from_src() -> None:
    """The imported package should be our src/pipeline folder."""
    package_folder = Path(pipeline.__file__).parent

    assert package_folder.name == "pipeline"
    assert package_folder.parent.name == "src"
