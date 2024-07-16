# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from typing import Any, Dict
import sys
import os

from sphinxawesome_theme.postprocess import Icons

sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "lightning-ir"
copyright = "2024, Webis"
author = "Webis"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx_toolbox.collapse",
    "myst_parser",
]

# autodoc_mock_imports = ['torch', 'transformers', 'lightning']


todo_include_todos = True
python_display_short_literal_types = True
python_use_unqualified_type_names = True
viewcode_line_numbers = True
autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinxawesome_theme"
html_title = "lightning-ir"
html_logo = "./logo.png"
html_logo = "./logo.png"
language = "en"

html_theme_options: Dict[str, Any] = {
    "source_repository": "https://github.com/webis-de/lightning-ir",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
    # Sphinx Awesome Configurations:
    "logo_light": "logo.png",
    "logo_dark": "logo.png",
    "show_breadcrumbs": True,
    "show_prev_next": True,
    "show_scrolltop": True,
}

html_permalinks_icon = Icons.permalinks_icon


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
