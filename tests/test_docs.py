from pathlib import Path


def test_api_overview_html_exists():
    assert Path("docs/api_overview.html").is_file()


def test_readme_links_api_overview_html():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "[API overview](docs/api_overview.html)" in readme
