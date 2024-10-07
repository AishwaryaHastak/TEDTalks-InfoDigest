import setuptools

__version__ = "1.0.0"

REPO_NAME = "ted_rag"
AUTHOR_USER_NAME = "AishwaryaHastak"
SRC_REPO = "src"
AUTHOR_EMAIL = "h4hastak@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="RAG Pipeline Implementation", 
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)