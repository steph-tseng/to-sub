import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="tosync",
    version="1.0.1",
    description="Transcript to subtitles",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/steph-tseng/to-sub",
    author="Stephanie Tseng",
    author_email="lordnyx1031@gmail.com",
    classifiers=["Programming Language :: Python :: 3.9"],
    packages=["tosync"],
    include_package_data=True,
    install_requires=["feedparser", "html2text"],
    entry_points={
        "console_scripts": [
            "tosync=reader.__main__:main",
        ]
    },
)