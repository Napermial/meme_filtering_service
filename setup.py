from setuptools import setup

setup(
    name="meme_data_enricher",
    version="0.1.0",
    py_modules=["meme_data_enricher"],
    install_requires=[
        "Click",
    ],
    entry_points={
        "console_scripts": [
            "meme-data-enricher = cli:meme_data_enricher",
        ],
    },
)
