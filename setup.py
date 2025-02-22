from setuptools import setup, find_packages

setup(
    name="african_translation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'datasets>=2.12.0',
        'sacrebleu>=2.3.1',
        'nltk>=3.8.1',
        'numpy>=1.24.0',
        'pandas>=2.0.0'
    ]
) 