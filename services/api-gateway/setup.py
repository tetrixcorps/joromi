from setuptools import setup, find_packages

setup(
    name="ml-services-api",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-multipart>=0.0.5",
        "aiohttp>=3.8.0",
        "pydantic>=1.8.0",
        "python-jose[cryptography]>=3.3.0",
    ],
) 