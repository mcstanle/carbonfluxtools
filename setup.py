import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="carbonfluxtools-mcstanle",
    version="0.0.1",
    author="Mike Stanley",
    author_email="mcstanle@andrew.cmu.edu",
    description="Computational support for carbon flux tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mcstanle/carbonfluxtools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
