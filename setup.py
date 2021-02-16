import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="armisc",
    version="1.0.0",
    author="Asif Rahman",
    author_email="asiftr@gmail.com",
    description=("Contains many functions useful for data analysis, high-level graphics, utility operations."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/asifr/armisc",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)