import setuptools
import subprocess
import os

with open("requirements.txt") as f:
    required = f.read().splitlines()

woollylib_version = (
    subprocess.run(["git", "describe", "--tags"], stdout=subprocess.PIPE)
    .stdout.decode("utf-8")
    .strip()
)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="woollylib",
    version=woollylib_version,
    author="Team Woolly",
    author_email="teamwoolly.in@gmail.com",
    description="This is a pytorch based utilities library which will help you for training and visualizing computer vision models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/woolly-of-cv/pytorch-lib",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=required,
)
