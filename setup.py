import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="swarmnet",  # Replace with your own username
    version="0.1.0",
    author="Siyu Zhou",
    author_email="siyu.zhou@asu.edu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/siyuzhou/swarmnet",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
