import codecs

from setuptools import find_packages
from setuptools import setup


requirements = []

with open("./test-requirements.txt") as test_reqs_txt:
    test_requirements = [line for line in test_reqs_txt]


long_description = ""
with codecs.open("./README.md", encoding="utf-8") as readme_md:
    long_description = readme_md.read()

setup(
    name="screendl",
    use_scm_version={"write_to": "screendl/_version.py"},
    description="Deep learning-based cancer drug response prediction with ScreenDL.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="#",
    packages=find_packages(exclude=["tests.*", "tests"]),
    setup_requires=["setuptools_scm"],
    install_requires=requirements,
    tests_require=test_requirements,
    python_requires=">=3.8",
    zip_safe=False,
    test_suite="tests",
    include_package_data=True,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
    ],
    maintainer="Casey Sederman",
    maintainer_email="casey.sederman@hsc.utah.edu",
)