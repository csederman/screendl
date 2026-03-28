import codecs

from setuptools import find_packages
from setuptools import setup


requirements = [
    "numpy >= 1.21",
    "pandas >= 2.0.3",
    "openpyxl",
    "scikit-learn",
    "omegaconf",
    "tqdm",
    "cdrpy",
    "scipy >= 1.8.1",
    "mygene >= 3.2.2",
]

extras_require = {
    "gpu": ["tensorflow[and-cuda]"],
    "preprocess": [
        "inmoose == 0.2.1",
        "deepchem",
        "rdkit",
    ],
    "notebooks": [
        "jupyterlab",
        "altair",
        "shap",
        "statsmodels",
    ],
    "scripts": [
        "click",
        "hydra-core >= 1.3",
        "hydra-optuna-sweeper >= 1.2.0",
    ],
}

# default to CPU tensorflow so bare `pip install screendl` still works
extras_require["all-cpu"] = extras_require["scripts"] + extras_require["notebooks"]
extras_require["all-gpu"] = (
    extras_require["gpu"] + extras_require["scripts"] + extras_require["notebooks"]
)

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
    url="https://github.com/csederman/screendl",
    packages=find_packages(exclude=["tests.*", "tests"]),
    setup_requires=["setuptools_scm"],
    install_requires=requirements,
    tests_require=test_requirements,
    extras_require=extras_require,
    python_requires=">=3.8",
    zip_safe=False,
    test_suite="tests",
    include_package_data=True,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.12",
    ],
    maintainer="Casey Sederman",
    maintainer_email="casey.sederman@hsc.utah.edu",
)
