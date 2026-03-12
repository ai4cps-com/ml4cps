import os
from setuptools import setup, find_packages
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
main_ns = {}
exec((BASE_DIR / "selfx" / "version.py").read_text(encoding="utf-8"), main_ns) # pylint: disable=exec-used, consider-using-with

def read_req_file(req_type):
    with open(os.path.join("requirements", f"{req_type}.txt"), encoding="utf-8") as fp:
        requires = (line.strip() for line in fp)
        return [req for req in requires if req and not req.startswith("#")]

setup(
    name="ml4cps",
    version=main_ns["__version__"],
    packages=find_packages(),
    install_requires=[
        'dash', 'pandas', 'networkx', 'plotly', 'numpy', 'dash_daq', 'dash-bootstrap-components', 'pydotplus',
        'dash-cytoscape', 'simpy', 'mlflow', 'torch', 'z3-solver', 'scipy', 'sphinx', 'matplotlib', 'scikit-learn',
        'fastdtw', 'openai', 'loguru', 'regex', 'Levenshtein', 'tqdm', 'gymnasium'
    ],
    author="Nemanja Hranisavljevic & Tom Westermann",
    author_email="nemanja@ai4cps.com",
    description="Tools for learning, plotting, analyzing etc. of discrete, continuous, timed, and "
                "hybrid cyber-physical systems.",
    url="https://github.com/ai4cps-com/ml4cps",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)