from setuptools import setup, find_packages

setup(
    name="ml4cps",
    version="0.1.17",
    packages=find_packages(),
    install_requires=[
        'dash', 'pandas', 'networkx', 'plotly', 'numpy', 'dash_daq', 'dash-bootstrap-components', 'pydotplus',
        'dash-cytoscape', 'simpy', 'mlflow', 'torch', 'z3-solver', 'scipy', 'sphinx', 'matplotlib', 'scikit-learn',
        'fastdtw'
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