from setuptools import setup, find_packages

setup(
    name="Quorabust",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "xgboost>=2.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.7", "seaborn>=0.13"],
        "notebooks": ["jupyter>=1.0"],
        "dev": [
            "pytest>=7.4",
            "pytest-cov>=4.1",
            "ruff>=0.1",
        ],
    },
    author="AliSina Karimi",
    author_email="alisinakarimi.2003@gmail.com",
    description="Text features and models for Quora-style duplicate question detection.",
    url="https://github.com/AliSinaDevelo/Quorabust",
)
