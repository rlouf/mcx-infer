import setuptools

setuptools.setup(
    name="mcxinfer",
    version="0.0.1",
    author="Rémi Louf",
    author_email="remilouf@gmail.com",
    description="Bayesian Machine Learning in Python",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rlouf/mcx-infer",
    licence="Apache Licence Version 2.0",
    packages=["infer"],
    python_requires=">=3.5",
    install_requires=["daft", "matplotlib", "jax==0.1.77"],
)
