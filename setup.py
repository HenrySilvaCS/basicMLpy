import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="basicMLpy", 
    version="1.0.3",
    author="Henrique Silva",
    author_email="henriquesoares@dcc.ufmg.br",
    description="A collection of simple machine learning algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HenrySilvaCS/basicMLpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    py_modules=["classification","regression","model_selection","ensemble","nearest_neighbors","utils","loss_functions"],
    install_requires=[
       'numpy>=1.19',
       'scipy>=1.5.2',
       'scikit-learn>=0.23'
    ],
    python_requires='>=3.8',
)