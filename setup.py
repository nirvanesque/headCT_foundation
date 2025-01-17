from setuptools import find_packages, setup

setup(name='HeadCT-Foundation', packages=find_packages())
setup(
    name='HeadCT-Foundation',
    version='0.1.0',
    description='HeadCT-Foundation',
    #url='https://github.com/mahmoodlab/UNI',
    author='HH, JZ',
    author_email='',
    license='CC BY-NC 4.0',
    packages=find_packages(exclude=['__dep__', 'assets']),
    install_requires=["torch>=2.0.1", "timm==1.0.12", "monai==1.3.2",
                      "numpy", "pandas", "scikit-learn", "tqdm",
                      "transformers"],

    classifiers = [
    "Programming Language :: Python :: 3",
    "License :: CC BY-NC 4.0",
]
)