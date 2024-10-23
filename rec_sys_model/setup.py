from setuptools import setup, find_packages

setup(
    name='rec_sys_model',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'torch',
        'scikit-learn',
        'transformers'
    ]
)
