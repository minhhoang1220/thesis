# /.ndmh/setup.py
import os
from setuptools import setup, find_packages

# Get version from marketml/__init__.py
def get_version():
    version_filepath = os.path.join(os.path.dirname(__file__), 'marketml', '__init__.py')
    with open(version_filepath) as f:
        for line in f:
            if line.startswith('__version__'):
                return line.strip().split()[-1].strip("'")
    return "0.0.1"

# Requirements
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

# Long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='marketml',
    version=get_version(),
    author='Nguyen Duc Minh Hoang',
    author_email='hoang.nguyenducminh@gmail.com',
    description='Market trend forecasting and portfolio optimization for thesis',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/minhhoang1220/thesis.git',
    # Packages and exclusions
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    include_package_data=True,
    install_requires=install_requires,
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='finance machine-learning market-forecasting portfolio-optimization reinforcement-learning thesis',
    project_urls={
        'Source': 'https://github.com/minhhoang1220/thesis/',
    },
    # Entry points for CLI scripts
    entry_points={
        'console_scripts': [
            'marketml-pipeline=run_pipeline:main',
        ],
    },
)
