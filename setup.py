from setuptools import setup, find_packages

setup(
    name='marketml',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas>=1.3',
        'numpy>=1.20',
        'yfinance>=0.2.12',
        'scikit-learn>=1.0',
        'matplotlib',
        'seaborn',
        'gymnasium',
        'torch>=1.9',
        'openpyxl',
    ],
    author='Nguyen Duc Minh Hoang',
    email='hoang.nguyenducminh@gmail.com',
    url='https://github.com/minhhoang1220/thesis.git',
    description='Market trend forecasting and portfolio optimization for thesis',
    long_description='Thesis project using ML and RL for forecasting and optimization.',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Intended Audience :: Education',
    ],
    python_requires='>=3.8',
)

