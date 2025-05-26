# /.ndmh/setup.py
import os
from setuptools import setup, find_packages

# Function to read the version from marketml/__init__.py
def get_version():
    version_filepath = os.path.join(os.path.dirname(__file__), 'marketml', '__init__.py')
    with open(version_filepath) as f:
        for line in f:
            if line.startswith('__version__'):
                return line.strip().split()[-1].strip("'")
    return "0.0.1" # Fallback version

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

# Read long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='marketml',
    version=get_version(), # Lấy version từ marketml/__init__.py
    author='Nguyen Duc Minh Hoang', # Giữ nguyên
    author_email='hoang.nguyenducminh@gmail.com', # Giữ nguyên
    description='Market trend forecasting and portfolio optimization for thesis', # Giữ nguyên
    long_description=long_description, # Sử dụng nội dung README
    long_description_content_type="text/markdown", # Định dạng của long_description
    url='https://github.com/minhhoang1220/thesis.git', # Giữ nguyên
    # find_packages() sẽ tự động tìm package 'marketml' và các sub-packages của nó
    # nếu setup.py nằm cùng cấp với thư mục marketml (trong .ndmh/)
    packages=find_packages(exclude=["tests*", "notebooks*"]), # Loại trừ tests và notebooks khỏi package
    include_package_data=True, # Bao gồm các file non-code trong MANIFEST.in (nếu có)
    install_requires=install_requires, # Lấy dependencies từ requirements.txt
    python_requires='>=3.8', # Giữ nguyên
    classifiers=[ # Giữ nguyên hoặc cập nhật
        'Development Status :: 3 - Alpha', # Hoặc 4 - Beta, 5 - Production/Stable
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='finance machine-learning market-forecasting portfolio-optimization reinforcement-learning thesis',
    project_urls={ # Thêm các URL hữu ích
        'Source': 'https://github.com/minhhoang1220/thesis/',
    },
    # Entry points để tạo command-line scripts (tùy chọn)
    entry_points={
        'console_scripts': [
            'marketml-pipeline=run_pipeline:main', # Giả sử run_pipeline.py có hàm main()
        ],
    },
)