from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ad-intention",
    version="0.1.0",
    author="AdClarity Team",
    author_email="team@example.com",
    description="A system for classifying ad landing page URLs as Brand Awareness or Call to Action",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/ad-intention",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.5.0",
        "tqdm>=4.65.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.1.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
        "notebook": [
            "notebook>=7.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
        ],
        "ml": [
            "scikit-learn>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "adintent-classify=ad_intention.examples.classify_urls:main",
            "adintent-batch=ad_intention.examples.batch_processing:main",
        ],
    },
) 