from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="geospatial-vit",
    version="0.1.0",
    author="James Englis",
    author_email="your-email@example.com",
    description="Vision Transformer for Geospatial Drone Imagery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jamesenglis/geospatial-vit-drone",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "pillow>=10.0.0",
        "albumentations>=1.3.0",
        "timm>=0.9.0",
        "einops>=0.6.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
)
