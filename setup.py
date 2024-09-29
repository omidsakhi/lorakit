from setuptools import find_packages, setup


setup(
    name="lorakit",
    version="0.1.0",
    description="A simple SDXL fine-tuning toolkit based on AutoTrain Advanced",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Omid Sakhi",
    author_email="omid.sakhi@gmail.com",
    url="https://github.com/omidsakhi/lorakit",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        "torch",
        "torchvision",
        "pyyaml",
        "accelerate",
        "transformers",
        "diffusers",
        "peft",
    ],
    entry_points={
        'console_scripts': [
            'lorakit = lorakit.cli.lorakit:main',
        ],
    },
    python_requires=">=3.7",
)
