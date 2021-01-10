import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deeplabv3plus-keras",
    version="1.1.0",
    author="Inwoo Chung",
    author_email="gutomitai@gmail.com",
    description="Keras deeplabV3+ semantic segmentation model using Xception and MobileNetV2 as a base model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tonandr/deeplabv3plus_keras",
    packages=setuptools.find_packages(exclude=['analysis', 'samples', 'resource']),
    install_requires=['scipy'
                      , 'pandas'
                      , 'scikit-image'
                      , 'opencv-contrib-python'
                      , 'matplotlib'
                      , 'tqdm'],  # Optional
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)