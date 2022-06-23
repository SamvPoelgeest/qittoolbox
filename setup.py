from setuptools import setup, find_packages

# Utilizing pathlib taken from: https://github.com/pypa/sampleproject/blob/main/setup.py
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(  name='qittoolbox',
        version='0.0.1',
        description='A Small Quantum Information Theory Toolbox',
        long_description=long_description,
        long_description_content_type="text/markdown",
        classifiers = [
            "Development Status :: 3 - Alpha",
            "Topic :: Quantum Information Theory :: Toolbox for separability and optimization",
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: GNU GPL v3",
            "Operating System :: OS Independent",
        ],
        url='https://github.com/SamvPoelgeest/qittoolbox',
        author='Sam van Poelgeest',
        author_email='S.vanPoelgeest@student.tudelft.nl',
        license='GNU GPL v3',
        # package_dir={"": "qittoolbox"},
        packages=find_packages(),#(where="qittoolbox"),
        package_data={'': ['*.npz']},
        python_requires=">=3.6",
        install_requires=['numpy','scipy'],
        project_urls = { 
            "Bug Reports" : "https://github.com/SamvPoelgeest/qittoolbox/issues"
        },
        zip_safe=False
)