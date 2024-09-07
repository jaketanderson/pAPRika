"""
pAPRika
Advanced toolkit for binding free energy calculations
"""

from setuptools import find_packages, setup

import versioneer

short_description = __doc__.split("\n")

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except IOError:
    long_description = ("\n".join(short_description[2:]),)


setup(
    # Self-descriptive entries which should always be present
    name="paprika",
    author="David R. Slochower, Niel M. Henriksen, and Jeffry Setiadi",
    author_email="slochower@gmail.com, shireham@gmail.com, pea231@gmail.com",
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="BSD-3-Clause",
    # Which Python importable modules should be included when your package is installed
    packages=find_packages(),
    # Optional include package data to ship with your package
    # Comment out this line to prevent the files from being packaged with your software
    # Extend/modify the list to include/exclude other items as need be
    package_data={"paprika": ["data/*.dat"]},
    include_package_data=True,
    install_requires=[
        "numpy",
        "jax<=0.4.28; python_version == '3.9'", # Since python<=3.9 doesn't support `str | None` syntax, pin jax to a version that uses `None` or `Optional[str]`
    ],
    # Additional entries you may want simply uncomment the lines you want and fill in the data
    # author_email='me@place.org',      # Author email
    # url='http://www.my_package.com',  # Website
    # install_requires=[],              # Required packages, pulls from pip if needed; do not use for Conda deployment
    # platforms=['Linux',
    #            'Mac OS-X',
    #            'Unix',
    #            'Windows'],            # Valid platforms your code works on, adjust to your flavor
    # python_requires=">=3.5",          # Python version restrictions
    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    # zip_safe=False,
)
