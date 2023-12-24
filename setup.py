from setuptools import find_packages,setup

def read_requirement_file(file_path='requirements.txt'):
    constant = '-e .'
    with open(file_path, 'r') as f:
        packages = f.readlines()
        packages = [package.replace('/n','') for package in packages]

        if constant in packages:
            packages.remove(constant)
            
        return packages
        

setup(
    name='Sensor Fault Detection',
    version='1.0',
    author_email='oms421621@gmail.com',
    packages=find_packages(),
    install_requires=read_requirement_file()
)