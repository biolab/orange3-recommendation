from setuptools import setup
from setuptools.command.install import install


class MyCommand(install):
    def run(self):
        print("Hello, developer, how are you? :)")
        install.run(self)


if __name__ == '__main__':
    setup(
        cmdclass={
            'install': MyCommand,
        }
    )