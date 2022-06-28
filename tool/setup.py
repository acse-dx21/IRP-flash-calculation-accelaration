from distutils.core import setup, Extension

def main():
    setup(name="fputs",
          version="1.0.0",
          description="Python interface for the fputs C library function",
          author="cxa",
          author_email="1598828268@qq.com",
          ext_modules=[Extension("Py2Cpp", ["Py2Cpp.cpp"])])

if __name__ == "__main__":
    main()