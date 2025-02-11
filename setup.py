from setuptools import setup, Extension
from sys import platform

SOURCES = ["src/lcms2/_lcms2.c",
           "Little-CMS/src/cmsalpha.c",
           "Little-CMS/src/cmscam02.c",
           "Little-CMS/src/cmscgats.c",
           "Little-CMS/src/cmscnvrt.c",
           "Little-CMS/src/cmserr.c",
           "Little-CMS/src/cmsgamma.c",
           "Little-CMS/src/cmsgmt.c",
           "Little-CMS/src/cmshalf.c",
           "Little-CMS/src/cmsintrp.c",
           "Little-CMS/src/cmsio0.c",
           "Little-CMS/src/cmsio1.c",
           "Little-CMS/src/cmslut.c",
           "Little-CMS/src/cmsmd5.c",
           "Little-CMS/src/cmsmtrx.c",
           "Little-CMS/src/cmsnamed.c",
           "Little-CMS/src/cmsopt.c",
           "Little-CMS/src/cmspack.c",
           "Little-CMS/src/cmspcs.c",
           "Little-CMS/src/cmsplugin.c",
           "Little-CMS/src/cmsps2.c",
           "Little-CMS/src/cmssamp.c",
           "Little-CMS/src/cmssm.c",
           "Little-CMS/src/cmstypes.c",
           "Little-CMS/src/cmsvirt.c",
           "Little-CMS/src/cmswtpnt.c",
           "Little-CMS/src/cmsxform.c"]

INCLUDE_DIRECTORIES = ["Little-CMS/include", "Little-CMS/src"]
LIBRARY_DIRECTORIES = list()

if platform == "win32":
     sdk = "C:\\Program Files (x86)\\Windows Kits\\10\\"
     sdk_include = sdk + "Include\\10.0.26100.0\\"
     sdk_lib = sdk + "Lib\\10.0.26100.0\\"
     for p in ("cppwinrt", "shared", "ucrt", "um", "winrt"):
         INCLUDE_DIRECTORIES.append(sdk_include + p)

     for p in ("ucrt", "um"):
         LIBRARY_DIRECTORIES.append(sdk_lib + p + "\\x64")


setup_args = dict(
    ext_modules=[
        Extension(
            name="_lcms2",
            language="c",
            define_macros=[('MAJOR_VERSION', '0'), ('MINOR_VERSION', '4')],
            include_dirs=INCLUDE_DIRECTORIES,
            library_dirs=LIBRARY_DIRECTORIES,
            sources=SOURCES,
            py_limited_api=True
        )
    ]
)

setup(**setup_args)