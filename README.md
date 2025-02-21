# pylcms2

Simplified Python binding of [LittleCMS2](https://github.com/mm2/Little-CMS) library.

# Installation

Wheel files for 64-bit Windows and Python 3.13 are available from PyPI. Use `pip` to install the package:

```python
pip install pylcms2
```

The package can be uninstalled using the corresponding command:

```python
python -m pip uninstall -y pylcms2
```

# Getting started

Little CMS functionality is implemented by two classes: `Profile` and `Transform`.

## Profiles

A `Profile` object can be created using one of the built-in profiles, by reading a profile file (using the `filename` argument) or from data stored in memory as Python `bytes` object (`buffer` argument).

Built-in profiles include `sRGB`, `Lab` and `XYZ` provided by the Little CMS library and additional RGB profiles: 
`Adobe RGB (1998)`, `Apple RGB`, `Best RGB`, `Beta RGB`, `Bruce RGB`, `CIE RGB`, `ColorMatch RGB`, `Don RGB 4`, `ECI RGB v2`, `Ekta Space PS5`, `NTSC RGB`, `PAL/SECAM RGB`, `ProPhoto RGB`, `SMPTE-C RGB`, `_sRGB`, and `Wide Gamut RGB`.

```
import pylcms2

x = pylcms2.Profile("sRGB")
p = pylcms2.Profile("ProPhoto RGB")
y = pylcms2.Profile(filename="CMYK.icm")

with open("CMYK.icm", "rb") as f:
    z = pylcms2.Profile(buffer=f.read)
```

## Transforms

Transforms between profiles are implemented by the respective class. To create an instance of `Transform` call its constructor and provide six arguments: (1) source profile, (2) source data type string, (3) destination profile, (4) destination data type string, (5) rendering intent string (optional, defaults to "PERCEPTUAL"), and (6) flags (optional, the default value is "NONE").

```python
lab_profile = pylcms2.create_profile("Lab")
rgb_profile = pylcms2.create_profile("sRGB")
transform = pylcms2.Transform(lab_profile, "Lab_DBL",
                              rgb_profile, "RGB_16",
                              "PERCEPTUAL",
                              "NONE")
```

Supported format strings can be listed as follows:

```python
pylcms2.DATA_TYPES.keys()
```

Rendering intents could be one of the following: 'PERCEPTUAL', 'RELATIVE_COLORIMETRIC', 'SATURATION', and 'ABSOLUTE_COLORIMETRIC' (see `pylcms2.INTENT` dictionary).

Likewise, supported transform flags are available as keys of the `pylcms2.FLAG` dictionary. Several flags can be combined by including them in the `flags` argument string separating them by commas, semicolons, spaces or the pipe symbol `|`, for example

```python
flags = "GAMUTCHECK,SOFTPROOFING,PRESERVEBLACK"
```

The transform is calculated using the `Transform.apply` method, for example:

```python
neutral_grey = transform.apply([50.0, 0.0, 0.0])
```

The method takes a Numpy array (or an object convertible to NumPy array) and returns a transformed array of appropriate shape. The data type of the input array must correspond to the input data type specified during transform creation. When lists and other array-compatible objects are used the data types will be converted, when possible. For instance in the example above the input data could be provided as `[50, 0, 0]` (list containing integers):

```python
neutral_grey = transform.apply([50, 0, 0])
```

## Building form source

To build `pylcms2` for your own platform, clone the repository with its submodules:

```bash
git clone --recurse-submodules git@github.com:RomanKosobrodov/python-lcms2.git
```

Check that the `build` package is installed:

```python
python -m pip install build --upgrade
```

navigate to the root directory of the repository (the one containing the `pyproject.toml` file) and run:

```python
python -m build --wheel
```

If everything goes smoothly you will have a wheel file for you platform generated in the `dist` directory. It can be installed by pip as follows:

```python
python -m pip install --upgrade dist/pylcms2*.whl
```

Building the package on Windows might involve minor modifications to include and library paths. An attempt was made to automate the process (see `setup.py`) though there is no guarantee that the code works on your version of Windows.

# Issues, bugs and feedback

You are likely to hit a few bugs and issues while using `pylcms2`. Please report them through [GitHub Issues]("https://github.com/RomanKosobrodov/python-lcms2/issues").
