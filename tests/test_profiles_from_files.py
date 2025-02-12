import os
import lcms2


def get_filepath(filename):
    return os.path.join(os.path.dirname(__file__),
                        'cms_data',
                        filename)


def test_invalid_profile():
    try:
        filename = get_filepath('empty.icm')
        lcms2.cmsOpenProfileFromFile(filename)
    except lcms2.CmsError:
        return
    except Exception as e:
        raise e


def test_missing_profile():
    try:
        filename = get_filepath('this_profile_does_not_exist.icm')
        lcms2.cmsOpenProfileFromFile(filename)
    except lcms2.CmsError:
        return
    except Exception as e:
        raise e

def test_valid_cmyk_profile():
    filename = get_filepath("CMYK.icm")
    profile = lcms2.cmsOpenProfileFromFile(filename)
    name = lcms2.cmsGetProfileName(profile)
    assert "CMYK" in name
    info = lcms2.cmsGetProfileInfo(profile)
    assert "Offset printing" in info
    copyright = lcms2.cmsGetProfileCopyright(profile)
    assert "Public" in copyright

