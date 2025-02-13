import os
import pylcms2


def get_filepath(filename):
    return os.path.join(os.path.dirname(__file__),
                        'cms_data',
                        filename)


def test_invalid_profile():
    try:
        filename = get_filepath('empty.icm')
        pylcms2.cmsOpenProfileFromFile(filename)
    except pylcms2.CmsError:
        return
    except Exception as e:
        raise e


def test_missing_profile():
    try:
        filename = get_filepath('this_profile_does_not_exist.icm')
        pylcms2.cmsOpenProfileFromFile(filename)
    except pylcms2.CmsError:
        return
    except Exception as e:
        raise e

def test_valid_cmyk_profile():
    filename = get_filepath("CMYK.icm")
    profile = pylcms2.cmsOpenProfileFromFile(filename)
    name = pylcms2.cmsGetProfileName(profile)
    assert "CMYK" in name
    info = pylcms2.cmsGetProfileInfo(profile)
    assert "Offset printing" in info
    copyright = pylcms2.cmsGetProfileCopyright(profile)
    assert "Public" in copyright

