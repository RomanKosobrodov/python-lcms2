import os
import pylcms2


def get_filepath(filename):
    return os.path.join(os.path.dirname(__file__),
                        'cms_data',
                        filename)


def test_invalid_profile():
    try:
        filename = get_filepath('empty.icm')
        pylcms2.open_profile(filename)
    except Exception as e:
        return
    assert False


def test_missing_profile():
    try:
        filename = get_filepath('this_profile_does_not_exist.icm')
        pylcms2.open_profile(filename)
    except Exception as e:
        return
    assert False

def test_valid_cmyk_profile():
    filename = get_filepath("CMYK.icm")
    p = pylcms2.open_profile(filename)
    assert "CMYK" in p.name
    assert "Offset printing" in p.info
    assert "Public" in p.copyright


if __name__ == "__main__":
    test_invalid_profile()
    test_missing_profile()
    test_valid_cmyk_profile()