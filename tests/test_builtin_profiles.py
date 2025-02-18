import pylcms2

def test_version():
    assert len(pylcms2.get_version()) >= 3

def is_valid_handle(h):
    t = type(h)
    return t.__module__ == 'builtins' and t.__name__ == 'PyCapsule'

def test_builtins():
    for profile_name in ("sRGB", "Lab", "XYZ"):
        p = pylcms2.Profile(profile_name)
        assert profile_name in p.name
        assert "use freely" in p.copyright
        assert is_valid_handle(p.handle)


if __name__ == "__main__":
    test_builtins()