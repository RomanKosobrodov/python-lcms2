import lcms2

def test_version():
    assert len(lcms2.get_version()) >= 3

def test_builtins():
    profiles = {
        "sRGB": lcms2.cmsCreate_sRGBProfile,
        "Lab": lcms2.cmsCreateLabProfile,
        "gray": lcms2.cmsCreateGrayProfile,
        "XYZ": lcms2.cmsCreateXYZProfile
    }

    for profile_name, profile_function in profiles.items():
        p = profile_function()
        name = lcms2.cmsGetProfileName(p)
        profile_copyright = lcms2.cmsGetProfileCopyright(p)
        profile_info = lcms2.cmsGetProfileInfo(p)
        assert profile_name in name
        assert "use freely" in  profile_copyright
        assert isinstance(profile_info, str)


if __name__ == "__main__":
    test_builtins()