import pylcms2

def test_version():
    assert len(pylcms2.get_version()) >= 3

def test_builtins():
    profiles = {
        "sRGB": pylcms2.cmsCreate_sRGBProfile,
        "Lab": pylcms2.cmsCreateLabProfile,
        "gray": pylcms2.cmsCreateGrayProfile,
        "XYZ": pylcms2.cmsCreateXYZProfile
    }

    for profile_name, profile_function in profiles.items():
        p = profile_function()
        name = pylcms2.cmsGetProfileName(p)
        profile_copyright = pylcms2.cmsGetProfileCopyright(p)
        profile_info = pylcms2.cmsGetProfileInfo(p)
        assert profile_name in name
        assert "use freely" in  profile_copyright
        assert isinstance(profile_info, str)


if __name__ == "__main__":
    test_builtins()