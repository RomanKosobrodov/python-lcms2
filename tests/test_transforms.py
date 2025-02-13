import pylcms2


def test_xyz_to_srgb():
    xyz_profile = pylcms2.cmsCreateXYZProfile()
    srgb_profile = pylcms2.cmsCreate_sRGBProfile()
    transform = pylcms2.cmsCreateTransform(xyz_profile,
                                           pylcms2.TYPE_XYZ_DBL,
                                           srgb_profile,
                                           pylcms2.TYPE_RGB_8,
                                           pylcms2.INTENT_PERCEPTUAL)

    xyz = [0.7, 0.4, 0.2, 0, pylcms2.COLOR_DBL]
    srgb = [0, 0, 0, 0, pylcms2.COLOR_BYTE]
    pylcms2.cmsDoTransform(transform, xyz, srgb)
    srgb_reference = [255, 84, 134]
    assert srgb[:3] == srgb_reference
