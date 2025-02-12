import lcms2


def test_xyz_to_srgb():
    xyz_profile = lcms2.cmsCreateXYZProfile()
    srgb_profile = lcms2.cmsCreate_sRGBProfile()
    transform = lcms2.cmsCreateTransform(xyz_profile,
                                         lcms2.TYPE_XYZ_DBL,
                                         srgb_profile,
                                         lcms2.TYPE_RGB_8,
                                         lcms2.INTENT_PERCEPTUAL)

    xyz = [0.7, 0.4, 0.2, 0, lcms2.COLOR_DBL]
    srgb = [0, 0, 0, 0, lcms2.COLOR_BYTE]
    lcms2.cmsDoTransform(transform, xyz, srgb)
    srgb_reference = [255, 84, 134]
    assert srgb[:3] == srgb_reference
