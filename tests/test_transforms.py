import pylcms2
import os
import numpy as np


def test_neutral_grey():
    x = pylcms2.create_profile("Lab")
    y = pylcms2.create_profile("sRGB")
    t = pylcms2.Transform(x, "Lab_DBL",
                          y, "RGB_16",
                          "PERCEPTUAL",
                          "NONE")
    r = t.apply([50, 0, 0])
    reference = [30561, 30561, 30561]
    assert np.allclose(r, reference)


def test_RGB_to_CMYK():
    fn = os.path.join(os.path.dirname(__file__),
                      'cms_data',
                      "CMYK.icm")
    x = pylcms2.Profile("sRGB")
    y = pylcms2.Profile(filename=fn)
    t = pylcms2.Transform(x, "RGB_DBL",
                          y, "CMYK_DBL",
                          "PERCEPTUAL",
                          "NONE")
    r = t.apply([0.1, 0.2, 0.3])
    assert len(r) == 4



def test_apply_sliced():
    x = pylcms2.create_profile("sRGB")
    y = pylcms2.create_profile("XYZ")
    t = pylcms2.Transform(x, "RGB_8",
                          y, "XYZ_DBL",
                          "PERCEPTUAL",
                          "NONE")
    np.random.seed(456325)
    z = np.random.uniform(0, 255, (5, 3)).astype("uint8")
    reference = t.apply(z)
    sliced = t.apply(z[::2, :])
    assert np.allclose(reference[::2, :], sliced)


if __name__ == "__main__":
    test_RGB_to_CMYK()
    test_neutral_grey()
    test_apply_sliced()
