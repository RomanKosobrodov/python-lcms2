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


def test_rgb_to_cmyk():
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


def test_proofing_transform():
    src = pylcms2.Profile("Lab")
    dst = pylcms2.Profile("sRGB")
    proof = pylcms2.Profile("sRGB")
    t1 = pylcms2.Transform(src, "Lab_DBL",
                          dst, "RGB_16",
                          proofing_profile=proof,
                          proofing_intent="PRESERVE_K_ONLY_PERCEPTUAL",
                          intent="PERCEPTUAL",
                          flags="NONE")
    t2 = pylcms2.Transform(src, "Lab_DBL",
                          dst, "RGB_16",
                          intent="PERCEPTUAL",
                          flags="NONE")
    x = [50, 11, -5]
    r1 = t1.apply(x)
    r2 = t2.apply(x)
    assert np.allclose(r1, r2, atol=0)


if __name__ == "__main__":
    test_rgb_to_cmyk()
    test_neutral_grey()
    test_apply_sliced()
    test_proofing_transform()
