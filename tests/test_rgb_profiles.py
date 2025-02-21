import pylcms2
import numpy as np


def test_srgb_profile():
    reference = pylcms2.Profile("sRGB")
    p = pylcms2.implementation.create_profile("_sRGB")
    lab = pylcms2.Profile("Lab")
    t1 = pylcms2.Transform(lab, "Lab_DBL", reference, "RGB_16")
    t2 = pylcms2.Transform(lab, "Lab_DBL", p, "RGB_16")
    v = [50.0, 2.0, -3.0]
    z1 = t1.apply(v)
    z2 = t2.apply(v)
    assert np.allclose(z1, z2, atol=1)


if __name__ == "__main__":
    test_srgb_profile()
