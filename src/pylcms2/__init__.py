import os
import re
import _lcms2
import numpy as np

DATA_TYPES = _lcms2.DATA_TYPES

INTENT = {
	"PERCEPTUAL": 0,
	"RELATIVE_COLORIMETRIC": 1,
	"SATURATION": 2,
	"ABSOLUTE_COLORIMETRIC": 3
}

FLAG = {
	"NONE": 0x0,
	"NOTPRECALC": 0x0100,
	"GAMUTCHECK": 0x1000,
	"SOFTPROOFING": 0x4000,
	"BLACKPOINTCOMPENSATION": 0x2000,
	"PRESERVEBLACK": 0x8000,
	"NULLTRANSFORM": 0x0200,
	"HIGHRESPRECALC": 0x0400,
	"LOWRESPRECALC": 0x0800
}

class CMSError(Exception):
	def __init__(self, message="LibCMS2 error"):
		super().__init__(message)


def get_version():
	ver = str(_lcms2.get_version())
	return f"{ver[0]}.{ver[1:]}"


def create_profile(profile):
	if profile not in ("sRGB", "Lab", "XYZ"):
		raise CMSError(f"Invalid profile '{profile}'. It must be one of: 'sRGB', 'Lab' or 'XYZ'")
	p = _lcms2.create_profile(profile)
	return p


def open_profile(filename):
	if not isinstance(filename, str):
		raise CMSError("filename must be a string containing a path to a profile")
	if not os.path.isfile(filename):
		raise CMSError(f"Unable to find '{filename}'.")

	p = _lcms2.open_profile(filename)
	return p

def profile_from_memory(buffer):
	if not isinstance(buffer, bytes):
		raise CMSError("filename must be a bytes object")

	p = _lcms2.profile_from_memory(buffer)
	return p


def profile_to_bytes(profile):
	if not isinstance(profile, (_lcms2.Profile, Profile)):
		raise CMSError("wrong type of argument: expected Profile")
	return _lcms2.profile_to_bytes(profile)


class Profile:
	def __init__(self, builtin=None, filename=None, buffer=None):
		if builtin is not None and filename is None and buffer is None:
			self._assign(profile=create_profile(builtin))
			return
		if filename is not None and builtin is None and buffer is None:
			self._assign(profile=open_profile(filename))
			return
		if buffer is not None and builtin is None and filename is None:
			self._assign(profile=profile_from_memory(buffer))
			return
		raise CMSError("Only one source of profile data can be specified: a built-in profile name, filename or bytes object")

	def _assign(self, profile):
		self.handle = profile.handle
		self.name = profile.name
		self.info = profile.info
		self.copyright = profile.copyright

	def to_bytes(self):
		return _lcms2.profile_to_bytes(self)

	def save(self, filename):
		with open(filename, "wb") as f:
			f.write(self.to_bytes())	


class Transform:
	def __init__(self, src_profile, src_format, dst_profile, dst_format, intent="PERCEPTUAL", flags="NONE"):
		if not isinstance(src_profile, (Profile, _lcms2.Profile)):
			raise CMSError(f"Wrong type of src_profile. Expected Profile but got '{type(src_profile)}'")
		if src_format not in DATA_TYPES.keys():
			raise CMSError(f"Invalid source data format: '{src_format}'")
		if not isinstance(dst_profile, (Profile, _lcms2.Profile)):
			raise CMSError(f"Wrong type of dst_profile. Expected Profile but got '{type(src_profile)}'")
		if dst_format not in DATA_TYPES.keys():
			raise CMSError(f"Invalid destination data format: '{dst_format}'")
		if intent not in INTENT.keys():
			raise CMSError(f"Invalid rendering intent: '{intent}'")
		flag_list = re.split("[ ,;|]", flags)
		transform_flags = FLAG["NONE"]
		for f in flag_list:
			if f not in FLAG.keys():
				raise CMSError(f"Invalid flag: '{f}'")
			transform_flags = transform_flags | FLAG[f]
		
		self.src_format = src_format
		self.dst_format = dst_format
		self.transform = _lcms2.Transform(src_profile, DATA_TYPES[src_format][0], 
							dst_profile, DATA_TYPES[dst_format][0], 
							INTENT[intent], transform_flags)

	def apply(self, src):
		_, src_numpy_type, src_channels = DATA_TYPES[self.src_format]	
		_, dst_numpy_type, dst_channels = DATA_TYPES[self.dst_format]
		x = src
		if not isinstance(src, np.ndarray):
			x = np.array(src, dtype=src_numpy_type)
		if x.dtype != src_numpy_type:
			raise CMSError(f"Source data type ({x.dtype}) does not match transform input type ({src_numpy_type}")
		if x.shape[-1] != src_channels:
			raise CMSError(f"Wrong number of input channels ({x.shape[-1]}); expected {src_channels}")
		dst_shape = np.empty_like(x.shape)
		dst_shape[:-1] = x.shape[:-1]
		dst_shape[-1] = dst_channels
		dst = np.empty(shape=dst_shape, dtype=dst_numpy_type)
		self.transform.apply(x, dst, x.size//src_channels)
		return dst

