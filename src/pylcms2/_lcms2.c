/* _lcms2 - small module which provides binding to LittleCMS library version 2.
 *
 * Copyright (C) 2017 by Igor E.Novikov
 *
 * 	This program is free software: you can redistribute it and/or modify
 *	it under the terms of the GNU General Public License as published by
 *	the Free Software Foundation, either version 3 of the License, or
 *	(at your option) any later version.
 *
 *	This program is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *	GNU General Public License for more details.
 *
 *	You should have received a copy of the GNU General Public License
 *	along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <lcms2.h>
#include <lcms2_internal.h>
#include <string.h>

#include "profile.h"

#define BUFFER_SIZE 1000

static void
free_profile_handle(PyObject *capsule){
    cmsHPROFILE profile = (cmsHPROFILE) PyCapsule_GetPointer(capsule, NULL);
    if (profile != NULL) {
        cmsCloseProfile(profile);
    }
}

static  PyObject*
create_python_string(const char* src, cmsUInt32Number count)
{
    if (count == 0)
        return PyUnicode_FromString("");
    else
        return PyUnicode_FromStringAndSize(src, count - 1); // trim 0x0 at the end
}

static profile_object*
create_profile_from_handle(cmsHPROFILE profile_handle) {

    if(profile_handle == NULL) {
	    PyErr_SetString(PyExc_RuntimeError, "Profile is invalid or the file is corrupt.");
        return NULL;
    }

    profile_object* result = (profile_object*) PyObject_CallNoArgs((PyObject*)&profile_type);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create Profile object");
        return NULL;
    }

	char* buffer = malloc(BUFFER_SIZE);
	cmsUInt32Number info_size;

	info_size = cmsGetProfileInfoUTF8(profile_handle,
			cmsInfoDescription,
			cmsNoLanguage, cmsNoCountry,
			buffer, BUFFER_SIZE);
    result->name = create_python_string(buffer, info_size);

	info_size = cmsGetProfileInfoUTF8(profile_handle,
			cmsInfoModel,
			cmsNoLanguage, cmsNoCountry,
			buffer, BUFFER_SIZE);
    result->info = create_python_string(buffer, info_size);

	info_size = cmsGetProfileInfoUTF8(profile_handle,
			cmsInfoCopyright,
			cmsNoLanguage, cmsNoCountry,
			buffer, BUFFER_SIZE);
    result->copyright = create_python_string(buffer, info_size);

    free(buffer);

    if (result->name == NULL || result->info == NULL || result->copyright == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Unable retrieve profile information");
        return NULL;
    }

    result->handle = PyCapsule_New(profile_handle,
                                   NULL,
                                   free_profile_handle);
    if (result->handle == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Unable create handle object for profile");
        return NULL;
    }

    return result;
}


static PyObject *
create_profile(PyObject *self, PyObject *args)
{
    char *profile_name = NULL;
	if (!PyArg_ParseTuple(args, "s", &profile_name)){
	    PyErr_SetString(PyExc_ValueError, "Invalid argument - expected a string");
		return NULL;
	}

    cmsHPROFILE profile_handle = NULL;
    if (strcmp(profile_name, "sRGB") == 0) {
        profile_handle = cmsCreate_sRGBProfile();
	}
    else if (strcmp(profile_name, "XYZ") == 0) {
        profile_handle = cmsCreateXYZProfile();
    }
    else if (strcmp(profile_name, "Lab") == 0) {
        profile_handle = cmsCreateLab4Profile(0);
    }

	return (PyObject*) create_profile_from_handle(profile_handle);
}

static PyObject *
open_profile(PyObject *self, PyObject *args)
{
    char *file_name = NULL;
	if (!PyArg_ParseTuple(args, "s", &file_name)){
	    PyErr_SetString(PyExc_ValueError, "Invalid argument - expected a string");
		return NULL;
	}
    cmsHPROFILE profile_handle = cmsOpenProfileFromFile(file_name, "r");
    return (PyObject*) create_profile_from_handle(profile_handle);
}


cmsUInt32Number
getLCMStype (char* mode) {

  if (strcmp(mode, "RGB") == 0) {
    return TYPE_RGBA_8;
  }
  else if (strcmp(mode, "RGBA") == 0) {
    return TYPE_RGBA_8;
  }
  else if (strcmp(mode, "RGBX") == 0) {
    return TYPE_RGBA_8;
  }
  else if (strcmp(mode, "RGBA;16") == 0) {
    return TYPE_RGBA_16;
  }
  else if (strcmp(mode, "CMYK") == 0) {
    return TYPE_CMYK_8;
  }
  else if (strcmp(mode, "CMYK;16") == 0) {
    return TYPE_CMYK_16;
  }
  else if (strcmp(mode, "L") == 0) {
    return TYPE_GRAY_8;
  }
  else if (strcmp(mode, "L;16") == 0) {
    return TYPE_GRAY_16;
  }
  else if (strcmp(mode, "LAB") == 0) {
    return TYPE_Lab_8;
  }
  else if (strcmp(mode, "LAB;float") == 0) {
    return TYPE_Lab_DBL;
  }
  else if (strcmp(mode, "XYZ;float") == 0) {
    return TYPE_XYZ_DBL;
  }

  else {
    return TYPE_GRAY_8;
  }
}


static PyObject *
pycms_OpenProfile(PyObject *self, PyObject *args) {

	char *profile = NULL;
	cmsHPROFILE hProfile;

	if (!PyArg_ParseTuple(args, "s", &profile)){
		Py_INCREF(Py_None);
		return Py_None;
	}

	hProfile = cmsOpenProfileFromFile(profile, "r");

	if(hProfile==NULL) {
		Py_INCREF(Py_None);
		return Py_None;
	}

    PyObject* capsule = PyCapsule_New((void *)hProfile,
                                       NULL,
                                       (void *)cmsCloseProfile);
    return Py_BuildValue("O", capsule);
}

static PyObject *
pycms_CreateRGBProfile(PyObject *self, PyObject *args) {

	cmsHPROFILE hProfile;

	hProfile = cmsCreate_sRGBProfile();

	if(hProfile==NULL) {
		Py_INCREF(Py_None);
		return Py_None;
	}

    PyObject* capsule = PyCapsule_New((void *)hProfile,
                                       NULL,
                                       (void *)cmsCloseProfile);
	return Py_BuildValue("O", capsule);
}

static PyObject *
pycms_CreateLabProfile(PyObject *self, PyObject *args) {

	cmsHPROFILE hProfile;

	hProfile = cmsCreateLab4Profile(0);

	if(hProfile==NULL) {
		Py_INCREF(Py_None);
		return Py_None;
	}
    PyObject* capsule = PyCapsule_New((void *)hProfile,
                                       NULL,
                                       (void *)cmsCloseProfile);
	return Py_BuildValue("O", capsule);
}

static PyObject *
pycms_CreateGrayProfile(PyObject *self, PyObject *args) {

	cmsHPROFILE hProfile;
	cmsToneCurve *tonecurve;

	tonecurve = cmsBuildGamma(NULL, 2.2);
	hProfile = cmsCreateGrayProfile(0, tonecurve);
	cmsFreeToneCurve(tonecurve);

	if(hProfile==NULL) {
		Py_INCREF(Py_None);
		return Py_None;
	}

    PyObject* capsule = PyCapsule_New((void *)hProfile,
                                       NULL,
                                       (void *)cmsCloseProfile);
	return Py_BuildValue("O", capsule);
}

static PyObject *
pycms_CreateXYZProfile(PyObject *self, PyObject *args) {

	cmsHPROFILE hProfile;

	hProfile = cmsCreateXYZProfile();

	if(hProfile==NULL) {
		Py_INCREF(Py_None);
		return Py_None;
	}

    PyObject* capsule = PyCapsule_New((void *)hProfile,
                                   NULL,
                                   (void *)cmsCloseProfile);
	return Py_BuildValue("O", capsule);
}

static PyObject *
pycms_BuildTransform (PyObject *self, PyObject *args) {

	char *inMode;
	char *outMode;
	int renderingIntent;
	int inFlags;
	cmsUInt32Number flags;
	void *inputProfile;
	void *outputProfile;
	cmsHPROFILE hInputProfile, hOutputProfile;
	cmsHTRANSFORM hTransform;

	if (!PyArg_ParseTuple(args, "OsOsii", &inputProfile, &inMode, &outputProfile, &outMode, &renderingIntent, &inFlags)) {
		Py_INCREF(Py_None);
		return Py_None;
	}

	hInputProfile = (cmsHPROFILE) PyCapsule_GetPointer(inputProfile, NULL);
	hOutputProfile = (cmsHPROFILE) PyCapsule_GetPointer(outputProfile, NULL);

	flags = (cmsUInt32Number) inFlags;

	hTransform = cmsCreateTransform(hInputProfile, getLCMStype(inMode),
			hOutputProfile, getLCMStype(outMode), renderingIntent, flags);

	if(hTransform==NULL) {
		Py_INCREF(Py_None);
		return Py_None;
	}

    PyObject* capsule = PyCapsule_New((void *)hTransform,
                               NULL,
                               (void *)hTransform);
	return Py_BuildValue("O", capsule);
}

static PyObject *
pycms_BuildProofingTransform (PyObject *self, PyObject *args) {

	char *inMode;
	char *outMode;
	int renderingIntent;
	int proofingIntent;
	int inFlags;
	cmsUInt32Number flags;
	void *inputProfile;
	void *outputProfile;
	void *proofingProfile;

	cmsHPROFILE hInputProfile, hOutputProfile, hProofingProfile;
	cmsHTRANSFORM hTransform;

	if (!PyArg_ParseTuple(args, "OsOsOiii", &inputProfile, &inMode, &outputProfile, &outMode,
			&proofingProfile, &renderingIntent, &proofingIntent, &inFlags)) {
		Py_INCREF(Py_None);
		return Py_None;
	}

	hInputProfile = (cmsHPROFILE) PyCapsule_GetPointer(inputProfile, NULL);
	hOutputProfile = (cmsHPROFILE) PyCapsule_GetPointer(outputProfile, NULL);
	hProofingProfile = (cmsHPROFILE) PyCapsule_GetPointer(proofingProfile, NULL);
	flags = (cmsUInt32Number) inFlags;

	hTransform = cmsCreateProofingTransform(hInputProfile, getLCMStype(inMode),
			hOutputProfile, getLCMStype(outMode), hProofingProfile, renderingIntent, proofingIntent, flags);

	if(hTransform==NULL) {
		Py_INCREF(Py_None);
		return Py_None;
	}

    PyObject* capsule = PyCapsule_New((void *)hTransform,
                               NULL,
                               (void *)hTransform);
	return Py_BuildValue("O", capsule);
}

#define COLOR_BYTE 0
#define COLOR_WORD 1
#define COLOR_DBL 2

static PyObject *
pycms_TransformPixel (PyObject *self, PyObject *args) {

	unsigned char *inbuf=malloc(4);
	unsigned short *outbuf=malloc(8);
	double *d_outbuf=malloc(32);
	int channel1,channel2,channel3,channel4,out_type;
	void *transform;
	cmsHTRANSFORM hTransform;
	PyObject *result;

	if (!PyArg_ParseTuple(args, "Oiiiii", &transform, &channel1,
			&channel2, &channel3, &channel4, &out_type)) {
		free(inbuf);
		free(outbuf);
		free(d_outbuf);
		Py_INCREF(Py_None);
		return Py_None;
	}

	inbuf[0]=(unsigned char)channel1;
	inbuf[1]=(unsigned char)channel2;
	inbuf[2]=(unsigned char)channel3;
	inbuf[3]=(unsigned char)channel4;

	hTransform = (cmsHTRANSFORM) PyCapsule_GetPointer(transform, NULL);

	if(out_type==COLOR_WORD){
		cmsDoTransform(hTransform, inbuf, outbuf, 1);
		result = Py_BuildValue("[iiii]", outbuf[0], outbuf[1], outbuf[2], outbuf[3]);
	}else if(out_type==COLOR_DBL){
		cmsDoTransform(hTransform, inbuf, d_outbuf, 1);
		result = Py_BuildValue("[dddd]", d_outbuf[0], d_outbuf[1], d_outbuf[2], d_outbuf[3]);
	}else{
		cmsDoTransform(hTransform, inbuf, inbuf, 1);
		result = Py_BuildValue("[iiii]", inbuf[0], inbuf[1], inbuf[2], inbuf[3]);
	}

	free(inbuf);
	free(outbuf);
	free(d_outbuf);

	return result;
}

static PyObject *
pycms_TransformPixel16b (PyObject *self, PyObject *args) {

	unsigned short *inbuf=malloc(8);
	unsigned char *outbuf=malloc(4);
	double *d_outbuf=malloc(32);
	int channel1,channel2,channel3,channel4,out_type;
	void *transform;
	cmsHTRANSFORM hTransform;
	PyObject *result;

	if (!PyArg_ParseTuple(args, "Oiiiii", &transform, &channel1,
			&channel2, &channel3, &channel4, &out_type)) {
		free(inbuf);
		free(outbuf);
		free(d_outbuf);
		Py_INCREF(Py_None);
		return Py_None;
	}

	inbuf[0]=(unsigned short)channel1;
	inbuf[1]=(unsigned short)channel2;
	inbuf[2]=(unsigned short)channel3;
	inbuf[3]=(unsigned short)channel4;

	hTransform = (cmsHTRANSFORM) PyCapsule_GetPointer(transform, NULL);

	if(out_type==COLOR_BYTE){
		cmsDoTransform(hTransform, inbuf, outbuf, 1);
		result = Py_BuildValue("[iiii]", outbuf[0], outbuf[1], outbuf[2], outbuf[3]);
	}else if(out_type==COLOR_DBL){
		cmsDoTransform(hTransform, inbuf, d_outbuf, 1);
		result = Py_BuildValue("[dddd]", d_outbuf[0], d_outbuf[1], d_outbuf[2], d_outbuf[3]);
	}else{
		cmsDoTransform(hTransform, inbuf, inbuf, 1);
		result = Py_BuildValue("[iiii]", inbuf[0], inbuf[1], inbuf[2], inbuf[3]);
	}

	free(inbuf);
	free(outbuf);
	free(d_outbuf);

	return result;
}

static PyObject *
pycms_TransformPixelDbl (PyObject *self, PyObject *args) {

	double *inbuf=malloc(32);
	unsigned char *c_outbuf=malloc(4);
	unsigned short *outbuf=malloc(8);
	int out_type;
	void *transform;
	cmsHTRANSFORM hTransform;
	PyObject *result;

	if (!PyArg_ParseTuple(args, "Oddddi", &transform, &inbuf[0],
			&inbuf[1], &inbuf[2], &inbuf[3], &out_type)) {
		free(inbuf);
		free(outbuf);
		free(c_outbuf);
		Py_INCREF(Py_None);
		return Py_None;
	}

	hTransform = (cmsHTRANSFORM) PyCapsule_GetPointer(transform, NULL);

	if(out_type==COLOR_WORD){
		cmsDoTransform(hTransform, inbuf, outbuf, 1);
		result = Py_BuildValue("[iiii]", outbuf[0], outbuf[1], outbuf[2], outbuf[3]);
	}else if(out_type==COLOR_BYTE){
		cmsDoTransform(hTransform, inbuf, c_outbuf, 1);
		result = Py_BuildValue("[iiii]", c_outbuf[0], c_outbuf[1], c_outbuf[2], c_outbuf[3]);
	}else{
		cmsDoTransform(hTransform, inbuf, inbuf, 1);
		result = Py_BuildValue("[dddd]", inbuf[0], inbuf[1], inbuf[2], inbuf[3]);
	}

	free(inbuf);
	free(outbuf);
	free(c_outbuf);

	return result;
}



static PyObject *
pycms_GetProfileName (PyObject *self, PyObject *args) {

	void *profile;
	cmsHPROFILE hProfile;
	char *buffer;
	PyObject *ret;

	if (!PyArg_ParseTuple(args, "O", &profile)) {
		Py_INCREF(Py_None);
		return Py_None;
	}

	buffer=malloc(BUFFER_SIZE);
	hProfile = (cmsHPROFILE) PyCapsule_GetPointer(profile, NULL);

	cmsUInt32Number info_size;
	info_size = cmsGetProfileInfoUTF8(hProfile,
			cmsInfoDescription,
			cmsNoLanguage, cmsNoCountry,
			buffer, BUFFER_SIZE);

    if (info_size == 0)
        buffer[0] = 0x0;

	ret=Py_BuildValue("s", buffer);
	free(buffer);
	return ret;
}

static PyObject *
pycms_GetProfileInfo (PyObject *self, PyObject *args) {

	void *profile;
	cmsHPROFILE hProfile;
	char *buffer;
	PyObject *ret;

	if (!PyArg_ParseTuple(args, "O", &profile)) {
		Py_INCREF(Py_None);
		return Py_None;
	}

	buffer=malloc(BUFFER_SIZE);
	hProfile = (cmsHPROFILE) PyCapsule_GetPointer(profile, NULL);

	cmsUInt32Number info_size;
	info_size = cmsGetProfileInfoUTF8(hProfile,
			cmsInfoModel,
			cmsNoLanguage, cmsNoCountry,
			buffer, BUFFER_SIZE);

    if (info_size == 0)
        buffer[0] = 0x0;

	ret=Py_BuildValue("s", buffer);

	free(buffer);
	return ret;
}

static PyObject *
pycms_GetProfileInfoCopyright (PyObject *self, PyObject *args) {

	void *profile;
	cmsHPROFILE hProfile;
	char *buffer;
	PyObject *ret;

	if (!PyArg_ParseTuple(args, "O", &profile)) {
		Py_INCREF(Py_None);
		return Py_None;
	}

	buffer=malloc(BUFFER_SIZE);
	hProfile = (cmsHPROFILE) PyCapsule_GetPointer(profile, NULL);

	cmsUInt32Number info_size;
	info_size = cmsGetProfileInfoUTF8(hProfile,
			cmsInfoCopyright,
			cmsNoLanguage, cmsNoCountry,
			buffer, BUFFER_SIZE);

    if (info_size == 0)
        buffer[0] = 0x0;

	ret=Py_BuildValue("s", buffer);

	free(buffer);
	return ret;
}

static PyObject *
pycms_GetVersion (PyObject *self, PyObject *args) {
	return Py_BuildValue("i",  LCMS_VERSION);
}

static PyObject* apply_transform(PyObject *self, PyObject *args) {
	PyObject *transform_obj;
	PyObject *src_obj;

	if (!PyArg_ParseTuple(args, "OO", &transform_obj, &src_obj)) {
		Py_INCREF(Py_None);
		return Py_None;
	}

	_cmsTRANSFORM* transform;
	transform = (_cmsTRANSFORM*) PyCapsule_GetPointer(transform_obj, NULL);
    return Py_BuildValue("[i, i, i, i]", transform->InputFormat,
    transform->OutputFormat, TYPE_XYZ_DBL, TYPE_RGBA_8);
}

static
PyMethodDef pycms_methods[] = {
    {"create_profile", create_profile, METH_VARARGS},
    {"open_profile", open_profile, METH_VARARGS},
    {"apply_transform", apply_transform, METH_VARARGS},
	{"getVersion", pycms_GetVersion, METH_VARARGS},
	{"openProfile", pycms_OpenProfile, METH_VARARGS},
	{"createRGBProfile", pycms_CreateRGBProfile, METH_VARARGS},
	{"createLabProfile", pycms_CreateLabProfile, METH_VARARGS},
	{"createGrayProfile", pycms_CreateGrayProfile, METH_VARARGS},
	{"createXYZProfile", pycms_CreateXYZProfile, METH_VARARGS},
	{"buildTransform", pycms_BuildTransform, METH_VARARGS},
	{"buildProofingTransform", pycms_BuildProofingTransform, METH_VARARGS},
	{"transformPixel", pycms_TransformPixel, METH_VARARGS},
	{"transformPixel16b", pycms_TransformPixel16b, METH_VARARGS},
	{"transformPixelDbl", pycms_TransformPixelDbl, METH_VARARGS},
	{"getProfileName", pycms_GetProfileName, METH_VARARGS},
	{"getProfileInfo", pycms_GetProfileInfo, METH_VARARGS},
	{"getProfileInfoCopyright", pycms_GetProfileInfoCopyright, METH_VARARGS},
	{NULL, NULL}
};


static struct PyModuleDef lcms2_def =
{
    PyModuleDef_HEAD_INIT,
    "_lcms2",     /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    pycms_methods
};

PyMODINIT_FUNC PyInit__lcms2(void)
{
    if (PyType_Ready(&profile_type) < 0)
        return NULL;

    PyObject *m;
    m = PyModule_Create(&lcms2_def);
    if (m == NULL)
        return NULL;

    if (PyModule_AddObjectRef(m, "Profile", (PyObject *) &profile_type) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    import_array(); // initialize  NumPy

    return m;
}
