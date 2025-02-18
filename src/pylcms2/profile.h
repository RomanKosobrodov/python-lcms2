#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <stddef.h> // offsetof

#include <lcms2.h>
#include <lcms2_internal.h>

typedef struct {
    PyObject_HEAD
    PyObject *name;
    PyObject *info;
    PyObject *copyright;
    PyObject *handle;
} profile_object;

static void
profile_dealloc(profile_object *self)
{
    Py_XDECREF(self->name);
    Py_XDECREF(self->info);
    Py_XDECREF(self->copyright);
    Py_XDECREF(self->handle);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *
profile_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    profile_object *self;
    self = (profile_object *) type->tp_alloc(type, 0);

    if (self == NULL) {
        return NULL;
    }

    self->name = PyUnicode_FromString("");
    if (self->name == NULL) {
        Py_DECREF(self);
        return NULL;
    }
    self->info = PyUnicode_FromString("");
    if (self->info == NULL) {
        Py_DECREF(self);
        return NULL;
    }
    self->copyright = PyUnicode_FromString("");
    if (self->copyright == NULL) {
        Py_DECREF(self);
        return NULL;
    }
    self->handle = Py_BuildValue("");
    if (self->handle == NULL) {
        Py_DECREF(self);
        return NULL;
    }

    return (PyObject *) self;
}

static int
profile_init(profile_object *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"name", "info", "copyright", "handle", NULL};
    PyObject *name = NULL;
    PyObject *info = NULL;
    PyObject *copyright = NULL;
    PyObject *handle = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOO", kwlist,
                                     &name, &info, &copyright, &handle))
        return -1;

    if (name) {
        Py_XSETREF(self->name, Py_NewRef(name));
    }
    if (info) {
        Py_XSETREF(self->info, Py_NewRef(info));
    }
    if (copyright) {
        Py_XSETREF(self->copyright, Py_NewRef(copyright));
    }
    if (handle) {
        Py_XSETREF(self->handle, Py_NewRef(handle));
    }

    return 0;
}

static PyMemberDef profile_members[] = {
    {"name", Py_T_OBJECT_EX, offsetof(profile_object, name), 0, "profile name"},
    {"info", Py_T_OBJECT_EX, offsetof(profile_object, info), 0, "profile information"},
    {"copyright", Py_T_OBJECT_EX, offsetof(profile_object, copyright), 0, "copyright information"},
    {"handle", Py_T_OBJECT_EX, offsetof(profile_object, handle), 0, "Little CMS2 profile handle"},
    {NULL}  /* Sentinel */
};

static PyMethodDef profile_methods[] = {
    {NULL}  /* Sentinel */
};

static PyTypeObject profile_type = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pylcms2.Profile",
    .tp_doc = PyDoc_STR("Little CMS2 profile"),
    .tp_basicsize = sizeof(profile_object),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = profile_new,
    .tp_init = (initproc) profile_init,
    .tp_dealloc = (destructor) profile_dealloc,
    .tp_members = profile_members,
    .tp_methods = profile_methods,
};



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

static PyObject *
profile_from_memory(PyObject *self, PyObject *args)
{
    PyBytesObject *buffer = NULL;
	if (!PyArg_ParseTuple(args, "S",  &buffer)){
	    PyErr_SetString(PyExc_ValueError, "Invalid argument - expected a bytes object");
		return NULL;
	}

    if (buffer == NULL)
    {
	    PyErr_SetString(PyExc_ValueError, "Unable to parse input parameter as a bytes-like object");
		return NULL;
    }
    uint32_t size = (uint32_t) PyBytes_GET_SIZE(buffer);
    char* ptr = PyBytes_AsString(buffer);
    cmsHPROFILE profile_handle = cmsOpenProfileFromMem(ptr, size);
    return (PyObject*) create_profile_from_handle(profile_handle);   
}


static PyObject *
profile_to_bytes(PyObject *self, PyObject *args)
{
    PyObject *profile = NULL;

    if (!PyArg_ParseTuple(args, "O", &profile))
    {                                 
        PyErr_SetString(PyExc_ValueError, "Unable to parse input argument");
        return NULL;
    }

    if (PyObject_HasAttrString(profile, "handle") == 0)
    {
        PyErr_SetString(PyExc_ValueError, "Invalid profile: handle member is missing");
        return NULL;
    }

    PyObject *handle = PyObject_GetAttrString(profile, "handle");
    if (handle == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "Invalid profile: error accessing the handle member");
        return NULL;
    }

    cmsHPROFILE cms_profile = (cmsHPROFILE)PyCapsule_GetPointer(handle, NULL);
    if (cms_profile == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "Invalid profile: error retrieving the underlying Little CMS profile");
        return NULL;    
    }

    int success;
    uint32_t size_bytes;
    success = cmsSaveProfileToMem(cms_profile, NULL, &size_bytes);
    if (!success)
    {
        PyErr_SetString(PyExc_ValueError, "Unable to calculate size of the underlying Little CMS profile");
        return NULL;            
    }

    PyObject* buffer = PyBytes_FromStringAndSize(NULL, size_bytes);
    if (buffer == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "Unable to create buffer for the result");
        return NULL;            
    }
    char *ptr = PyBytes_AsString(buffer);
    success = cmsSaveProfileToMem(cms_profile, ptr, &size_bytes);
    if (!success)
    {
        PyErr_SetString(PyExc_ValueError, "Unable to copy profile to the buffer");
        return NULL;            
    }

    return buffer;
}