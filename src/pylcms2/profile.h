#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <stddef.h> // for offsetof()

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