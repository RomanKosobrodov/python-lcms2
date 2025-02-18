#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <stddef.h> // offsetof
#include <lcms2.h>

typedef struct
{
    PyObject_HEAD int input_format;
    int output_format;
    int rendering_intent;
    int flags;
    PyObject *handle;
} transform_object;

static void
free_transform_handle(PyObject *capsule)
{
    cmsHTRANSFORM transform = (cmsHPROFILE)PyCapsule_GetPointer(capsule, NULL);
    if (transform != NULL)
    {
        cmsDeleteTransform(transform);
    }
}

static void
transform_dealloc(transform_object *self)
{
    Py_XDECREF(self->handle);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
transform_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    transform_object *self;
    self = (transform_object *)type->tp_alloc(type, 0);

    if (self == NULL)
    {
        return NULL;
    }

    self->input_format = -1;
    self->output_format = -1;
    self->rendering_intent = -1;
    self->flags = -1;

    self->handle = Py_BuildValue("");
    if (self->handle == NULL)
    {
        Py_DECREF(self);
        return NULL;
    }

    return (PyObject *)self;
}

static int
transform_init(transform_object *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"src_profile", "src_type", "dst_profile", "dst_type",
                             "rendering_intent", "flags", NULL};
    PyObject *src_profile = NULL;
    PyObject *dst_profile = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OiOiii", kwlist,
                                     &src_profile, &self->input_format,
                                     &dst_profile, &self->output_format,
                                     &self->rendering_intent,
                                     &self->flags))
    {                                 
        return -1;
    }

    if (PyObject_HasAttrString(src_profile, "handle") == 0 ||
        PyObject_HasAttrString(dst_profile, "handle") == 0)
    {
        return -1;
    }

    PyObject *src_handle = PyObject_GetAttrString(src_profile, "handle");
    PyObject *dst_handle = PyObject_GetAttrString(dst_profile, "handle");
    if (src_handle == NULL || dst_handle == NULL)
    {
        return -1;
    }

    cmsHPROFILE cms_src_profile = (cmsHPROFILE)PyCapsule_GetPointer(src_handle, NULL);
    cmsHPROFILE cms_dst_profile = (cmsHPROFILE)PyCapsule_GetPointer(dst_handle, NULL);
    if (cms_src_profile == NULL || cms_dst_profile == NULL)
    {
        return -1;
    }

    cmsHTRANSFORM transform_handle = cmsCreateTransform(cms_src_profile, self->input_format,
                                                        cms_dst_profile, self->output_format,
                                                        self->rendering_intent,
                                                        self->flags);

    if (transform_handle == NULL)
    {
        return -1;
    }

    self->handle = PyCapsule_New(transform_handle,
                                 NULL,
                                 free_transform_handle);

    return 0;
}

static PyMemberDef transform_members[] = {
    {"input_format", Py_T_INT, offsetof(transform_object, input_format), 0, "input data format"},
    {"output_format", Py_T_INT, offsetof(transform_object, output_format), 0, "output data format"},
    {"rendering_intent", Py_T_INT, offsetof(transform_object, rendering_intent), 0, "rendering intent"},
    {"flags", Py_T_INT, offsetof(transform_object, flags), 0, "transform flags"},
    {"handle", Py_T_OBJECT_EX, offsetof(transform_object, handle), 0, "Little CMS2 transform handle"},
    {NULL} /* Sentinel */
};



static PyObject *
apply(transform_object *self, PyObject *args, PyObject *kwds)
{
    PyObject *src = NULL;
    PyObject *dst = NULL;
    int num_pixels = 0;
    if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &src,
        &PyArray_Type, &dst, &num_pixels))
    {
	    PyErr_SetString(PyExc_ValueError, "Invalid argument - expected a numpy array");        
        return NULL;
    }
    
    if (src == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "Unable to parse input array");        
        return NULL;
    }
    
    if (dst == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "Unable to parse output array");        
        return NULL;
    }

    if (PyArray_SIZE(self) == 0) 
    {
        return Py_BuildValue("");
    }

    _cmsTRANSFORM* transform = (_cmsTRANSFORM*) PyCapsule_GetPointer(self->handle, NULL);
    if (transform == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "Unable to retrieve transform handle");        
        return NULL;        
    }

    PyArray_ENABLEFLAGS(dst, NPY_ARRAY_CARRAY);

    int array_flags = PyArray_FLAGS(src);
    if (array_flags & NPY_ARRAY_ALIGNED != 0 &&
        array_flags & NPY_ARRAY_C_CONTIGUOUS != 0)
    {        
        cmsDoTransform( transform, 
                        PyArray_DATA(src),
                        PyArray_DATA(dst),
                        num_pixels);
        return Py_BuildValue("");
    }
    else
    {
        NpyIter* iter = NpyIter_New(src, NPY_ITER_READONLY|
                                NPY_ITER_EXTERNAL_LOOP|
                                NPY_ITER_REFS_OK,
                            NPY_KEEPORDER, NPY_NO_CASTING,
                            NULL);
        if (iter == NULL) 
        {
            PyErr_SetString(PyExc_ValueError, "Unable to create input iterator");   
            return NULL;
        }
    
        NpyIter_IterNextFunc *iter_next = NpyIter_GetIterNext(iter, NULL);
        if (iter_next == NULL) {
            NpyIter_Deallocate(iter);
            PyErr_SetString(PyExc_ValueError, "Unable to create next iteration function");   
            return NULL;
        }

        char** data_ptr = NpyIter_GetDataPtrArray(iter);
        npy_intp* stride_ptr = NpyIter_GetInnerStrideArray(iter);
        npy_intp* inner_size_ptr = NpyIter_GetInnerLoopSizePtr(iter);

        npy_intp num_src_channels = PyArray_Size(src) / num_pixels;
        char* dst_ptr = (char *)PyArray_DATA(dst);
        npy_intp num_dst_channels = PyArray_Size(dst) / num_pixels;
        npy_intp step_bytes = num_dst_channels * PyArray_ITEMSIZE(dst);
        
        do 
        {
            char* data = *data_ptr;
            npy_intp stride = *stride_ptr;
            npy_intp count = *inner_size_ptr;

            if (count % num_src_channels == 0 && stride == 1)
            {
                npy_intp num_samples = count / num_src_channels;
                cmsDoTransform(transform, 
                                data,
                                dst_ptr,
                                num_samples);
                dst_ptr += num_samples * step_bytes;    
            }
            else
            {
                NpyIter_Deallocate(iter);
                PyErr_SetString(PyExc_ValueError, "Error iterating the input array: wrong data layout");   
                return NULL;                
            }
  
        } while(iter_next(iter));
    
        NpyIter_Deallocate(iter);

        return Py_BuildValue("");
    }
}


static PyMethodDef transform_methods[] = {
    {"apply", (PyCFunction) apply, METH_VARARGS,
        "apply transform to a source array and store the results in the destination array"},
    {NULL} /* Sentinel */
};

static PyTypeObject transform_type = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
                   .tp_name = "pylcms2.Transform",
    .tp_doc = PyDoc_STR("Little CMS2 transform"),
    .tp_basicsize = sizeof(transform_object),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = transform_new,
    .tp_init = (initproc)transform_init,
    .tp_dealloc = (destructor)transform_dealloc,
    .tp_members = transform_members,
    .tp_methods = transform_methods,
};