.. title:: clang-tidy - cuda-unsafe-kernel-call

cuda-unsafe-kernel-call
=======================

Finds CUDA kernel calls which do not have any post-invocation error handling 
implemented for them. It expects to capture the error after the kernel 
invocation using a call to ``cudaGetLastError()``.

Specification
-------------

The check finds the declaration of ``cudaGetLastError()`` by checking that:

 - tt has the expected name

 - its return type is ``cudaError_t``

 - it is included in a file that ends with either ``cuda_runtime.h`` or
  ``cuda_runtime_wrapper.h`` (those headers are automatically included from the
  during CUDA code compilation)

The check then generates a Control Flow Graph for the program. To check that the
kernel call is error-handled in a valid way it expects that the first expression
tree or function call in the control flow graph, right after the kernel call, is
the call to ``cudaGetLastError()``. This call must also happen in a direct line
from the kernel call, i.e. no node on the path from the kernel call to call to
``cudaGetLastError()`` can have more than 1 incoming or outgoing control flow
branches.

Example:

.. code-block:: c++

  __global__
  void kernel();

  void foo() {
    kernel<<<64, 128>>>();
  }

results in the following warnings::

    1 warning generated when compiling for host.
      test.cu:5:3: warning: Possible unchecked error after a kernel launch. Try using `cudaGetLastError()` right after the kernel call to get the error or specify a project-wide kernel call error handler. [cuda-unsafe-kernel-call]
      kernel<<<64, 128>>>();
      ^

Options
-------

.. option:: HandlerName

    The name of the function or macro that should be used in fix it hints to
    check the error after a kernel invocation. It will be placed as the next
    statement after the kernel call. Even if it is a function call or a macro
    that does not call ``cudaGetLastError()``, it will be accepted as a valid
    way to handle a kernel call. If the specified handler is a function name
    then it can be scoped; however, for performance reasons, if the function
    name is scoped then it has to be its fully scoped name.

.. option:: AcceptedHandlers

    The list of handler functions or macros that are allowed for the specific
    project. Just like the handler specified in HandlerName, be it a macro or
    a function, they will also be allowed as a valid way to handle the kernel
    call even if they would not be accepted otherwise. If the specified handler
    is a function name then it can be scoped; however, for performance reasons,
    if the function name is scoped then it has to be its fully scoped name.
