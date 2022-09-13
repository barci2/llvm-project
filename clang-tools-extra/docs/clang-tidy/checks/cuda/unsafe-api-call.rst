.. title:: clang-tidy - cuda-unsafe-api-call

cuda-unsafe-api-call
====================

Finds usages of CUDA API where the error returned by the API function is not
handled in any way. It has narrower specification of what it allows and what it
doesn't than the
:doc:`bugprone-unused-return-value <../bugprone/unused-return-value>`
check, which makes it useful for more applications.

Specification
-------------

A function is considered to be a part of CUDA API if:

- it is included in a file that ends with either ``cuda_runtime.h`` or
  ``cuda_runtime_wrapper.h`` (those headers are automatically included from the
  during CUDA code compilation)

- its return type is ``cudaError_t``

If a call to a function like that is made, it has to be used in another
statement, for example passed as a function argument, assigned to a variable or
used in an if statement. The only exception is passing the value to a macro,
which is considered a valid way to handle the error even if the macro does not
use the return value of the call (this is to allow dummy marker macros that
just pass the value through). It is recommended that a project-wide handler(s)
is set to handle such errors, but this is not a default requirement of the check
(though it can be set with the check's options).

Example:

.. code-block:: c++

  void foo() {
    cudaDeviceReset();
  }

results in the following warnings::

    1 warning generated when compiling for host.
    test.cu:2:3: warning: Unchecked CUDA API call. Consider adding logic to check if an error has been returned or specify the error handler for this project. [cuda-unsafe-api-call]
      cudaDeviceReset();
      ^

Options
-------

.. option:: HandlerName

    The name of the function or macro that should be used in fix it hints to
    handle the return value of an API call.

.. option:: AcceptedHandlers

    The list of handler functions or macros that are allowed for the specific
    project. If specified, the only valid way to handle the error returned
    by a CUDA API call is to pass it as one of the arguments to said handlers.
    If the HandlerName option is also specified then it will be implicitly
    added as one of the accepted handlers.
