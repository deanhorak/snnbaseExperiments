# Contributing

Use a focused branch and keep each pull request limited to one experiment or infrastructure change.
New experiments need a stated hypothesis, documented dataset provenance, repeatable commands,
machine-readable or clearly tabulated metrics, and tests that do not require a network connection.

Before opening a pull request:

```sh
cmake --preset default
cmake --build --preset default
ctest --preset default
```

Format C++ changes with `clang-format` using the repository configuration. Do not commit datasets,
build output, credentials, or results that cannot be traced to an exact configuration. By
contributing, you agree that your contribution is licensed under the MIT License.

