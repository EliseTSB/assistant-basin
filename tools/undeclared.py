#!/usr/bin/env python3
"""undeclared.py FILE...: list names a Python file reads that nothing declares, without running it.

Measures: names the compiler treats as global, that are not module-level, not builtins. It reads a path no test reaches,
such as the real-model branch beside a stub. Written after Part 126 (2026-10-30), where a helper read MODEL_ID and
REVISION, which were imported inside another function, and the job died on the A770 while 9 stub tests stayed green.
It cannot see an attribute error, a wrong argument, or a name set by `import *`. Exit 1 if it finds any.
"""
import builtins
import symtable
import sys


def undeclared(path):
    table = symtable.symtable(open(path, encoding="utf-8").read(), path, "exec")
    module = {symbol.get_name() for symbol in table.get_symbols()}
    found = []

    def walk(scope):
        for symbol in scope.get_symbols():
            name = symbol.get_name()
            if symbol.is_global() and symbol.is_referenced() and name not in module and not hasattr(builtins, name):
                found.append((scope.get_name(), name))
        for child in scope.get_children():
            walk(child)

    walk(table)
    return found


if __name__ == "__main__":
    bad = 0
    for file_path in sys.argv[1:]:
        for scope, name in undeclared(file_path):
            print(f"{file_path}: {scope}: {name}")
            bad += 1
    sys.exit(1 if bad else 0)
