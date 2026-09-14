from __future__ import annotations

import importlib
import inspect
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# MODULES TO AUDIT
# ============================================================

MODULE_NAMES = [
    "engine.environment",
    "engine.possessions",
    "engine.scoring",
    "engine.monte_carlo",
]


# ============================================================
# DISPLAY HELPERS
# ============================================================

def section(title: str) -> None:

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def subsection(title: str) -> None:

    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


def safe_signature(
    obj: Any,
) -> str:

    try:
        return str(
            inspect.signature(obj)
        )

    except Exception as exc:
        return (
            f"<signature unavailable: "
            f"{type(exc).__name__}: {exc}>"
        )


def safe_source(
    obj: Any,
) -> str:

    try:
        return inspect.getsource(
            obj
        )

    except Exception as exc:
        return (
            "<source unavailable: "
            f"{type(exc).__name__}: {exc}>"
        )


def module_defined_classes(
    module: ModuleType,
) -> list[
    tuple[str, type]
]:

    classes: list[
        tuple[str, type]
    ] = []

    for name, obj in inspect.getmembers(
        module,
        inspect.isclass,
    ):

        if name.startswith("_"):
            continue

        # Only classes actually defined in this module.
        # This excludes imported pandas/numpy/dataclass classes.
        if obj.__module__ != module.__name__:
            continue

        classes.append(
            (
                name,
                obj,
            )
        )

    return classes


def public_methods(
    cls: type,
) -> list[
    tuple[str, Any]
]:

    methods: list[
        tuple[str, Any]
    ] = []

    for name, member in inspect.getmembers(
        cls,
    ):

        if name.startswith("_"):
            continue

        if not callable(member):
            continue

        methods.append(
            (
                name,
                member,
            )
        )

    return methods


def print_instance_attributes(
    instance: Any,
) -> None:

    if not hasattr(
        instance,
        "__dict__",
    ):

        print(
            "No instance __dict__."
        )

        return

    attributes = vars(
        instance
    )

    if not attributes:

        print(
            "No instance attributes yet."
        )

        return

    for key, value in sorted(
        attributes.items()
    ):

        print(
            f"{key:<38} "
            f"{type(value).__name__}"
        )


def print_dataclass_values(
    instance: Any,
) -> None:

    if not is_dataclass(
        instance
    ):

        return

    try:

        values = asdict(
            instance
        )

    except Exception:

        return

    for key, value in values.items():

        print(
            f"{key:<40} = {value}"
        )


# ============================================================
# CLASS AUDIT
# ============================================================

def audit_class(
    class_name: str,
    cls: type,
) -> None:

    subsection(
        f"CLASS: {class_name}"
    )

    print(
        f"Module:      {cls.__module__}"
    )

    print(
        f"Constructor: {class_name}"
        f"{safe_signature(cls)}"
    )

    # ========================================================
    # DATACLASS / CONFIG VALUES
    # ========================================================

    print()
    print(
        "DEFAULT INSTANCE"
    )

    instance = None

    try:

        instance = cls()

        print(
            "Instantiation: SUCCESS"
        )

        if is_dataclass(
            instance
        ):

            print()
            print(
                "DATACLASS VALUES"
            )

            print_dataclass_values(
                instance
            )

        else:

            print()
            print(
                "INSTANCE ATTRIBUTES"
            )

            print_instance_attributes(
                instance
            )

    except Exception as exc:

        print(
            "Instantiation: FAILED"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

    # ========================================================
    # PUBLIC METHODS
    # ========================================================

    print()
    print(
        "PUBLIC METHODS"
    )

    methods = public_methods(
        cls
    )

    if not methods:

        print(
            "No public callable methods."
        )

    else:

        for name, member in methods:

            print(
                f"{name}"
                f"{safe_signature(member)}"
            )

    # ========================================================
    # CLASS SOURCE
    # ========================================================

    print()
    print(
        "CLASS SOURCE"
    )

    print(
        safe_source(
            cls
        )
    )


# ============================================================
# MODULE AUDIT
# ============================================================

def audit_module(
    module_name: str,
    index: int,
) -> None:

    section(
        f"{index}. MODULE: {module_name}"
    )

    try:

        module = importlib.import_module(
            module_name
        )

    except Exception as exc:

        print(
            "IMPORT FAILED"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return

    print(
        "Import: SUCCESS"
    )

    try:

        module_file = inspect.getfile(
            module
        )

        print(
            f"File:   {module_file}"
        )

    except Exception:

        pass

    # ========================================================
    # PUBLIC SYMBOLS
    # ========================================================

    subsection(
        "PUBLIC MODULE SYMBOLS"
    )

    public_symbols = [
        name
        for name in dir(module)
        if not name.startswith("_")
    ]

    for name in public_symbols:

        try:

            obj = getattr(
                module,
                name,
            )

            print(
                f"{name:<40} "
                f"{type(obj).__name__}"
            )

        except Exception as exc:

            print(
                f"{name:<40} "
                f"<error: {exc}>"
            )

    # ========================================================
    # MODULE CLASSES
    # ========================================================

    classes = module_defined_classes(
        module
    )

    subsection(
        "CLASSES DEFINED IN MODULE"
    )

    if not classes:

        print(
            "No locally-defined public classes found."
        )

    else:

        for class_name, cls in classes:

            print(
                class_name
            )

    # ========================================================
    # DETAILED CLASS AUDIT
    # ========================================================

    for class_name, cls in classes:

        audit_class(
            class_name,
            cls,
        )

    # ========================================================
    # FULL MODULE SOURCE
    # ========================================================

    subsection(
        "FULL MODULE SOURCE"
    )

    print(
        safe_source(
            module
        )
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 "
        "— LIVE SCORING ENGINE DISCOVERY"
    )

    print(
        "Purpose:"
    )

    print(
        "Discover the exact production interfaces "
        "for the existing environment, possession, "
        "scoring and Monte Carlo modules."
    )

    print()
    print(
        "No class names are assumed."
    )

    print(
        "The audit discovers all public classes "
        "directly from each module."
    )

    print()
    print(
        "This job is READ ONLY."
    )

    print(
        "No CSV, database, model or prediction "
        "artifact will be modified."
    )

    # ========================================================
    # AUDIT MODULES
    # ========================================================

    for index, module_name in enumerate(
        MODULE_NAMES,
        start=1,
    ):

        audit_module(
            module_name=module_name,
            index=index,
        )

    # ========================================================
    # FINISH
    # ========================================================

    section(
        "DISCOVERY COMPLETE"
    )

    print(
        "✓ environment module inspected"
    )

    print(
        "✓ possessions module inspected"
    )

    print(
        "✓ scoring module inspected"
    )

    print(
        "✓ monte_carlo module inspected"
    )

    print()
    print(
        "NEXT:"
    )

    print(
        "Use these exact discovered interfaces "
        "to build the live 2026 scoring and "
        "Monte Carlo production pipeline."
    )


if __name__ == "__main__":
    main()