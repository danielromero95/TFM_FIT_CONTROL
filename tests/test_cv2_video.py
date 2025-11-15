"""Comprobación manual del entorno de OpenCV.

Este módulo no forma parte de la batería automática de pytest; únicamente sirve
para que podamos ejecutar ``python tests/test_cv2_video.py`` y verificar la
configuración multimedia local cuando sea necesario."""

if __name__ != "__main__":
    import pytest

    pytest.skip(
        "Manual OpenCV environment check, not part of the automated suite.",
        allow_module_level=True,
    )


def _describe_environment() -> None:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - diagnóstico manual
        print(f"cv2 import failed: {exc}")
        return

    version = getattr(cv2, "__version__", "<unknown>")
    build_info = getattr(cv2, "getBuildInformation", lambda: "(build info unavailable)")
    print(f"cv2 version: {version}")
    try:
        print(build_info())
    except Exception as exc:  # pragma: no cover - diagnóstico manual
        print(f"Could not obtain build information: {exc}")


if __name__ == "__main__":  # pragma: no cover - ejecución manual
    _describe_environment()
