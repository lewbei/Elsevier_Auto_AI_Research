import sys
import compileall


def main() -> int:
    ok = compileall.compile_dir(".", force=True, quiet=1)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

