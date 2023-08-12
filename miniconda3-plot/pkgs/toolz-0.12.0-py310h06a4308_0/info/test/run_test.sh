

set -ex



pip check
pytest --doctest-modules --pyargs toolz
exit 0
