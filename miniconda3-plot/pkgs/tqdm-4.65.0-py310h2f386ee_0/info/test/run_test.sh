

set -ex



pip check
tqdm --help
tqdm -v | rg 4.65.0
pytest -k "not tests_perf"
exit 0
