"""Atomic text writes: round-trip, overwrite, and no leftover temp files."""

from metaextract.fileio import atomic_write_text


def test_writes_and_reads_back(tmp_path):
    p = tmp_path / "x.json"
    atomic_write_text(p, '{"a": 1}')
    assert p.read_text(encoding="utf-8") == '{"a": 1}'


def test_overwrites_in_place(tmp_path):
    p = tmp_path / "x.txt"
    atomic_write_text(p, "first")
    atomic_write_text(p, "second")
    assert p.read_text(encoding="utf-8") == "second"


def test_leaves_no_temp_file_behind(tmp_path):
    p = tmp_path / "x.txt"
    atomic_write_text(p, "data")
    # only the target should remain — the temp was renamed into place
    assert [f.name for f in tmp_path.iterdir()] == ["x.txt"]
