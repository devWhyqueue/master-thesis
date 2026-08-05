from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from imbalance_benchmark.datasets.tcga_ut import pack, source, squashfs
from imbalance_benchmark.datasets.tcga_ut.pack import (
    class_partial_done,
    combine_partials,
    materialize,
    read_class_partial,
    tree_hash,
    validate_manifest_cohort,
    write_class_partial,
)
from imbalance_benchmark.datasets.tcga_ut.source import (
    ZENODO_RECORD_ID,
    ZENODO_VERSION,
    extract_class_images,
    parse_zenodo_metadata,
    stage_nested_archive,
    validate_shared_archive,
)
from imbalance_benchmark.datasets.tcga_ut.squashfs import publish_squashfs


def _nested_zip_bytes(images: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in images.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _write_shared_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def _zenodo_files(members: dict[str, bytes]) -> list[dict[str, object]]:
    return [
        {
            "key": name,
            "size": len(data),
            "md5": hashlib.md5(data).hexdigest(),
            "url": f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{name}",
        }
        for name, data in members.items()
    ]


def test_parse_zenodo_metadata_locks_version_and_file_count() -> None:
    payload = {
        "metadata": {"version": "1.0"},
        "files": [
            {"key": f"{i}.zip", "size": 1, "checksum": "md5:" + "0" * 32}
            for i in range(33)
        ],
    }

    metadata = parse_zenodo_metadata(payload)

    assert metadata == {
        "record_id": ZENODO_RECORD_ID,
        "version": "1.0",
        "files": [
            {"key": f"{i}.zip", "size": 1, "md5": "0" * 32, "url": ""} for i in range(33)
        ],
    }
    with pytest.raises(ValueError, match="version"):
        parse_zenodo_metadata({**payload, "metadata": {"version": "2.0"}})
    with pytest.raises(ValueError, match="Expected 33"):
        parse_zenodo_metadata({**payload, "files": payload["files"][:5]})


def test_validate_shared_archive_matches_official_file_list(tmp_path: Path) -> None:
    members = {"ClassA.zip": b"aaaa", "LICENSE": b"license-text"}
    shared_zip = tmp_path / "data_raw.zip"
    _write_shared_zip(shared_zip, members)
    files = _zenodo_files(members)

    matched = validate_shared_archive(shared_zip, files)

    assert set(matched) == {"ClassA.zip", "LICENSE"}


def test_validate_shared_archive_rejects_size_mismatch(tmp_path: Path) -> None:
    members = {"ClassA.zip": b"aaaa"}
    shared_zip = tmp_path / "data_raw.zip"
    _write_shared_zip(shared_zip, members)
    files = _zenodo_files(members)
    files[0]["size"] = 999

    with pytest.raises(ValueError, match="differ from Zenodo metadata"):
        validate_shared_archive(shared_zip, files)


def test_stage_nested_archive_extracts_verified_member(tmp_path: Path) -> None:
    members = {"ClassA.zip": _nested_zip_bytes({"a.jpg": b"img"})}
    shared_zip = tmp_path / "data_raw.zip"
    _write_shared_zip(shared_zip, members)
    files = _zenodo_files(members)
    matched = validate_shared_archive(shared_zip, files)

    extracted = stage_nested_archive(
        shared_zip, matched["ClassA.zip"], str(files[0]["md5"]), tmp_path / "scratch"
    )

    assert extracted.read_bytes() == members["ClassA.zip"]


def test_stage_nested_archive_rejects_md5_mismatch(tmp_path: Path) -> None:
    members = {"ClassA.zip": _nested_zip_bytes({"a.jpg": b"img"})}
    shared_zip = tmp_path / "data_raw.zip"
    _write_shared_zip(shared_zip, members)
    files = _zenodo_files(members)
    matched = validate_shared_archive(shared_zip, files)

    with pytest.raises(ValueError, match="MD5 differs"):
        stage_nested_archive(shared_zip, matched["ClassA.zip"], "0" * 32, tmp_path / "scratch")


def test_extract_class_images_hashes_every_jpg_sorted(tmp_path: Path) -> None:
    nested = tmp_path / "ClassA.zip"
    with zipfile.ZipFile(nested, "w") as archive:
        archive.writestr("b.jpg", b"second")
        archive.writestr("a.jpg", b"first")
        archive.writestr("readme.txt", b"ignore me")

    rows = extract_class_images(nested, "ClassA", tmp_path / "images")

    assert [row[0] for row in rows] == ["ClassA/a.jpg", "ClassA/b.jpg"]
    assert rows[0][1] == len(b"first")
    assert rows[0][2] == hashlib.sha256(b"first").hexdigest()


def test_class_partial_supports_resume(tmp_path: Path) -> None:
    partials = tmp_path / "partials"
    rows = [("A/a.jpg", 4, "hash1")]
    write_class_partial(partials, "A", rows)

    assert class_partial_done(partials, "A") is True
    assert read_class_partial(partials, "A") == rows

    (partials / "A.jsonl.sha256").unlink()

    assert class_partial_done(partials, "A") is False


def test_combine_partials_produces_deterministic_sorted_manifest(tmp_path: Path) -> None:
    partials = tmp_path / "partials"
    write_class_partial(partials, "B", [("B/2.jpg", 2, "h2"), ("B/1.jpg", 1, "h1")])
    write_class_partial(partials, "A", [("A/1.jpg", 1, "ha")])
    manifest_path = tmp_path / "manifest.jsonl"

    rows = combine_partials(partials, ["A", "B"], manifest_path)

    assert [row[0] for row in rows] == ["A/1.jpg", "B/1.jpg", "B/2.jpg"]
    assert len(manifest_path.read_text(encoding="utf-8").splitlines()) == 3
    assert (tmp_path / "manifest.jsonl.sha256").is_file()


def test_tree_hash_is_order_independent() -> None:
    rows = [("A/1.jpg", 1, "ha"), ("B/1.jpg", 1, "hb")]

    assert tree_hash(rows) == tree_hash(list(reversed(rows)))
    assert tree_hash(rows) != tree_hash([("A/1.jpg", 1, "different")])


def test_validate_manifest_cohort_rejects_wrong_image_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pack, "EXPECTED_IMAGE_COUNT", 2)
    monkeypatch.setattr(pack, "EXPECTED_CLASS_COUNT", 1)

    with pytest.raises(ValueError, match="images"):
        validate_manifest_cohort([("A/1.jpg", 1, "h1")], ["A"])


def test_validate_manifest_cohort_rejects_wrong_class_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pack, "EXPECTED_IMAGE_COUNT", 1)
    monkeypatch.setattr(pack, "EXPECTED_CLASS_COUNT", 2)

    with pytest.raises(ValueError, match="classes"):
        validate_manifest_cohort([("A/1.jpg", 1, "h1")], ["A"])


def test_publish_squashfs_atomically_renames(tmp_path: Path) -> None:
    partial = tmp_path / "out.sqfs.partial"
    partial.write_bytes(b"payload")
    final = tmp_path / "out.sqfs"

    publish_squashfs(partial, final)

    assert final.read_bytes() == b"payload"
    assert not partial.exists()


def _materialize_config(tmp_path: Path, shared_zip: Path) -> dict[str, object]:
    return {
        "materialize_tcga_ut": {
            "record_id": ZENODO_RECORD_ID,
            "zenodo_version": ZENODO_VERSION,
            "shared_zip": str(shared_zip),
            "scratch_root": str(tmp_path / "scratch"),
            "output_sqfs": str(tmp_path / "out" / "tcga.sqfs"),
            "materialization_sidecar": str(tmp_path / "out" / "tcga.provenance.json"),
            "manifest_partials_dir": str(tmp_path / "partials"),
        }
    }


def _stub_build_squashfs(_source: Path, partial: Path) -> None:
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(b"sqfs-bytes")


def _stub_mount_and_hash_matching(partials_dir: Path):
    def _mount_and_hash(_sqfs_path: Path, _mount_dir: Path) -> str:
        manifest_path = partials_dir / "canonical_manifest.jsonl"
        rows = [
            (record["path"], record["size"], record["sha256"])
            for record in (
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
            )
        ]
        return pack.tree_hash(rows)

    return _mount_and_hash


def _setup_two_class_materialize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    members = {
        "ClassA.zip": _nested_zip_bytes({"a1.jpg": b"a1", "a2.jpg": b"a2"}),
        "ClassB.zip": _nested_zip_bytes({"b1.jpg": b"b1"}),
        "LICENSE": b"license",
    }
    shared_zip = tmp_path / "data_raw.zip"
    _write_shared_zip(shared_zip, members)
    files = _zenodo_files(members)
    config = _materialize_config(tmp_path, shared_zip)
    partials_dir = Path(config["materialize_tcga_ut"]["manifest_partials_dir"])

    monkeypatch.setattr(source, "EXPECTED_FILE_COUNT", 3)
    monkeypatch.setattr(pack, "EXPECTED_CLASS_COUNT", 2)
    monkeypatch.setattr(pack, "EXPECTED_IMAGE_COUNT", 3)
    monkeypatch.setattr(
        source,
        "load_or_fetch_metadata",
        lambda *_: {"record_id": ZENODO_RECORD_ID, "version": ZENODO_VERSION, "files": files},
    )
    monkeypatch.setattr(squashfs, "build_squashfs", _stub_build_squashfs)
    monkeypatch.setattr(squashfs, "mount_and_hash", _stub_mount_and_hash_matching(partials_dir))
    return config


def test_materialize_publishes_only_when_mounted_hash_matches_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _setup_two_class_materialize(tmp_path, monkeypatch)

    sidecar = materialize(config)

    assert sidecar["validated"] is True
    assert sidecar["zenodo_record_id"] == ZENODO_RECORD_ID
    output_sqfs = Path(config["materialize_tcga_ut"]["output_sqfs"])
    assert output_sqfs.is_file()
    assert output_sqfs.with_suffix(output_sqfs.suffix + ".sha256").is_file()
    sidecar_path = Path(config["materialize_tcga_ut"]["materialization_sidecar"])
    assert sidecar_path.with_suffix(sidecar_path.suffix + ".sha256").is_file()


def test_materialize_refuses_to_publish_on_tree_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _setup_two_class_materialize(tmp_path, monkeypatch)
    monkeypatch.setattr(squashfs, "mount_and_hash", lambda *_: "0" * 64)

    with pytest.raises(RuntimeError, match="refusing to publish"):
        materialize(config)

    output_sqfs = Path(config["materialize_tcga_ut"]["output_sqfs"])
    assert not output_sqfs.exists()


def test_materialize_resumes_without_reextracting_completed_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _setup_two_class_materialize(tmp_path, monkeypatch)
    materialize(config)

    def _fail(*_args: object, **_kwargs: object) -> None:
        pytest.fail("materialize re-extracted an already-materialized class")

    monkeypatch.setattr(source, "extract_class_images", _fail)
    monkeypatch.setattr(source, "stage_nested_archive", _fail)

    sidecar = materialize(config)

    assert sidecar["validated"] is True
