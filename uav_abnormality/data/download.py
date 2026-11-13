"""Reproducible public dataset acquisition; no cloud credentials are required."""
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tarfile
import urllib.error
import urllib.request
import zipfile


def _inside(root, name):
    root = Path(root).resolve()
    destination = (root / name).resolve()
    if destination != root and root not in destination.parents:
        raise ValueError("Archive or manifest path escapes destination: %s" % name)
    return destination


def file_digest(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path, entry):
    path = Path(path)
    if not path.is_file():
        return False
    if entry.get("size") is not None and path.stat().st_size != int(entry["size"]):
        return False
    checksum = entry.get("checksum")
    if checksum:
        algorithm, expected = checksum.split(":", 1)
        if file_digest(path, algorithm).lower() != expected.lower():
            return False
    return True


def download_file(entry, destination, timeout=60):
    """Download into .part, resume HTTP ranges, then validate before atomic rename.

    A server which ignores Range restarts the partial file. A corrupt existing
    completed file is preserved until the verified replacement is ready.
    """
    target = _inside(destination, entry["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if verify_file(target, entry):
        return target
    partial = target.with_name(target.name + ".part")
    if (entry.get("size") is not None or entry.get("checksum")) and verify_file(partial, entry):
        os.replace(str(partial), str(target))
        return target
    offset = partial.stat().st_size if partial.exists() else 0
    if entry.get("size") is not None and offset >= int(entry["size"]):
        partial.unlink()
        offset = 0
    headers = {"User-Agent": "UAV-Abnormality-Detect/0.1"}
    if offset:
        headers["Range"] = "bytes=%d-" % offset
    request = urllib.request.Request(entry["url"], headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", None)
        resume = offset > 0 and status == 206
        if resume:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith("bytes %d-" % offset):
                raise ValueError("Server returned an unexpected Content-Range")
        with partial.open("ab" if resume else "wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    if not verify_file(partial, entry):
        raise ValueError("Size/checksum mismatch for %s; retained .part for inspection" % target)
    os.replace(str(partial), str(target))
    return target


def download_manifest(manifest, destination, patterns=None, max_bytes=None):
    """Download selected entries; a byte cap is checked before network access."""
    if not isinstance(manifest, dict):
        manifest = json.loads(Path(manifest).read_text())
    entries = manifest["files"]
    if patterns:
        entries = [e for e in entries if any(fnmatch.fnmatch(e["path"], p) for p in patterns)]
    if not entries:
        raise ValueError("No manifest files match selection")
    total = sum(int(e.get("size", 0)) for e in entries)
    if max_bytes is not None and (any(e.get("size") is None for e in entries) or total > max_bytes):
        raise ValueError("Selection exceeds byte cap or has an unknown size")
    results = []
    for entry in entries:
        path = download_file(entry, destination)
        results.append({"path": entry["path"], "size": path.stat().st_size,
                        "sha256": file_digest(path), "url": entry["url"]})
    Path(destination).mkdir(parents=True, exist_ok=True)
    receipt_path = Path(destination) / "download_receipt.json"
    existing = json.loads(receipt_path.read_text()).get("files", []) if receipt_path.exists() else []
    merged = {entry["path"]: entry for entry in existing if not Path(entry["path"]).is_absolute()}
    merged.update({entry["path"]: entry for entry in results})
    receipt = {"dataset": manifest.get("dataset"), "files": list(merged.values())}
    receipt_path.write_text(json.dumps(receipt, indent=2))
    return receipt


def safe_extract(archive, destination, max_uncompressed_bytes=100 * 1024 ** 3):
    """Extract ordinary files only. Refuse traversal, links and archive bombs."""
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            entries = handle.infolist()
            if sum(e.file_size for e in entries) > max_uncompressed_bytes:
                raise ValueError("Uncompressed archive exceeds byte cap")
            for entry in entries:
                _inside(root, entry.filename)
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError("Archive links are not allowed")
            handle.extractall(root)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            entries = handle.getmembers()
            if sum(e.size for e in entries) > max_uncompressed_bytes:
                raise ValueError("Uncompressed archive exceeds byte cap")
            for entry in entries:
                _inside(root, entry.name)
                if not (entry.isfile() or entry.isdir()):
                    raise ValueError("Archive links/special files are not allowed")
            handle.extractall(root, members=entries)
    else:
        raise ValueError("Supported archive formats are ZIP and TAR")
    return root


def zenodo_manifest(record_id="3923056"):
    """Resolve Zenodo metadata; persist the returned manifest to pin a version."""
    url = "https://zenodo.org/api/records/%s" % record_id
    with urllib.request.urlopen(url, timeout=60) as response:
        record = json.load(response)
    return {"dataset": "LAPSE-RATE-UKY", "record_id": str(record["id"]),
            "doi": record.get("doi"), "license": record.get("metadata", {}).get("license"),
            "metadata_url": url,
            "files": [{"path": f["key"], "url": f["links"]["self"],
                       "size": f["size"], "checksum": f["checksum"]}
                      for f in record["files"]]}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--pattern", action="append", help="Repeat to select exact names or shell-style patterns")
    parser.add_argument("--max-bytes", type=int, default=None)
    arguments = parser.parse_args()
    receipt = download_manifest(arguments.manifest, arguments.destination, arguments.pattern, arguments.max_bytes)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
