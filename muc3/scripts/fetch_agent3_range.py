"""Range-fetch one entry from the 622MB trained_models.zip on S3 without
downloading the whole file. Lists the central directory, then extracts only
the requested entry. Writes a JSON manifest of all entries."""
import io, json, struct, sys, zlib
import requests

URL = "https://microrts.s3.amazonaws.com/microrts/gym-microrts-paper/trained_models.zip"
OUTDIR = "/home/hatch/workspace/step3-scratch"
WANT_SUFFIX = "ppo_gridnet_coacai/agent-3.pt"

def get_range(start, end):
    r = requests.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
    r.raise_for_status()
    assert r.status_code == 206, f"expected 206, got {r.status_code}"
    return r.content

head = requests.head(URL, timeout=60)
total = int(head.headers["Content-Length"])
print("total bytes:", total, "last-modified:", head.headers.get("Last-Modified"))

# End of central directory is in the last 64KB (no zip64 expected, but check)
tail = get_range(total - 65536, total - 1)
eocd_sig = b"PK\x05\x06"
idx = tail.rfind(eocd_sig)
assert idx >= 0, "EOCD not found"
eocd = tail[idx:idx+22]
(cd_size, cd_offset, n_entries) = struct.unpack("<I I H", eocd[12:22])[0], None, None
# unpack properly: disk(2) cd_disk(2) entries_disk(2) entries_total(2) cd_size(4) cd_offset(4) comment_len(2)
_, _, _, n_entries, cd_size, cd_offset, _ = struct.unpack("<HHHHIIH", eocd[4:22])
print(f"central dir: offset={cd_offset} size={cd_size} entries={n_entries}")

cd = get_range(cd_offset, cd_offset + cd_size - 1)
entries = []
p = 0
while p < len(cd):
    assert cd[p:p+4] == b"PK\x01\x02", f"bad CD signature at {p}"
    (ver_made, ver_need, flag, method, mtime, mdate, crc, csize, usize,
     nlen, elen, clen, disk, iattr, eattr, lho) = struct.unpack("<HHHHHHIIIHHHHHII", cd[p+4:p+46])
    name = cd[p+46:p+46+nlen].decode()
    entries.append({"name": name, "method": method, "compressed": csize,
                    "uncompressed": usize, "local_header_offset": lho, "flag": flag})
    p += 46 + nlen + elen + clen

with open(f"{OUTDIR}/trained_models_zip_manifest.json", "w") as f:
    json.dump({"url": URL, "total_bytes": total, "entries": entries}, f, indent=2)
print(f"listed {len(entries)} entries; manifest written")
for e in entries:
    print(f"  {e['name']}  method={e['method']} c={e['compressed']} u={e['uncompressed']}")

matches = [e for e in entries if e["name"].endswith(WANT_SUFFIX)]
print("matches for", WANT_SUFFIX, ":", len(matches))
if matches:
    e = matches[0]
    # local header: 30 bytes + name + extra
    lh = get_range(e["local_header_offset"], e["local_header_offset"] + 29)
    assert lh[:4] == b"PK\x03\x04"
    nlen, elen = struct.unpack("<HH", lh[26:30])
    data_start = e["local_header_offset"] + 30 + nlen + elen
    raw = get_range(data_start, data_start + e["compressed"] - 1)
    if e["method"] == 8:
        data = zlib.decompress(raw, -15)
    elif e["method"] == 0:
        data = raw
    else:
        raise SystemExit(f"unsupported method {e['method']}")
    assert len(data) == e["uncompressed"], (len(data), e["uncompressed"])
    out = f"{OUTDIR}/agent-3.pt"
    with open(out, "wb") as f:
        f.write(data)
    import hashlib
    print("wrote", out, len(data), "bytes sha256:", hashlib.sha256(data).hexdigest())
