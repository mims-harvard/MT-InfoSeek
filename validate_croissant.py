"""Structural validator for the repository's Croissant metadata."""
import json
import sys
import pathlib

path = pathlib.Path(__file__).resolve().parent / "croissant.json"
d = json.load(path.open())

DS_REQ = ["@context", "@type", "name", "url", "license", "conformsTo", "distribution", "recordSet"]
FO_REQ = ["@id", "contentUrl", "encodingFormat"]
RS_REQ = ["@id", "field"]
FIELD_REQ = ["@id", "dataType"]
RAI_REQ = [
    "rai:dataLimitations",
    "rai:dataBiases",
    "rai:personalSensitiveInformation",
    "rai:dataUseCases",
    "rai:dataSocialImpact",
]

errors = []
for k in DS_REQ:
    if k not in d:
        errors.append(f"dataset missing {k}")
for k in RAI_REQ:
    if k not in d:
        errors.append(f"dataset missing RAI field {k}")
for fo in d.get("distribution", []):
    fid = fo.get("@id")
    for k in FO_REQ:
        if k not in fo:
            errors.append(f"FileObject {fid} missing {k}")
    for k in ("rai:hasSyntheticData", "prov:wasDerivedFrom", "prov:wasGeneratedBy"):
        if k not in fo:
            errors.append(f"FileObject {fid} missing {k}")
for rs in d.get("recordSet", []):
    rid = rs.get("@id")
    for k in RS_REQ:
        if k not in rs:
            errors.append(f"RecordSet {rid} missing {k}")
    for fld in rs.get("field", []):
        fid = fld.get("@id")
        for k in FIELD_REQ:
            if k not in fld:
                errors.append(f"Field {fid} missing {k}")
        if "source" not in fld and "value" not in fld:
            errors.append(f"Field {fid} missing source or value")

# data/data_20q.py is the released artifact referenced by croissant.json, but the
# evaluator imports the package copy — the two must stay byte-identical.
repo = pathlib.Path(__file__).resolve().parent
released_20q = repo / "data" / "data_20q.py"
package_20q = repo / "20q" / "twenty_questions" / "data" / "data_20q.py"
if released_20q.read_bytes() != package_20q.read_bytes():
    errors.append("data/data_20q.py differs from 20q/twenty_questions/data/data_20q.py")

print("Total errors:", len(errors))
for e in errors:
    print(" -", e)
print()
print("Dataset name :", d["name"])
print("Dataset url  :", d["url"])
print("Conforms to  :", d["conformsTo"])
print("License      :", d["license"])
print("FileObjects  :")
for fo in d["distribution"]:
    print(f"  - {fo['@id']:<24} -> {fo['contentUrl']}  ({fo['encodingFormat']}, sha256={fo.get('sha256','')[:12]}…)")
print("RecordSets   :")
for rs in d["recordSet"]:
    print(f"  - {rs['@id']:<28} {len(rs.get('field', []))} fields")

if errors:
    sys.exit(1)
