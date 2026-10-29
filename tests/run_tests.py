"""Runs the test-suite and writes docs/TEST_RESULTS.md and docs/test_results.json."""
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
xml_path = ROOT / "docs" / "junit.xml"
xml_path.parent.mkdir(exist_ok=True)
subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", f"--junitxml={xml_path}", "-s"],
               cwd=ROOT, check=False)

cases = []
for tc in ET.parse(xml_path).getroot().iter("testcase"):
    status = "Pass"
    for child in tc:
        if child.tag in ("failure", "error"):
            status = "Fail"
        elif child.tag == "skipped":
            status = "Skipped"
    cases.append({"name": tc.get("name"), "status": status, "seconds": float(tc.get("time"))})

SCENARIO = {
    "TC-01a": ("UC-01", "Register with a valid, unique email"), "TC-01b": ("UC-01", "Register with an already-used email"),
    "TC-02a": ("UC-02", "Login with correct credentials"), "TC-02b": ("UC-02", "Login with repeated incorrect password"),
    "TC-03a": ("UC-03", "Get recommendations from valid symptoms"), "TC-03b": ("UC-03", "Low-confidence prediction"),
    "TC-04a": ("UC-04", "View history with prior queries"), "TC-04b": ("UC-04", "View history with no prior queries"),
    "TC-05a": ("UC-05", "Submit valid feedback"), "TC-05b": ("UC-05", "Close feedback form without submitting"),
    "TC-06a": ("UC-06", "Upload a valid dataset"), "TC-06b": ("UC-06", "Upload an invalid dataset"),
    "TC-07a": ("UC-07", "Retrain and deploy an improved model"), "TC-07b": ("UC-07", "Retrain yields a lower-accuracy model"),
}
rows = []
for tcid, (uc, scen) in SCENARIO.items():
    key = tcid.replace("-", "_")
    related = [c for c in cases if f"test_{key}" in c["name"]]
    ok = bool(related) and all(c["status"] == "Pass" for c in related)
    rows.append({"id": tcid, "uc": uc, "scenario": scen, "tests": [c["name"] for c in related], "status": "Pass" if ok else "Fail"})

extra = [c for c in cases if not re.match(r"test_TC_\d\d[ab]", c["name"])]
passed = sum(c["status"] == "Pass" for c in cases)
out = ["# Test results", "", f"{passed} of {len(cases)} automated tests passed.", "",
       "## Design Document test cases", "", "| ID | Use case | Scenario | Automated checks | Status |", "|---|---|---|---|---|"]
for r in rows:
    out.append(f"| {r['id']} | {r['uc']} | {r['scenario']} | {len(r['tests'])} | **{r['status']}** |")
out += ["", "## Additional tests", "", "| Test | Status |", "|---|---|"] + [f"| {c['name']} | {c['status']} |" for c in extra]
(ROOT / "docs" / "TEST_RESULTS.md").write_text("\n".join(out) + "\n", encoding="utf-8")
(ROOT / "docs" / "test_results.json").write_text(json.dumps({"cases": rows, "all": cases}, indent=2), encoding="utf-8")
print("\n".join(out[:24]))
