"""The line AutoResearcher reads a check's findings from; a crash prints none."""
import json

FINDINGS = "===AR_FINDINGS==="


def print_findings(findings: list[str]) -> None:
    print(FINDINGS + json.dumps(findings), flush=True)
