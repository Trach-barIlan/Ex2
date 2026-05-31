"""Read model outputs from a JSON/JSONL file and compute simple Hebrewness metrics.

Input format (JSONL or JSON array): each object should contain at least:
  - "prompt"
  - "base_output"
  - "finetuned_output"

Output: JSONL file where each line is the input object augmented with:
  - "base_non_hebrew"
  - "finetuned_non_hebrew"
  - "notes"

Usage:
  python3 evaluate_from_json.py --input eval_outputs.jsonl --output evaluation_enriched.jsonl
"""

import argparse
import json
from typing import Iterable

HEBREW_RANGES = [(0x0590, 0x05FF), (0xFB1D, 0xFB4F)]


def is_hebrew_char(ch: str) -> bool:
    o = ord(ch)
    for a, b in HEBREW_RANGES:
        if a <= o <= b:
            return True
    return False


def count_non_hebrew(text: str) -> int:
    if text is None:
        return 0
    return sum(1 for c in text if (not c.isspace()) and (not is_hebrew_char(c)))


def count_non_space_chars(text: str) -> int:
    if text is None:
        return 0
    return sum(1 for c in text if not c.isspace())


def iter_input_records(path: str) -> Iterable[dict]:
    # Try JSONL first (one json object per line)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return
    except Exception:
        # Fall back to loading whole file as JSON (array)
        pass

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
        if isinstance(data, list):
            for obj in data:
                yield obj
        else:
            yield data


def main():
    parser = argparse.ArgumentParser(description="Enrich existing model-output JSON/JSONL with Hebrewness metrics")
    parser.add_argument("--input", type=str, required=True, help="Input JSON or JSONL file with prompt/base_output/finetuned_output")
    parser.add_argument("--output", type=str, default="evaluation_enriched.jsonl", help="Output JSONL file")
    parser.add_argument("--keep-outputs", action="store_true", help="Keep original base/finetuned outputs in the output JSON")
    parser.add_argument("--md-output", type=str, default="evaluation_enriched.md", help="Optional markdown output file to write the results table")
    args = parser.parse_args()

    records = list(iter_input_records(args.input))
    if not records:
        print(f"No records found in {args.input}")
        return

    summaries = []
    with open(args.output, "w", encoding="utf-8") as outfh:
        for rec in records:
            prompt = rec.get("prompt")
            base = rec.get("base_output", "")
            finetuned = rec.get("finetuned_output", "")

            base_nh = count_non_hebrew(base)
            finetuned_nh = count_non_hebrew(finetuned)
            base_total = count_non_space_chars(base)
            finetuned_total = count_non_space_chars(finetuned)

            base_pct = (base_nh / base_total * 100) if base_total > 0 else 0.0
            finetuned_pct = (finetuned_nh / finetuned_total * 100) if finetuned_total > 0 else 0.0
            delta = finetuned_nh - base_nh
            delta_pct = finetuned_pct - base_pct

            if finetuned_nh < base_nh:
                note = f"Reduced non-Hebrew chars: {base_nh} -> {finetuned_nh} (improved)"
            elif finetuned_nh > base_nh:
                note = f"Increased non-Hebrew chars: {base_nh} -> {finetuned_nh} (worse)"
            else:
                note = f"No change in non-Hebrew chars: {base_nh}"

            # Build summary output (omit full outputs by default)
            summary = {
                "prompt": prompt,
                "base_non_hebrew": base_nh,
                "base_non_hebrew_pct": round(base_pct, 2),
                "finetuned_non_hebrew": finetuned_nh,
                "finetuned_non_hebrew_pct": round(finetuned_pct, 2),
                "delta_non_hebrew": delta,
                "delta_non_hebrew_pct": round(delta_pct, 2),
                "notes": (rec.get("notes") or "").strip() or note,
            }

            # Optionally include the original outputs if requested
            if args.keep_outputs:
                summary["base_output"] = base
                summary["finetuned_output"] = finetuned

            outfh.write(json.dumps(summary, ensure_ascii=False) + "\n")
            summaries.append(summary)

    # Generate Markdown table and aggregate summary
    try:
        import statistics
    except Exception:
        statistics = None

    def truncate(s, n=80):
        s = s.replace("|", "\\|")
        return (s[: n - 1] + "…") if len(s) > n else s

    md_lines = []
    md_lines.append("# Evaluation Results\n")
    md_lines.append("| # | Prompt | base_non_hebrew | base_% | finetuned_non_hebrew | finetuned_% | delta | delta_% | notes |")
    md_lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for i, s in enumerate(summaries, start=1):
        md_lines.append(
            f"| {i} | {truncate(s.get('prompt',''))} | {s.get('base_non_hebrew',0)} | {s.get('base_non_hebrew_pct',0)}% | {s.get('finetuned_non_hebrew',0)} | {s.get('finetuned_non_hebrew_pct',0)}% | {s.get('delta_non_hebrew',0)} | {s.get('delta_non_hebrew_pct',0)}% | {truncate(s.get('notes',''),60)} |"
        )

    # Aggregates
    deltas = [s.get('delta_non_hebrew') for s in summaries]
    deltas_pct = [s.get('delta_non_hebrew_pct') for s in summaries]
    improved = sum(1 for d in deltas if d < 0)
    worse = sum(1 for d in deltas if d > 0)
    same = sum(1 for d in deltas if d == 0)
    mean_delta = statistics.mean(deltas) if statistics and deltas else None
    median_delta = statistics.median(deltas) if statistics and deltas else None
    mean_delta_pct = statistics.mean(deltas_pct) if statistics and deltas_pct else None
    median_delta_pct = statistics.median(deltas_pct) if statistics and deltas_pct else None

    md_lines.append("\n## Aggregate Summary\n")
    md_lines.append(f"- Prompts evaluated: {len(summaries)}")
    md_lines.append(f"- Improved (fewer non-Hebrew chars): {improved}")
    md_lines.append(f"- Worse (more non-Hebrew chars): {worse}")
    md_lines.append(f"- Unchanged: {same}")
    if mean_delta is not None:
        md_lines.append(f"- Mean delta (finetuned - base): {mean_delta:.2f}")
        md_lines.append(f"- Median delta: {median_delta:.2f}")
        md_lines.append(f"- Mean delta % (finetuned - base): {mean_delta_pct:.2f}%")
        md_lines.append(f"- Median delta %: {median_delta_pct:.2f}%")

    md_text = "\n".join(md_lines)

    # Print markdown to stdout
    print("\n--- Markdown Evaluation Table ---\n")
    print(md_text)

    # Optionally write to a markdown file
    if args.md_output:
        with open(args.md_output, "w", encoding="utf-8") as mdfh:
            mdfh.write(md_text)
        print(f"Wrote markdown table to {args.md_output}")

    print(f"Wrote {len(records)} enriched records to {args.output}")


if __name__ == "__main__":
    main()
