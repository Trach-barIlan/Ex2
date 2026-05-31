import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Run a large Python script with live output and logging.")
    parser.add_argument("script", help="Path to the Python script to run")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to the script after --")
    parser.add_argument("--log", default="run_large_python.log", help="Log file to write combined stdout/stderr")
    args = parser.parse_args()

    script_path = os.path.abspath(args.script)
    if not os.path.exists(script_path):
        print(f"Script not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    extra_args = args.script_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    command = [sys.executable, "-u", script_path, *extra_args]

    print(f"Running: {' '.join(command)}", flush=True)
    print(f"Logging to: {os.path.abspath(args.log)}", flush=True)

    with open(args.log, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()
        if return_code != 0:
            print(f"\nProcess exited with code {return_code}", file=sys.stderr)
            sys.exit(return_code)

    print("\nDone.")


if __name__ == "__main__":
    main()
