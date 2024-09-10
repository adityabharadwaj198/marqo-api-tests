import argparse
import pytest
import subprocess
from typing import List, Optional
from tests.marqo_base_test import BaseTestCase

def run_tests(mode: str, from_version: str, to_version: str, marqo_api: str):
    if mode == "prepare":
        tests = [test for test in BaseTestCase.__subclasses__()
                 if getattr(test, 'marqo_from_version', '0') <= from_version]
        for test in tests:
            test().prepare()
    elif mode == "test":
        pytest.main([f"--marqo-api={marqo_api}",
                     f"-m", f"marqo_from_version<='{from_version}' or marqo_version<='{to_version}'"])

def start_marqo_container(version: str, image: Optional[str] = None, transfer_state: Optional[str] = None):
    image = image or f"marqoai/marqo:{version}"
    cmd = ["docker", "run", "-d", "--name", f"marqo-{version}", image]
    if transfer_state:
        cmd.extend(["--volumes-from", transfer_state])
    subprocess.run(cmd, check=True)

def stop_marqo_container(version: str):
    subprocess.run(["docker", "stop", f"marqo-{version}"], check=True)
    subprocess.run(["docker", "rm", f"marqo-{version}"], check=True)

def backwards_compatibility_test(from_version: str, to_version: str, from_image: Optional[str] = None, to_image: Optional[str] = None):
    try:
        start_marqo_container(from_version, from_image)
        run_tests("prepare", from_version, to_version, "http://localhost:8882")

        stop_marqo_container(from_version)

        start_marqo_container(to_version, to_image, transfer_state=f"marqo-{from_version}")

        run_tests("test", from_version, to_version, "http://localhost:8882")
    finally:
        stop_marqo_container(to_version)

def rollback_test(from_version: str, to_version: str, from_image: Optional[str] = None, to_image: Optional[str] = None):
    try:
        backwards_compatibility_test(from_version, to_version, from_image, to_image)

        stop_marqo_container(to_version)

        start_marqo_container(from_version, from_image, transfer_state=f"marqo-{to_version}")

        run_tests("test", from_version, to_version, "http://localhost:8882")
    finally:
        stop_marqo_container(from_version)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marqo Testing Runner")
    parser.add_argument("--mode", choices=["backwards_compatibility", "rollback"], required=True)
    parser.add_argument("--from_version", required=True)
    parser.add_argument("--to_version", required=True)
    parser.add_argument("--marqo-api", default="http://localhost:8882")
    parser.add_argument("--from_image", required=False)
    parser.add_argument("--to_image", required=False)
    args = parser.parse_args()
    if args.mode == "backwards_compatibility":
        backwards_compatibility_test(args.from_version, args.to_version, args.from_image, args.to_image)
    elif args.mode == "rollback":
        rollback_test(args.from_version, args.to_version, args.from_image, args.to_image)