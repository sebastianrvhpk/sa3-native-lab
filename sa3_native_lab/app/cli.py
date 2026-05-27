from __future__ import annotations

import argparse
import os

from .server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SA3 Native Lab API daemon.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8733)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    if args.artifact_root:
        os.environ["SA3_LAB_HOME"] = args.artifact_root
    if args.repo_root:
        os.environ["SA3_REPO_ROOT"] = args.repo_root

    if args.reload:
        uvicorn.run(
            "sa3_native_lab.app.server:create_app",
            host=args.host,
            port=args.port,
            reload=True,
            factory=True,
        )
        return

    app = create_app(artifact_root=args.artifact_root, repo_root=args.repo_root)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
