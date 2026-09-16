#!/usr/bin/env python3
"""Quick demo run with relaxed validation gates.

This script demonstrates the full pipeline with settings tuned to find gaps
even with smaller corpora. Production runs should use config.yaml.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.main import main


def run_demo():
    """Execute demo with relaxed config."""
    print("=" * 70)
    print("ESV-Gap Demo - Relaxed Validation")
    print("=" * 70)
    print("\nThis demo uses relaxed thresholds to demonstrate gap detection")
    print("with smaller corpora. For production use, see config.yaml.\n")

    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️  Warning: GROQ_API_KEY not set. LLM-based stages will fail.")
        print("   Set it with: export GROQ_API_KEY=your_key_here\n")

    # Use relaxed config
    config_path = Path(__file__).parent / "config_relaxed.yaml"
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)

    print(f"📋 Using config: {config_path}\n")

    # Run with demo query
    demo_query = "explainable AI interpretability"
    print(f"🔍 Query: {demo_query}")
    print("⏳ Starting pipeline...\n")

    try:
        main(
            config_path=str(config_path),
            query=demo_query,
            resume=False,
        )
        print("\n✅ Demo completed successfully!")
        print("📁 Check the 'runs/' directory for results.")

    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_demo()
