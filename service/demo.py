#!/usr/bin/env python3
"""
Demo Script for 2.5D MRI Segmentation Pipeline

Complete end-to-end demo that runs:
1. Preprocessing (validates existing data)
2. Training (trains a small model quickly)
3. Inference (runs predictions on test data)

This script runs with sensible defaults and no command-line arguments.
Perfect for quick demonstrations and testing the complete pipeline.

Usage:
    python service/demo.py
"""

import sys
from pathlib import Path
import subprocess
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_banner(text: str):
    """Print a formatted banner."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def run_subprocess(cmd: list, description: str) -> bool:
    """
    Run a subprocess command.

    Args:
        cmd: Command to run as list
        description: Description of what's running

    Returns:
        True if successful, False otherwise
    """
    print(f"▶ Running: {description}")
    print(f"  Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=str(Path(__file__).parent.parent),
        )
        print(f"\n✓ Success: {description}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {description}")
        print(f"  Error code: {e.returncode}\n")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ Interrupted: {description}\n")
        return False


if __name__ == "__main__":
    start_time = time.time()

    print_banner("2.5D MRI SEGMENTATION - COMPLETE DEMO")

    print("This demo will run the complete pipeline:")
    print("  1. Preprocessing (validates existing data)")
    print("  2. Training (small model, 10 epochs for demo)")
    print("  3. Inference (predictions on test data)")
    print()
    print("Demo Configuration:")
    print("  • Model: SMP ResNet34 U-Net")
    print("  • Stack depth: 5 slices")
    print("  • Image size: 256x256")
    print("  • Batch size: 4 (small for compatibility)")
    print("  • Epochs: 10 (quick demo)")
    print("  • Data: Class 2 (PIRADS 3)")
    print()

    input("Press Enter to start the demo (or Ctrl+C to cancel)...")

    # ========================================================================
    # STEP 1: PREPROCESSING
    # ========================================================================
    print_banner("STEP 1/3: PREPROCESSING")

    print("Validating data and running necessary preprocessing...")
    print("This step will skip already-completed conversions.\n")

    preprocess_cmd = ["python", "service/preprocess.py", "--all"]

    if not run_subprocess(preprocess_cmd, "Data Preprocessing"):
        print("⚠ Preprocessing failed or was interrupted.")
        print("The pipeline needs valid data to continue.")
        sys.exit(1)

    print("✓ Data is ready for training!\n")
    time.sleep(2)

    # ========================================================================
    # STEP 2: TRAINING
    # ========================================================================
    print_banner("STEP 2/3: TRAINING")

    print("Training a 2.5D segmentation model...")
    print("For demo purposes, we'll train for only 10 epochs.")
    print("In production, you'd typically train for 50-100+ epochs.\n")

    # Create demo output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    demo_checkpoint_dir = f"checkpoints/demo_{timestamp}"

    train_cmd = [
        "python",
        "service/train.py",
        "--model",
        "smp",
        "--encoder",
        "resnet34",
        "--manifest",
        "data/processed/class2/manifest.csv",
        "--batch-size",
        "4",
        "--epochs",
        "10",
        "--lr",
        "1e-4",
        "--loss",
        "dice_bce",
        "--image-size",
        "256",
        "256",
        "--stack-depth",
        "5",
        "--output-dir",
        demo_checkpoint_dir,
        "--num-workers",
        "2",
    ]

    print(f"Checkpoint directory: {demo_checkpoint_dir}\n")

    if not run_subprocess(train_cmd, "Model Training"):
        print("⚠ Training failed or was interrupted.")
        print("This could be due to:")
        print("  • Missing dependencies (torch, segmentation-models-pytorch)")
        print("  • Insufficient GPU memory (try reducing --batch-size)")
        print("  • No training data with masks")
        sys.exit(1)

    print("✓ Model training complete!\n")

    # Check if checkpoint exists
    checkpoint_path = Path(demo_checkpoint_dir) / "best_model.pth"
    if not checkpoint_path.exists():
        print(f"⚠ Checkpoint not found at {checkpoint_path}")
        print("Training may have completed but no best model was saved.")
        print("This could mean no validation data with masks was available.")
        sys.exit(1)

    print(f"✓ Best model saved: {checkpoint_path}\n")
    time.sleep(2)

    # ========================================================================
    # STEP 3: INFERENCE
    # ========================================================================
    print_banner("STEP 3/3: INFERENCE")

    print("Running inference on test data...")
    print("This will generate segmentation masks and visualizations.\n")

    demo_output_dir = f"predictions/demo_{timestamp}"

    inference_cmd = [
        "python",
        "service/inference.py",
        "--checkpoint",
        str(checkpoint_path),
        "--manifest",
        "data/processed/class2/manifest.csv",
        "--output",
        demo_output_dir,
        "--visualize",
        "--num-vis",
        "10",
        "--batch-size",
        "4",
        "--num-workers",
        "2",
    ]

    print(f"Output directory: {demo_output_dir}\n")

    if not run_subprocess(inference_cmd, "Inference"):
        print("⚠ Inference failed or was interrupted.")
        print("The model was trained but predictions couldn't be generated.")
        sys.exit(1)

    print("✓ Inference complete!\n")
    time.sleep(1)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_banner("DEMO COMPLETE!")

    elapsed_time = time.time() - start_time
    elapsed_mins = int(elapsed_time // 60)
    elapsed_secs = int(elapsed_time % 60)

    print(f"✓ Total time: {elapsed_mins}m {elapsed_secs}s\n")

    print("Demo Summary:")
    print("=" * 80)

    print("\n1. ✓ Preprocessing completed")
    print("   • Validated existing data conversions")
    print("   • Data ready at: data/processed/\n")

    print("2. ✓ Training completed")
    print(f"   • Model: SMP ResNet34 U-Net")
    print(f"   • Trained for: 10 epochs")
    print(f"   • Checkpoint: {checkpoint_path}\n")

    print("3. ✓ Inference completed")
    print(f"   • Predictions: {demo_output_dir}/class2/masks/")
    print(f"   • Visualizations: {demo_output_dir}/class2/visualizations/")
    print(f"   • Summary: {demo_output_dir}/class2/summary.csv\n")

    print("=" * 80)
    print("\nNext Steps:")
    print("=" * 80)

    print("\n1. Review Results:")
    print(f"   • Check visualizations: {demo_output_dir}/class2/visualizations/")
    print(f"   • Review summary report: {demo_output_dir}/class2/summary.csv")

    print("\n2. Evaluate Model (if you have ground truth masks):")
    print(f"   python service/test.py \\")
    print(f"       --checkpoint {checkpoint_path} \\")
    print(f"       --visualize --save-predictions")

    print("\n3. Train for Real (longer training):")
    print("   python service/train.py \\")
    print("       --model smp --encoder resnet34 \\")
    print("       --epochs 100 --batch-size 8")

    print("\n4. Run Inference on Different Data:")
    print(f"   python service/inference.py \\")
    print(f"       --checkpoint {checkpoint_path} \\")
    print("       --manifest data/processed/class3/manifest.csv \\")
    print("       --output predictions/class3/ --visualize")

    print("\n5. Read Documentation:")
    print("   • Quick Start: QUICK_START_SERVICE.md")
    print("   • Full Guide: service/README.md")
    print("   • Pipeline Details: tools/README_2D5_PIPELINE.md")

    print("\n" + "=" * 80)
    print("\n🎉 Demo successfully completed!")
    print("\nYou now have:")
    print("  ✓ A trained segmentation model")
    print("  ✓ Predictions on test data")
    print("  ✓ Visualizations to review")
    print("\nYour 2.5D MRI segmentation pipeline is ready for production use!")
    print("\n" + "=" * 80 + "\n")
