"""
CLI entry point for the Natural Scene Text Detection & Recognition pipeline.

Usage examples:
    # Single image
    python main.py --input data/sample_images/sign.jpg --output output/

    # Batch processing
    python main.py --input data/sample_images/ --output output/ --batch

    # Custom thresholds
    python main.py --input photo.jpg --east-conf 0.6 --tess-conf 50 --width 640 --height 640
"""

import argparse
import json
import logging
import os
import sys
import time

from src.config import (
    EAST_CONF_THRESHOLD,
    EAST_INPUT_HEIGHT,
    EAST_INPUT_WIDTH,
    EAST_NMS_THRESHOLD,
    LOG_FORMAT,
    LOG_LEVEL,
    TESSERACT_CONF_THRESHOLD,
)


def setup_logging(level_str):
    """Configure root logger with the given level string."""
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(level=level, format=LOG_FORMAT)


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="scene-text",
        description=(
            "Natural Scene Text Detection & Recognition — "
            "detect and recognize text in natural scene images using "
            "EAST text detector + Tesseract OCR."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --input photo.jpg\n"
            "  python main.py --input photos/ --batch --output results/\n"
            "  python main.py --input sign.png --east-conf 0.6 --width 640 --height 640\n"
        ),
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to a single image file, or a directory (when used with --batch).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output",
        help="Directory to save annotated images and JSON results (default: output/).",
    )
    parser.add_argument(
        "--batch",
        "-b",
        action="store_true",
        default=False,
        help="Treat --input as a directory and process all images inside it.",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        default=False,
        help="When used with --batch, also process images in subdirectories.",
    )
    parser.add_argument(
        "--east-conf",
        type=float,
        default=EAST_CONF_THRESHOLD,
        help=f"EAST detection confidence threshold, 0.0–1.0 (default: {EAST_CONF_THRESHOLD}).",
    )
    parser.add_argument(
        "--east-nms",
        type=float,
        default=EAST_NMS_THRESHOLD,
        help=f"EAST NMS IoU threshold, 0.0–1.0 (default: {EAST_NMS_THRESHOLD}).",
    )
    parser.add_argument(
        "--tess-conf",
        type=float,
        default=TESSERACT_CONF_THRESHOLD,
        help=f"Tesseract minimum word confidence, 0–100 (default: {TESSERACT_CONF_THRESHOLD}).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=EAST_INPUT_WIDTH,
        help=f"EAST input width, must be a multiple of 32 (default: {EAST_INPUT_WIDTH}).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=EAST_INPUT_HEIGHT,
        help=f"EAST input height, must be a multiple of 32 (default: {EAST_INPUT_HEIGHT}).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Print results to stdout only; do not save files to disk.",
    )
    parser.add_argument(
        "--log-level",
        default=LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help=f"Logging verbosity (default: {LOG_LEVEL}).",
    )

    args = parser.parse_args(argv)

    # Validate dimensions are multiples of 32
    if args.width % 32 != 0:
        parser.error(f"--width must be a multiple of 32, got {args.width}")
    if args.height % 32 != 0:
        parser.error(f"--height must be a multiple of 32, got {args.height}")

    # Validate input path exists
    if not os.path.exists(args.input):
        parser.error(f"Input path does not exist: {args.input}")

    if args.batch and not os.path.isdir(args.input):
        parser.error(f"--batch requires --input to be a directory, got: {args.input}")

    if not args.batch and os.path.isdir(args.input):
        parser.error(
            f"--input is a directory but --batch was not specified. "
            f"Use --batch to process all images in a directory."
        )

    return args


def print_detections(result):
    """Pretty-print detection results to stdout."""
    image_path = result["image_path"]
    detections = result["detections"]
    total = result["total_detections"]
    elapsed = result["processing_time_ms"]

    print(f"\n{'=' * 60}")
    print(f"Image: {image_path}")
    print(f"Size:  {result['image_size'][1]}x{result['image_size'][0]}")
    print(f"Time:  {elapsed:.1f} ms")
    print(f"{'=' * 60}")

    if not detections:
        print("  (no text detected)")
        return

    text_detections = [d for d in detections if d["text"]]

    if not text_detections:
        print(f"  {len(detections)} regions detected but no text recognized.")
        return

    print(f"  Found {total} text region(s):\n")
    for det in text_detections:
        det_id = det["id"]
        text = det["text"]
        conf = det["confidence"]
        det_conf = det.get("detection_confidence", 0.0)
        source = det.get("source", "unknown")
        print(f'  [{det_id}] "{text}"')
        print(f"      OCR confidence:       {conf:.1f}%")
        print(f"      Detection confidence: {det_conf:.1f}%")
        print(f"      Source:               {source}")
        print()


def run_single(args):
    """Process a single image."""
    from src.pipeline import SceneTextPipeline, save_results

    logger = logging.getLogger(__name__)

    pipeline = SceneTextPipeline(
        east_width=args.width,
        east_height=args.height,
        east_conf=args.east_conf,
        east_nms=args.east_nms,
        tess_conf=args.tess_conf,
    )

    result = pipeline.process_image(args.input)
    print_detections(result)

    if not args.no_save:
        saved = save_results(result, output_dir=args.output)
        print(f"\nSaved:")
        if saved.get("annotated_image"):
            print(f"  Annotated image: {saved['annotated_image']}")
        print(f"  JSON results:    {saved['json_file']}")

    return result


def run_batch(args):
    """Process all images in a directory."""
    from src.pipeline import SceneTextPipeline, save_batch_results

    logger = logging.getLogger(__name__)

    pipeline = SceneTextPipeline(
        east_width=args.width,
        east_height=args.height,
        east_conf=args.east_conf,
        east_nms=args.east_nms,
        tess_conf=args.tess_conf,
    )

    start = time.time()
    results, errors = pipeline.process_directory(args.input, recursive=args.recursive)
    total_time = time.time() - start

    # Print results for each image
    for result in results:
        print_detections(result)

    # Print errors
    if errors:
        print(f"\n{'!' * 60}")
        print(f"ERRORS ({len(errors)}):")
        for err in errors:
            print(f"  {err['image_path']}: {err['error']}")
        print(f"{'!' * 60}")

    # Print summary
    total_text = sum(r["total_detections"] for r in results)
    print(f"\n{'=' * 60}")
    print(f"BATCH SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Images processed:  {len(results)}")
    print(f"  Images failed:     {len(errors)}")
    print(f"  Total text found:  {total_text}")
    print(f"  Total time:        {total_time:.1f}s")
    if results:
        avg_time = sum(r["processing_time_ms"] for r in results) / len(results)
        print(f"  Avg time/image:    {avg_time:.1f} ms")
    print(f"{'=' * 60}")

    if not args.no_save:
        summary = save_batch_results(results, errors=errors, output_dir=args.output)
        print(f"\nResults saved to: {os.path.abspath(args.output)}")

    return results, errors


def main(argv=None):
    """Main entry point."""
    args = parse_args(argv)
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("Scene Text Detection & Recognition Pipeline")
    logger.info(
        "Configuration: width=%d, height=%d, east_conf=%.2f, tess_conf=%.0f",
        args.width,
        args.height,
        args.east_conf,
        args.tess_conf,
    )

    try:
        if args.batch:
            run_batch(args)
        else:
            run_single(args)
    except FileNotFoundError as e:
        logger.error(str(e))
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except EnvironmentError as e:
        logger.error(str(e))
        print(f"\nEnvironment error: {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
