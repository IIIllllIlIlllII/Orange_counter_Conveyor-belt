"""
Orange Counter on Conveyor Belt
================================
Real-time orange detection and counting on a conveyor belt
using Ultralytics YOLO + ObjectCounter with a counting line.

Usage:
    python orange_counter.py --source orange_video.mp4 --model best.pt --interactive --save
    python orange_counter.py --source orange_video.mp4 --model best.pt --points 0,613,1077,1076 --save
"""

import argparse
import cv2
from ultralytics import YOLO, solutions


# ── Colors (BGR) ─────────────────────────────────────────────────────────
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)


# ── Interactive line placement (2 points) ────────────────────────────────

def pick_line_interactive(video_path):
    """Click 2 points to define a counting line. R=reset, Enter=confirm."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Error: cannot open video '{video_path}'"
    ret, frame = cap.read()
    cap.release()
    assert ret, "Error: cannot read the first frame."

    points = []
    canvas = frame.copy()
    window = "Click 2 points for counting line | R=reset | Enter=confirm"

    def on_mouse(event, x, y, flags, param):
        nonlocal canvas
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            cv2.circle(canvas, (x, y), 6, GREEN, -1)
            cv2.circle(canvas, (x, y), 8, WHITE, 2)
            if len(points) == 2:
                cv2.line(canvas, points[0], points[1], GREEN, 2)
                cv2.putText(canvas, "Enter=confirm  R=reset",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
            cv2.imshow(window, canvas)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    cv2.putText(canvas, "Click 2 points to draw the counting line",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow(window, canvas)
    canvas = frame.copy()

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            points.clear()
            canvas = frame.copy()
            cv2.putText(canvas, "Click 2 points to draw the counting line",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(window, canvas)
        if key in (13, 32) and len(points) == 2:
            break
        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            print("Cancelled.")
            exit(0)

    cv2.destroyAllWindows()
    print(f"Counting line: {points[0]} -> {points[1]}")
    return points


# ── Argument parser ──────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Count oranges on a conveyor belt using YOLO."
    )
    parser.add_argument("--source", type=str, default="orange_video.mp4")
    parser.add_argument("--model", type=str, default="best.pt")
    parser.add_argument("--output", type=str, default="output.avi")
    parser.add_argument("--points", type=str, default=None,
                        help="Line coordinates: 'x1,y1,x2,y2' (e.g. '0,613,1077,1076')")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.source)
    assert cap.isOpened(), f"Error: cannot open video '{args.source}'"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"Video: {args.source} ({w}x{h} @ {fps} FPS)")

    video_writer = None
    if args.save:
        video_writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(f"Saving to: {args.output}")

    # Define counting line
    if args.points:
        coords = list(map(int, args.points.split(",")))
        region = [(coords[0], coords[1]), (coords[2], coords[3])]
        print(f"Counting line: {region[0]} -> {region[1]}")
    elif args.interactive:
        cap.release()
        region = pick_line_interactive(args.source)
        cap = cv2.VideoCapture(args.source)
    else:
        line_y = h // 2
        region = [(0, line_y), (w, line_y)]
        print(f"Counting line: {region[0]} -> {region[1]}")

    # Initialize ObjectCounter (display off — we draw our own HUD)
    counter = solutions.ObjectCounter(
        show=False,
        region=region,
        model=args.model,
        conf=args.conf,
        line_width=2,
        show_in=False,
        show_out=False,
    )

    print(f"Model: {args.model} | Conf: {args.conf}")
    print("-" * 50)
    print("Press 'q' to quit.\n")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        result = counter(frame)
        display_frame = result.plot_im.copy()

        # Draw simple count HUD
        total = counter.in_count + counter.out_count
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (200, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        cv2.putText(display_frame, f"Count: {total}", (20, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        if not args.no_show:
            cv2.imshow("Orange Counter", display_frame)

        if args.save and video_writer is not None:
            video_writer.write(display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

    print(f"\nDone! Processed {frame_count} frames.")
    print(f"  Count: {counter.in_count + counter.out_count}")


if __name__ == "__main__":
    main()
