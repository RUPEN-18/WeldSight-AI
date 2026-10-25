from pathlib import Path
import shutil


DATASET = Path("dataset")
SPLITS = ["train", "valid", "test"]


def polygon_to_box(values):
    # values = x1 y1 x2 y2 x3 y3 ...
    xs = values[0::2]
    ys = values[1::2]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return x_center, y_center, width, height


for split in SPLITS:
    labels_folder = DATASET / split / "labels"

    if not labels_folder.exists():
        print(f"{split}: labels folder not found")
        continue

    # Backup original labels
    backup_folder = DATASET / split / "labels_backup"

    if not backup_folder.exists():
        shutil.copytree(labels_folder, backup_folder)
        print(f"Backup created: {backup_folder}")

    converted_files = 0
    converted_rows = 0
    bad_rows = 0

    for label_file in labels_folder.glob("*.txt"):

        new_lines = []

        with open(label_file, "r", encoding="utf-8") as file:

            for line in file:
                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                try:
                    class_id = int(float(parts[0]))
                    coordinates = list(map(float, parts[1:]))

                    # Already normal YOLO detection format
                    if len(coordinates) == 4:
                        new_lines.append(
                            f"{class_id} "
                            + " ".join(f"{x:.6f}" for x in coordinates)
                        )

                    # Segmentation polygon
                    elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:

                        x, y, w, h = polygon_to_box(coordinates)

                        # Keep values within YOLO normalized range
                        x = max(0, min(1, x))
                        y = max(0, min(1, y))
                        w = max(0, min(1, w))
                        h = max(0, min(1, h))

                        if w > 0 and h > 0:
                            new_lines.append(
                                f"{class_id} "
                                f"{x:.6f} "
                                f"{y:.6f} "
                                f"{w:.6f} "
                                f"{h:.6f}"
                            )

                            converted_rows += 1

                    else:
                        bad_rows += 1
                        print(
                            f"Skipped malformed row in {label_file.name}: {line}"
                        )

                except ValueError:
                    bad_rows += 1
                    print(
                        f"Skipped invalid row in {label_file.name}: {line}"
                    )

        with open(label_file, "w", encoding="utf-8") as file:
            file.write("\n".join(new_lines))

        converted_files += 1

    print(f"\n{split.upper()}")
    print(f"Files processed: {converted_files}")
    print(f"Polygon rows converted: {converted_rows}")
    print(f"Bad rows skipped: {bad_rows}")
    print("-" * 50)


print("\nLABEL CONVERSION COMPLETE")