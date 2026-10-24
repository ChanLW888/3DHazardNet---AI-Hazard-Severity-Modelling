import os
import shutil
import numpy as np

# ============================================================
# CONFIG
# ============================================================

INPUT_ROOT = "/home/lcha0115/bt60_scratch/lcha_data/map-anything/personal_data/hazard_test_101/v5_priors_letterboxed"

OUTPUT_ROOT = "/home/lcha0115/bt60_scratch/lcha_data/map-anything/personal_data/hazard_test_101/v5_priors_all4"

CAMERAS = ["CAM2", "CAM6"]

# Required classes:
# 2 = Loose Trash
# 3 = Mud Spill
# 4 = Public Greenway
# 5 = Sidewalk

TARGET_CHANNELS = {
    2: "loose_trash",
    3: "mud_spill",
    4: "public_greenway",
    5: "sidewalk",
}


# ============================================================
# PROCESS
# ============================================================

total_files = 0
selected_files = 0

for camera in CAMERAS:

    input_dir = os.path.join(INPUT_ROOT, camera)
    output_dir = os.path.join(OUTPUT_ROOT, camera)

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(input_dir):
        print(f"\nWARNING: Directory does not exist:")
        print(f"  {input_dir}")
        continue

    print(f"\n{'=' * 70}")
    print(f"Processing {camera}")
    print(f"{'=' * 70}")

    camera_total = 0
    camera_selected = 0

    for filename in sorted(os.listdir(input_dir)):

        if not filename.endswith(".npz"):
            continue

        camera_total += 1
        total_files += 1

        input_path = os.path.join(input_dir, filename)

        try:
            data = np.load(input_path)
            prior = data["prior"]

            # Check that ALL FOUR required classes are present
            has_all_four = all(
                np.any(prior[channel] > 0)
                for channel in TARGET_CHANNELS
            )

            if has_all_four:

                output_path = os.path.join(output_dir, filename)

                # Copy original NPZ unchanged
                shutil.copy2(input_path, output_path)

                camera_selected += 1
                selected_files += 1

                print(
                    f"[SELECTED] {filename} "
                    f"-> Loose Trash + Mud Spill + "
                    f"Public Greenway + Sidewalk"
                )

        except Exception as e:

            print(f"[ERROR] {filename}: {e}")

    print(f"\n{camera} summary:")
    print(f"  Total NPZ files : {camera_total}")
    print(f"  All 4 classes   : {camera_selected}")
    print(f"  Not selected    : {camera_total - camera_selected}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print(f"\n{'=' * 70}")
print("FINAL SUMMARY")
print(f"{'=' * 70}")

print(f"Total files scanned : {total_files}")
print(f"Files with ALL 4    : {selected_files}")
print(f"Files not selected  : {total_files - selected_files}")

print("\nOutput directory:")
print(OUTPUT_ROOT)