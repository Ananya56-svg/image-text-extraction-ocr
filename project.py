import os
import cv2
import easyocr
import matplotlib.pyplot as plt

# ===============================
# CONFIGURATION
# ===============================
dataset_folders = ["images/train", "images/test"]
OCR_RESULTS_FILE = "ocr_results.txt"
GROUND_TRUTH_FILE = "ground_truth.txt"

RUN_OCR = not os.path.exists(OCR_RESULTS_FILE)

# ===============================
# OUTPUT DIRECTORIES
# ===============================
os.makedirs("output_images/original", exist_ok=True)
os.makedirs("output_images/ocr_result", exist_ok=True)
os.makedirs("output_images/plots", exist_ok=True)

# ===============================
# INITIALIZE OCR
# ===============================
reader = easyocr.Reader(['en'], gpu=False)

# ===============================
# STEP 1: OCR FOR ALL IMAGES (RUN ONCE)
# ===============================
if RUN_OCR:
    print("\n--- OCR results not found. Running OCR on all images ---")
    ocr_file = open(OCR_RESULTS_FILE, "w", encoding="utf-8")

    image_counter = 0

    for folder in dataset_folders:
        if not os.path.exists(folder):
            continue

        for image_name in os.listdir(folder):
            image_path = os.path.join(folder, image_name)
            img = cv2.imread(image_path)

            if img is None:
                continue

            image_counter += 1
            print(f"[{image_counter}] OCR → {image_name}")

            results = reader.readtext(img)
            extracted_text = " ".join([r[1] for r in results])

            ocr_file.write(f"\nImage: {image_name}\n")
            if extracted_text.strip() == "":
                ocr_file.write("NO TEXT DETECTED\n")
            else:
                ocr_file.write(extracted_text + "\n")

    ocr_file.close()
    print("OCR completed and saved.\n")

else:
    print("\n--- OCR results already exist. Skipping OCR step ---\n")

# ===============================
# ACCURACY FUNCTION
# ===============================
def calculate_accuracy(gt, ocr):
    gt = gt.lower().strip()
    ocr = ocr.lower().strip()

    if len(gt) == 0:
        return 0.0

    correct = sum(1 for g, o in zip(gt, ocr) if g == o)
    return (correct / len(gt)) * 100

# ===============================
# STEP 2: LOAD FILES
# ===============================
with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as gt_file:
    gt_lines = gt_file.readlines()

with open(OCR_RESULTS_FILE, "r", encoding="utf-8") as ocr_file:
    ocr_data = ocr_file.read()

accuracy_values = []
image_names = []

print("\n--- Accuracy Results (Manual Ground Truth) ---")

# ===============================
# STEP 3: COMPARE GT vs OCR
# ===============================
for line in gt_lines:
    if ":" not in line:
        continue

    image_name, gt_text = line.split(":", 1)
    image_name = image_name.strip()
    gt_text = gt_text.strip()

    marker = f"Image: {image_name}"
    start = ocr_data.find(marker)

    if start == -1:
        print(f"{image_name} → OCR NOT FOUND")
        continue

    end = ocr_data.find("Image:", start + 1)
    ocr_block = ocr_data[start:end] if end != -1 else ocr_data[start:]

    ocr_lines = ocr_block.splitlines()
    ocr_text = " ".join(ocr_lines[1:]).strip()

    acc = calculate_accuracy(gt_text, ocr_text)
    accuracy_values.append(acc)
    image_names.append(image_name)

    print(f"{image_name} → Accuracy: {acc:.2f}%")

# ===============================
# VISUAL 1: ACCURACY BAR CHART
# ===============================
plt.figure(figsize=(12, 6))
plt.bar(range(len(accuracy_values)), accuracy_values)
plt.xlabel("Images")
plt.ylabel("Accuracy (%)")
plt.title("OCR Accuracy on 50 Manually Verified Images")
plt.tight_layout()
plt.savefig("output_images/plots/accuracy_bar_chart.png")
plt.close()

# ===============================
# VISUAL 2: ACCURACY DISTRIBUTION HISTOGRAM
# ===============================
plt.figure(figsize=(8, 6))
plt.hist(accuracy_values, bins=10)
plt.xlabel("Accuracy (%)")
plt.ylabel("Number of Images")
plt.title("Distribution of OCR Accuracy")
plt.tight_layout()
plt.savefig("output_images/plots/accuracy_distribution_histogram.png")
plt.close()

# ===============================
# VISUAL 3: OCR IMAGES WITH ACCURACY TEXT
# ===============================
print("\n--- Saving OCR images with accuracy overlay ---")

for img_name, acc in zip(image_names, accuracy_values):
    found = False

    for folder in dataset_folders:
        img_path = os.path.join(folder, img_name)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            found = True
            break

    if not found or img is None:
        continue

    cv2.putText(
        img,
        f"Accuracy: {acc:.2f}%",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )

    cv2.imwrite(f"output_images/ocr_result/{img_name}", img)

# ===============================
# FINAL RESULT
# ===============================
if accuracy_values:
    avg_accuracy = sum(accuracy_values) / len(accuracy_values)
    print(f"\nAverage OCR Accuracy (50 images): {avg_accuracy:.2f}%")

print("\nPROCESS COMPLETED SUCCESSFULLY ✅")
print("Visuals generated:")
print("1. output_images/plots/accuracy_bar_chart.png")
print("2. output_images/plots/accuracy_distribution_histogram.png")
print("3. output_images/ocr_result/ (images with accuracy text)")

# ===============================
# DAY 6: LIVE CAMERA OCR MODULE
# ===============================

# ===============================
# DAY 6: STABLE LIVE CAMERA OCR
# ===============================

# ===============================
# DAY 7: HARDENED LIVE CAMERA OCR
# ===============================

RUN_CAMERA_OCR = True

if RUN_CAMERA_OCR:
    print("\n--- Advanced Live Camera OCR ---")
    print("Controls:")
    print("O → Toggle OCR ON/OFF")
    print("S → Save Snapshot")
    print("Q → Quit")

    os.makedirs("output_images/camera_snaps", exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ERROR: Camera not accessible")
    else:
        ocr_enabled = True
        frame_count = 0
        ocr_results = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Run OCR every 5 frames (FPS control)
            if ocr_enabled and frame_count % 5 == 0:
                ocr_results = reader.readtext(frame)

            # Draw OCR results
            for r in ocr_results:
                bbox, text, conf = r
                (tl, tr, br, bl) = bbox
                tl = tuple(map(int, tl))
                br = tuple(map(int, br))

                cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{text} ({conf:.2f})",
                    (tl[0], tl[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # Instructions overlay
            cv2.putText(
                frame,
                "O: OCR ON/OFF | S: Save | Q: Quit",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow("Advanced Live OCR", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('o'):
                ocr_enabled = not ocr_enabled
                ocr_results = []
                print(f"OCR Enabled: {ocr_enabled}")
            elif key == ord('s'):
                snap_name = f"output_images/camera_snaps/snap_{frame_count}.png"
                cv2.imwrite(snap_name, frame)
                print(f"Snapshot saved: {snap_name}")

        cap.release()
        cv2.destroyAllWindows()

    print("Advanced Camera OCR Stopped.")




