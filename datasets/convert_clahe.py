import os
import cv2 as cv

def apply_clahe_to_folder(input_folder, output_folder):
    """
    Apply CLAHE to all images in the input folder and save them to the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Check if the file is an image
        if filename.lower().endswith(('.png')):
            # Read the image in grayscale
            image = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Skipping invalid image file: {input_path}")
                continue

            # Apply CLAHE
            clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
            clahe_image = clahe.apply(image)

            # Save the processed image
            cv.imwrite(output_path, clahe_image)
            print(f"Processed and saved: {output_path}")

def process_brain_tumor_segmentation(input_base_folder, output_base_folder):
    """
    Process the images in the input base folder and save them to the output base folder.
    """
    subfolders = ['images', 'val_images']

    for subfolder in subfolders:
        input_folder = os.path.join(input_base_folder, subfolder)
        output_folder = os.path.join(output_base_folder, subfolder)

        if os.path.exists(input_folder):
            apply_clahe_to_folder(input_folder, output_folder)
        else:
            print(f"Folder not found: {input_folder}")

if __name__ == "__main__":
    input_base_folder = "Modified_2_Brain_Tumor_Segmentation"
    output_base_folder = "Modified_3_Brain_Tumor_Segmentation"

    process_brain_tumor_segmentation(input_base_folder, output_base_folder)