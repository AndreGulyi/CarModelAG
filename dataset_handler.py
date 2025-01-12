# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 23/12/24
import logging
import os
import pandas as pd
import json



import os
import json
from io import BytesIO
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch._higher_order_ops.flex_attention import trace_flex_attention_backward
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from torchvision import transforms
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import matplotlib.pyplot as plt

from carmodel import invalid_combinations, category_map

from pathlib import Path

def create_pdf_with_images(images, output_pdf, rows=5, cols=1, cell_width=200, cell_height=150, title="", summery=None,l=None):
    file_path = Path(output_pdf)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    page_width, page_height = letter
    pdf = canvas.Canvas(output_pdf, pagesize=letter)
    x_margin = 20
    y_margin = 35
    x_spacing = 10
    y_spacing = 35

    images_per_page = rows * cols
    cell_width = (page_width - x_margin * 2 - (cols - 1) * x_spacing) / cols
    cell_height = (page_height - y_margin * 2 - (rows - 1) * y_spacing) / rows

    # Calculate overall summary
    total_images = len(images)
    categories_set = set()
    for info in images:
        if isinstance(info["category"], list):
            categories_set.update(set(info["category"]))
        elif info["category"]:
            categories_set.add(info["category"])
    # categories_set = set(list(info["category"]) if isinstance(info["category"],list) else info["category"] for info in images if info["category"] is not None)
    labels_set = set(label for info in images for label in info.get("labels", []))

    # Add overall summary at the start of the PDF
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(colors.black)
    pdf.drawString(x_margin, page_height - 20, f"Overall Summary{title}:")

    pdf.setFont("Helvetica", 10)
    pdf.drawString(x_margin, page_height - 40, f"Total images: {total_images}")
    pdf.drawString(x_margin, page_height - 55, f"Categories: {', '.join(categories_set) if categories_set else 'None'}")
    pdf.drawString(x_margin, page_height - 70, f"Unique Labels: {', '.join(labels_set) if labels_set else 'None'}")

    # Move down for images to be added below the summary
    y_position = page_height - 100  # Adjust for space taken by summary

    # Loop through images and add them to the PDF
    for page_start in range(0, len(images), images_per_page):
        current_page_images = images[page_start:page_start + images_per_page]

        # Draw grid of images and their labels
        for i, image_info in enumerate(current_page_images):
            col = i % cols
            row = i // cols
            x = x_margin + col * (cell_width + x_spacing)
            y = y_position - (row + 1) * cell_height - row * y_spacing

            image_path = image_info['filename']
            labels = ", ".join(image_info.get('labels', []))
            category = image_info["category"]

            try:
                # Add the image
                img = Image.open(image_path)
                # img.thumbnail((int(cell_width), int(cell_height)))
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                buffer.seek(0)

                # Create an ImageReader object
                img_reader = ImageReader(buffer)

                # Draw the image on the PDF
                pdf.drawImage(img_reader, x, y, width=cell_width, height=cell_height)

                # Add the filename and labels below the image
                text_x = x + 5
                text_y = y - 10

                pdf.setFont("Helvetica", 8)
                pdf.setFillColor(colors.black)

                pdf.drawString(text_x, text_y, f"Name: {os.path.basename(image_path)}")
                pdf.drawString(text_x, text_y - 10, f"Labels: {labels}")
                pdf.drawString(text_x, text_y - 20, f"Category: {category}")
            except Exception as e:
                logging.debug(f"Error loading image {image_path}: {e}")

        # Add page number (bottom right)
        pdf.setFont("Helvetica", 8)
        pdf.drawString(page_width - 100, 30, f"Page {int(page_start / images_per_page) + 1}")

        # Start a new page after filling one page of images
        pdf.showPage()
    if summery:
        max_chunk_size = 200
        # Split the summary text into chunks
        chunks = [summery[i:i + max_chunk_size] for i in range(0, len(summery), max_chunk_size)]
        start_y = page_height - 120  # Adjust Y position for summary

        # Add each chunk of text to the PDF
        for chunk in chunks:
            pdf.setFont("Helvetica", 10)
            pdf.drawString(x_margin, start_y, chunk)
            start_y -= 15  # Adjust spacing between chunks
    # Save the PDF file
    pdf.save()
    logging.debug(f"pdf created: {output_pdf}")
def show_images_in_grid(images, rows=5, cols=10):
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    axes = axes.flatten()

    for ax, image_info in zip(axes, images):
        image_path = image_info['filename']
        labels = ", ".join(image_info.get('labels', []))

        try:
            img = Image.open(image_path)
            ax.imshow(img)
            ax.set_title(f"{os.path.basename(image_path)}\n{labels}", fontsize=8)
            ax.axis('off')
        except Exception as e:
            ax.set_visible(False)
            logging.debug(f"Error loading image {image_path}: {e}")

    # Hide any remaining empty subplots
    for ax in axes[len(images):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def find_closest_category(parts):
    closest_category = None
    max_score = 0

    for category, keywords in category_map.items():
        # Calculate the score: number of matching keywords in parts
        score = sum(1 for keyword in keywords if keyword in parts)
        if score > max_score:
            max_score = score
            closest_category = category

    return closest_category
def prepare_dataset(dataset_path):
    """
    Prepares the dataset by processing the annotations and creating a DataFrame
    containing image paths and their corresponding labels.
    """
    multi_category_images = []
    empty_category_images = []
    data = []
    # groups = ["6125c98e7eb77b4a469ef416"]  # Example, use all groups in your real dataset
    groups = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    for group in groups:
        group_path = os.path.join(dataset_path, group)
        region_file = os.path.join(group_path, "via_region_data.json")

        # Load and process via_region_data.json
        if not os.path.exists(region_file):
            continue

        with open(region_file, "r") as f:
            regions_data = json.load(f)

        for k, v in regions_data.items():
            image_path = os.path.join(group_path, v["filename"])
            if not os.path.exists(image_path):
                continue

            categories = set()
            parts = []
            if not v["regions"]:
                continue
            for r in v["regions"]:
                identity = r["region_attributes"].get("identity", "")
                parts.append(identity)

            for category, keywords in category_map.items():
                if all(keyword in parts for keyword in keywords):
                    categories.add(category)
            # Assign all categories if multiple, else skip if uncategorized
            if categories:
                for combination in invalid_combinations:
                    if combination.issubset(categories):
                        logging.debug(f"Invalid labeling found: Categories: {', '.join(categories)} in image {image_path}")
                        continue
                if "front" in "".join(categories) and "rear" in "".join(categories):
                    logging.debug(f"Invalid data labling found front&rear in same image {image_path}: {''.join(categories)}")
                    continue
                data.append({"filename": image_path, "labels": list(categories)})


                if len(categories)>1:
                    multi_category_images.append({"filename": image_path, "labels": missing_parts,
                                                  "category": ", ".join(categories)})
                    logging.debug("multi cateogry: ",categories, image_path)

            else:
                # if False:
                # Image.open(image_path).show()
                logging.debug("empty category ",image_path, parts)
                if not categories: # Empty categories
                    closed_category = find_closest_category(parts)
                    if closed_category:
                        missing_parts = [pp for pp in category_map[closed_category] if pp not in parts]
                    else:
                        missing_parts = []
                    empty_category_images.append({"filename": image_path, "labels": missing_parts,
                                                  "category": closed_category})

    # Convert to DataFrame
    df = pd.DataFrame(data)
    # if empty_category_images:
    #     show_images_in_grid(empty_category_images, rows=5, cols=10)
    output_pdf_path = "reports/empty_category_images.pdf"
    if empty_category_images:
        create_pdf_with_images(empty_category_images, output_pdf_path)
        logging.debug(f"PDF created with empty category images: {output_pdf_path}")
    if multi_category_images:
        create_pdf_with_images(multi_category_images, "reports/multi_category_images.pdf", rows=5, cols=2)

    return df
