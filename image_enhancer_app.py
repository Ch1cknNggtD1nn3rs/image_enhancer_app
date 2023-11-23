#!/usr/bin/env python3
# image enhancer with UI

from tkinter import messagebox
from tkinter import filedialog
import customtkinter as ctk
import shutil
import os
import threading
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

device = torch.device('cpu')

def select_images():
    # Open a file dialog and get the selected files
    file_paths = filedialog.askopenfilenames(filetypes=[('Image Files', '*.png;*.jpg;*.jpeg;*.gif;*.bmp')])

    # Create the LR directory if it doesn't exist
    if not os.path.exists('LR'):
        os.makedirs('LR')

    # Copy the selected files to the LR directory
    for file_path in file_paths:
        shutil.copy(file_path, 'LR')


def run_app():
    # Get all image files in the LR directory
    test_img_folder = 'LR/*'
    model_choice = str(model_var.get())
    model_path = "models/RRDB_"+model_choice+"_x4.pth"
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    print('Model path {:s}. \nTesting...'.format(model_path))

    # Create the Results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    idx = 0
    for path in glob.glob(test_img_folder):
        idx += 1
        base = osp.splitext(osp.basename(path))[0]
        print(idx, base)
        # read images
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        cv2.imwrite('results/{:s}_rlt.png'.format(base), output)

    # Once task is done, stop the progress bar, hide it and show a message box
    progress_bar.stop()
    progress_bar.pack_forget()
    messagebox.showinfo("...alakazam!", "Congratulations! The image enhancement process has concluded successfully.")


def start_task():
    # Show the progress bar
    progress_bar.pack()
    # Start the progress bar
    progress_bar.start()
    # Start the long running task in a separate thread so it doesn't block the GUI
    threading.Thread(target=run_app).start()



ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title('Pixel Perfection: Your Image Enhancement Wizard')
root.geometry('750x500')


step_zero_label = ctk.CTkLabel(root, text="Enlarge and enhance any '*.png;*.jpg;*.jpeg' image by 4x!", font=("Comic Sans MS", 18, "bold"))
step_zero_label.pack(padx=20, pady=30)

step_one_label = ctk.CTkLabel(root, text="Step 1. Choose your pics for a magic touch!", font=("Helvetica",14, "bold"))
step_one_label.pack(padx=200, pady=10, anchor='w')

# Create a button that will call select_images when clicked
select_button = ctk.CTkButton(root, text="Select Image(s)", command=select_images)
select_button.pack(pady = 5)

step_two_label = ctk.CTkLabel(root, text="\nStep 2. Time to choose your magic wand!", font=("Helvetica",14, "bold"))
step_two_label.pack(padx=200, pady=10, anchor='w')

model_var = ctk.StringVar(value="ESRGAN")
model_dropdown = ctk.CTkOptionMenu(root, values = ["ESRGAN", "PSNR"], variable=model_var)
model_dropdown.pack(pady=5)

step_three_label = ctk.CTkLabel(root, text="\nStep 3. Unleash the magic!", font=("Helvetica",14, "bold"))
step_three_label.pack(padx=200, pady=10, anchor='w')

run_button = ctk.CTkButton(root, text="Abracadabra!", font=("Comic Sans MS", 14), command=start_task)
run_button.pack(pady=5)

# Create a progress bar but don't pack it yet
progress_bar = ctk.CTkProgressBar(root, mode='indeterminate')

last_label = ctk.CTkLabel(root, text="\n\nPlease wait a few minutes as we work our digital magic!\nYour enhanced image(s) will be waiting for you in the ‘results’ folder.", font=("Helvetica", 12, "italic"))
last_label.pack(padx=20, pady=30)

root.mainloop()