# Interactive Notch Filtering GUI.
# Lets a user draw notches on an image's frequency spectrum to vizualize how different notch placements can be used to filter an image.
# Created by: Daniel Gore.

# Libs
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageGrab
import numpy as np
from notch_lib import * # Custom lib with filtering operations

# GUI frontend + backend calls
class InteractiveNotchGUI:
    def __init__(self, root):
        self.window = root
        self.window.title("Image Loader")
        self.window.state("zoomed")

        # Image params
        self.original_image = None
        self.img_width = None
        self.img_height = None

        # User drawing params
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.line_width = 5 # Default, changable with slider
        self.drawing_coords = []

        # GUI init
        self.main_frame = tk.Frame(self.window)
        self.main_frame.pack(expand=True)

        # Frontend tools
        self.load_button = tk.Button(self.main_frame, text = "Load Image", command = self.load_image)
        self.load_button.grid(row = 0, column = 0, padx = 5, pady = 10)

        self.noise_button = tk.Button(self.main_frame, text = "Add Noise", command = self.add_noise)
        self.noise_button.grid(row = 0, column = 1, padx = 5, pady = 10)

        self.reset_button = tk.Button(self.main_frame, text="Reset", command=self.reset_image)
        self.reset_button.grid(row = 0, column = 2, padx = 5, pady = 10)

        self.line_width_scale = tk.Scale(self.main_frame, from_ = 1, to_ = 50, orient = tk.HORIZONTAL, label = "Line Width", command = self.update_line_width)
        self.line_width_scale.set(self.line_width)
        self.line_width_scale.grid(row = 0, column = 3, padx = 5, pady = 10, sticky = "ew")

        self.copy_button = tk.Button(self.main_frame, text = "Generate Filtered Image", command = self.copy_to_canvas4)
        self.copy_button.grid(row = 0, column = 4, padx = 5, pady = 10)

        self.design_notches = tk.Button(self.main_frame, text = "Design Notches", command = self.design_notch)
        self.design_notches.grid(row = 0, column = 5, padx = 5, pady = 10)

        # Four canvases that contain origional image, FFTs, and filtered image
        self.canvas_frame = tk.Frame(self.main_frame)
        self.canvas_frame.grid(row = 1, column = 0, columnspan = 4)

        self.canvas1 = tk.Canvas(self.canvas_frame, bg = "gray", width = 350, height = 350)
        self.canvas1.grid(row = 0, column = 0, padx = 10, pady = 10)

        self.canvas2 = tk.Canvas(self.canvas_frame, bg = "gray", width = 350, height = 350)
        self.canvas2.grid(row = 0, column = 1, padx = 10, pady = 10)
        self.canvas2.bind("<Button-1>", self.start_draw)
        self.canvas2.bind("<B1-Motion>", self.draw)

        self.canvas3 = tk.Canvas(self.canvas_frame, bg = "gray", width = 350, height = 350)
        self.canvas3.grid(row = 1, column = 0, padx = 10, pady = 10)

        self.canvas4 = tk.Canvas(self.canvas_frame, bg = "gray", width = 350, height = 350)
        self.canvas4.grid(row = 1, column = 1, padx = 10, pady = 10)

        # Storage for FFT
        self.ft_image_tk = None

        # Storage for notch design
        self.filter_min = 0
        self.filter_max = 255
        self.std_dev = 1.0
        self.notch_token = 0

    def load_image(self):

        filepath = filedialog.askopenfilename(filetypes = [("Image files", "*.png;*.jpg;*.jpeg")])
        if filepath:
            # Load image + params
            self.original_image = Image.open(filepath).convert("L") # Convert to grayscale
            self.img_width, self.img_height = self.original_image.size
            window_width = self.window.winfo_width()
            window_height = self.window.winfo_height()

            # Calibration constant -> used to change aspect ratio of canvases upon loading an image
            calib_const = 2.4

            # Resize canvases
            if calib_const * self.img_width > window_width or calib_const * self.img_height > window_height:
                # Shrink scale is image > origional box size
                scale_factor = min(window_width / (calib_const * self.img_width), window_height / (calib_const * self.img_height))
                self.new_width = int(self.img_width * scale_factor)
                self.new_height = int(self.img_height * scale_factor)
            else:
                # Expand image otherwise
                max_width = window_width // 2
                max_height = window_height // 2
                aspect_ratio = self.img_width / self.img_height

                if self.img_width > self.img_height:
                    self.new_width = min(max_width, self.img_width)
                    self.new_height = int(self.new_width / aspect_ratio)
                else:
                    self.new_height = min(max_height, self.img_height)
                    self.new_width = int(self.new_height * aspect_ratio)

            # Apply resizing parameters on the canvases when an image is loaded in
            image_resized = self.original_image.resize((self.new_width, self.new_height))
            img_tk = ImageTk.PhotoImage(image_resized)

            self.canvas1.create_image(0, 0, anchor = tk.NW, image = img_tk)
            self.canvas1.image = img_tk
            self.display_fourier_transform(image_resized)

            self.canvas1.config(width = self.new_width, height = self.new_height)
            self.canvas2.config(width = self.new_width, height = self.new_height)
            self.canvas3.config(width = self.new_width, height = self.new_height)
            self.canvas4.config(width = self.new_width, height = self.new_height)

    def add_noise(self):

        # Check if an image is loaded into canvas 1 -> requirement for added noise
        if not hasattr(self.canvas1, 'image'):
            return

        # Create a new window for noise selection
        noise_window = tk.Toplevel(self.window)
        noise_window.title("Add Noise")
        
        # Add a slider for noise level (in dB)
        tk.Label(noise_window, text="Noise Level (dB)").pack(padx = 10, pady = 5)
        noise_slider = tk.Scale(noise_window, from_ = -10, to = 50, orient = tk.HORIZONTAL, resolution = 1)
        noise_slider.set(0) # Default slider @ 0dB
        noise_slider.pack(padx = 10, pady = 10)
        
        def apply_noise():

            # Get the current image from canvas1
            img = self.original_image.resize((self.new_width, self.new_height))
            img_array = np.array(img, dtype = np.float32)
            
            # Convert dB to linear scale for noise variance
            noise_level_db = noise_slider.get()
            self.noise_variance = 10 ** (noise_level_db / 10)  # Store the noise variance
            
            # Generate Gaussian noise
            noise = np.random.normal(0, np.sqrt(self.noise_variance), img_array.shape)
            noisy_img_array = img_array + noise
            
            # Clip values to 0-255 range and convert back to uint8
            noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)
            noisy_img = Image.fromarray(noisy_img_array)
            
            # Update canvas1 with noisy image
            img_tk = ImageTk.PhotoImage(noisy_img)
            self.canvas1.create_image(0, 0, anchor = tk.NW, image = img_tk)
            self.canvas1.image = img_tk
            
            # Update the Fourier transforms
            self.display_fourier_transform(noisy_img)
            
            noise_window.destroy()
        
        def apply_noise():

            # Current image from canvas1
            img = self.original_image.resize((self.new_width, self.new_height))
            img_array = np.array(img, dtype = np.float32)
            
            # dB -> linear scale and generate noise on 8 bit scale
            noise_level_db = noise_slider.get()
            noise_variance = 10 ** (noise_level_db / 10)
            noise = np.random.normal(0, np.sqrt(noise_variance), img_array.shape)
            noisy_img_array = img_array + noise
            noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)
            noisy_img = Image.fromarray(noisy_img_array)
            
            # Update canvases with noise
            img_tk = ImageTk.PhotoImage(noisy_img)
            self.canvas1.create_image(0, 0, anchor = tk.NW, image = img_tk)
            self.canvas1.image = img_tk
            self.display_fourier_transform(noisy_img)
            
            noise_window.destroy()
    
        # Save button
        save_button = tk.Button(noise_window, text = "Save", command = apply_noise)
        save_button.pack(padx = 10, pady = 10)

    def reset_image(self):

        # Reset canvas1 to original image
        if self.original_image:
            image_resized = self.original_image.resize((self.new_width, self.new_height))
            img_tk = ImageTk.PhotoImage(image_resized)
            
            self.canvas1.delete("all")
            self.canvas1.create_image(0, 0, anchor = tk.NW, image = img_tk)
            self.canvas1.image = img_tk
            
            self.display_fourier_transform(image_resized)
        
        self.canvas2.delete("drawing")
        self.canvas4.delete("all")
        self.drawing_coords = []

    def display_fourier_transform(self, image):

        image_array = np.array(image)
        f_transform = np.fft.fft2(image_array)
        self.f_transform_shifted = np.fft.fftshift(f_transform)

        # Display FFT magnitdue
        magnitude = np.abs(self.f_transform_shifted)
        magnitude_log = np.log(magnitude + 1)
        magnitude_normalized = np.uint8(255 * magnitude_log / np.max(magnitude_log))
        ft_image = Image.fromarray(magnitude_normalized)
        ft_img_tk = ImageTk.PhotoImage(ft_image)

        # Set the FFT on the 2nd and 3rd canvases
        self.canvas2.create_image(0, 0, anchor = tk.NW, image = ft_img_tk)
        self.canvas2.image = ft_img_tk 

        self.ft_image_tk = ft_img_tk
        self.canvas3.create_image(0, 0, anchor = tk.NW, image = ft_img_tk)
        self.canvas3.image = ft_img_tk

    def start_draw(self, event):

        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
        self.drawing_coords.append((event.x, event.y, self.line_width))

    def draw(self, event):

        # Draw using mouse pointer
        if self.drawing:
            self.canvas2.create_line(self.last_x, self.last_y, event.x, event.y, width = self.line_width, fill = "black", capstyle = tk.ROUND, smooth = tk.TRUE, tags = "drawing")
            self.last_x = event.x
            self.last_y = event.y
            self.drawing_coords.append((event.x, event.y, self.line_width))

    def update_line_width(self, value):

        self.line_width = int(value)

    def copy_to_canvas4(self):

        self.canvas4.delete("all")

        # Plot the filtered img
        if self.drawing_coords:
            self.x_coords = [point[0] for point in self.drawing_coords]
            self.y_coords = [self.canvas2.winfo_height() - point[1] for point in self.drawing_coords]
            self.line_widths = [point[2] for point in self.drawing_coords]

            # Generate the notch matrix
            if self.notch_token == 0:
                notch_mat = notch_matrix(self.x_coords, self.y_coords, self.line_widths, [self.new_width, self.new_height])
                notch_mat = np.transpose(notch_mat)
                self.notch_mat_fin = np.flipud(notch_mat)
            else:
                self.notch_token = 0

            # Copy the FFT over and multiply by notch matrix, then take the IFFT
            if hasattr(self.canvas2, 'image'):
                new_fft = self.f_transform_shifted * self.notch_mat_fin
                img_filtered = np.fft.ifftshift(new_fft)
                img_filtered = np.fft.ifft2(img_filtered)
                img_filtered = np.abs(img_filtered)

                # Normalize image and convert to uint8
                img_filtered_normalized = (img_filtered / np.max(img_filtered)) * 255
                img_filtered_normalized = img_filtered_normalized.astype(np.uint8)
                img_filtered_pil = Image.fromarray(img_filtered_normalized)
                img_spec = ImageTk.PhotoImage(img_filtered_pil)

                # Display on canvas4
                self.canvas4.create_image(0, 0, anchor = tk.NW, image = img_spec)
                self.canvas4.image = img_spec

                # Calculate and display SNR if noise was added
                if hasattr(self, 'noise_variance'):
                    # Get the original image (signal)
                    original_img = np.array(self.original_image.resize((self.new_width, self.new_height)), dtype=np.float32)
                    
                    # Calculate SNR (dB)
                    signal_power = np.mean(original_img**2)
                    noise_power = self.noise_variance
                    if noise_power > 0:
                        snr_linear = signal_power / noise_power
                        snr_db = 10 * np.log10(snr_linear)
                        print(f"Output Image SNR: {snr_db:.2f} dB")

    def design_notch(self):

        # Create a new window for the notch filter settings
        notch_window = tk.Toplevel(self.window)
        notch_window.title("Set Notch Filter Parameters")

        # Add a filter type option (Ideal or Gaussian)
        filter_type_label = tk.Label(notch_window, text = "Filter Type:")
        filter_type_label.pack(padx = 10, pady = 5)

        filter_type_var = tk.StringVar(value = "ideal") # Default type is ideal -> must manually switch to Gaussian
        filter_type_menu = tk.OptionMenu(notch_window, filter_type_var, "ideal", "gaussian")
        filter_type_menu.pack(padx = 10, pady = 5)

        # Min and Max sliders
        tk.Label(notch_window, text = "Min Filter Value").pack(padx = 10, pady = 5)
        min_slider = tk.Scale(notch_window, from_ = 0, to = 255, orient = tk.HORIZONTAL)
        min_slider.set(self.filter_min)
        min_slider.pack(padx = 10, pady = 5)

        tk.Label(notch_window, text = "Max Filter Value").pack(padx = 10, pady = 5)
        max_slider = tk.Scale(notch_window, from_ = 0, to = 255, orient = tk.HORIZONTAL)
        max_slider.set(self.filter_max)
        max_slider.pack(padx = 10, pady = 5)

        # Standard deviation field (only for Gaussian)
        std_dev_label = tk.Label(notch_window, text = "Standard Deviation:")
        std_dev_entry = tk.Entry(notch_window)
        std_dev_entry.insert(0, str(self.std_dev))

        # Function to hide the standard deviation query if the filter type is ideal
        def update_sigma_visibility(*args):

            if filter_type_var.get() == "ideal":
                std_dev_label.pack_forget()
                std_dev_entry.pack_forget()
            else:
                std_dev_label.pack(padx = 10, pady = 5) 
                std_dev_entry.pack(padx = 10, pady = 5)

        update_sigma_visibility()
        filter_type_var.trace("w", update_sigma_visibility)

        # Save button to change filter parameters
        def save_params():

            self.filter_min = min_slider.get()
            self.filter_max = max_slider.get()
            if filter_type_var.get() == "gaussian":
                self.std_dev = float(std_dev_entry.get())
            else:
                self.std_dev = None
            notch_window.destroy()

            # New notch matrix to adjust notch structure at each x, y center point
            notch_mat = designed_notch_matrix(self.x_coords, self.y_coords, self.line_widths, [self.new_width, self.new_height], self.filter_min, self.filter_max, self.std_dev)
            notch_mat = np.transpose(notch_mat)
            self.notch_mat_fin = np.flipud(notch_mat)
            self.notch_token = 1

        save_button = tk.Button(notch_window, text = "Save", command = save_params)
        save_button.pack(padx = 10, pady = 20)

# Create an instance of the GUI
GUI = tk.Tk()
instance_image_loader = InteractiveNotchGUI(GUI)
GUI.mainloop()
