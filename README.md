# AutomaticLicensePlateRecognition

## Overview

**Project_display** is a video display program designed to capture and showcase live video streams from your camera. Once running, the program automatically starts with the default camera. Users can press the **C** key to switch between different cameras.

## Features

- **Real-Time Video Display:** Automatically captures and displays video from the camera.
- **Camera Switching:** Quickly switch between connected cameras by pressing the **C** key.
- **Ease of Use:** Simple operation and intuitive controls for fast deployment and usage.

## Installation & Dependencies

Before running the program, please ensure you have the following installed:

- **Python 3.6** or higher.
- **OpenCV** library
Installation via pip:
    
    ```bash
    bash
    复制
    pip install opencv-python
    
    ```
    
- (Optional) Other dependencies such as **numpy**
Installation via pip:
    
    ```bash
    bash
    复制
    pip install numpy
    
    ```
    

## Project Structure

```
bash
复制
Project_display/
├── main.py           # Main program entry point
├── README.md         # Project documentation
└── (other files or directories)

```

## Usage

1. **Run the Program:**
Open a terminal or command prompt, navigate to the `Project_display` directory, and execute:
    
    ```bash
    bash
    复制
    python main.py
    
    ```
    
2. **Switch Cameras:**
When the program is running, press the **C** key to switch to the next camera (if multiple cameras are connected).
3. **Exit the Program:**
Press the **Q** key or close the video window to exit.

## Configuration & Customization

- **Camera Index:**
To modify the default camera index or adjust other parameters, locate the relevant variables in the `main.py` file and change them according to your device settings.
- **Feature Extensions:**
Feel free to extend the functionality, such as adding image processing, video recording, or special effects.

## FAQ

- **Camera Does Not Start:**
    - Verify that the camera is correctly connected and enabled.
    - Ensure no other program is using the camera device.
- **Key Press Not Responding:**
    - Make sure the video window is active and focused.
    - Check if the key event bindings in the code are functioning correctly.
- **Program Errors:**
    - Confirm that your Python version meets the requirements.
    - Ensure that all dependencies are installed correctly.

## Contributing

Contributions and suggestions are welcome! If you have any questions or ideas for improvements, please open an issue on our [Issues page](https://github.com/your-repo/issues).

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.

---

This README file provides a clear overview of the project, its usage, and how to get started. Adjust any sections as needed to better fit your project’s requirements.s
