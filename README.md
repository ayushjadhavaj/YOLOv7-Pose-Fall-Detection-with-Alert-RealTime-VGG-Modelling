
# YOLOv7-Pose Fall Detection with Alert (RealTime) + VGG Modelling on a small data(Research)

This project combines two main components: fall detection using the YOLOv7-POSE detection model and VGG modeling on a dataset. The goal is to detect falls in real-time and provide an alert, while also exploring the effectiveness of VGG for modeling with limited collected data.




## Acknowledgements

I would like to express our sincere gratitude to the following individuals and organizations for their invaluable support and contributions to this project:


- YOLOv7 and Darknet: I am grateful to the creators and contributors of the YOLOv7-POSE fall detection model and the Darknet framework. Their open-source implementations have served as the foundation for our fall detection system.

- VGG Network: I would like to acknowledge the authors of the VGG network, Karen Simonyan and Andrew Zisserman, for their groundbreaking work in computer vision. The VGG network architecture has been instrumental in our research on modeling with limited collected data.

- Dataset Providers: I would like to express our sincere appreciation to the following dataset providers for making their datasets available, which were used in this project.

- UTTEJ KUMAR K ANDAGATLA: I would like to thank Uttej Kumar K ANndagatla for providing the fall detection dataset used in our research. The dataset can be found at [Fall Detection Dataset](https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset). Their efforts in collecting and sharing the dataset have been instrumental in the success of our project of modelling with VGG16, CNN and Improved CNN.

- Kaggle: I also acknowledge Kaggle for hosting the fall detection dataset. Kaggle provides a valuable platform for data sharing and collaboration among the data science community.


- Open-Source Community: I would like to acknowledge the vibrant open-source community that has provided a wealth of resources, tutorials, and libraries, making it possible for us to build upon existing work and push the boundaries of our research.

- Friends and Family: Last but not least, I would like to express our gratitude to our friends and family for their unwavering support and encouragement throughout this project.

I apologize if I inadvertently missed anyone who deserves acknowledgment. I appreciate the collective effort that goes into the success of any research endeavor and recognize the importance of each contribution.

Thank you all for being part of this exciting journey!




## Installation

To install and set up the project, follow these steps:

## Prerequisites

Before installing and setting up the project, make sure you have the following prerequisites installed:

- [Python 3.9](https://www.python.org/downloads/) or higher: Make sure Python is installed on your system. You can download it from the official Python website.

- [Anaconda](https://www.anaconda.com/): We recommend using Anaconda to manage your Python environment. You can download and install Anaconda from their website.

- [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/): If you're using Windows, make sure you have Microsoft C++ Build Tools installed. This is required for certain dependencies to work properly. You can download and install it from the Microsoft website.

- [Telegram](https://telegram.org/): To utilize the Telegram functionality of the project, you need to have the Telegram app installed on your mobile device or desktop.


### Step 1: Clone the repository

```bash
git clone https://github.com/WongKinYiu/yolov7.git
cd your-project
```

### step 2: Create a virtual environment

```bash
conda create -n project-env python=3.9
conda activate project-env
```

### step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### step 4: Install pytorch
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```


## Deployment

To deploy the project and use the different scripts, follow these instructions:

### Fall Detection on Image

To perform fall detection on a single image, run the following command:

```bash
python detect.py --weights yolov7-w6-pose.pt --kpt-label --hide-labels --hide-conf --source ezio.jpg
```
### Fall Detection on Video

To perform fall detection on a video, run the following command:

```bash
python detect.py --weights yolov7-w6-pose.pt --kpt-label --hide-labels --hide-conf --source video.mp4 --line-thickness 4 --nosave --view-img
```
### Pose Estimation on Video

To perform pose estimation on a video, run the following command:

```bash
python yolov7_pose_estimationvideo.py --weights yolov7-w6-pose.pt --kpt-label --hide-labels --hide-conf --source video.mp4 --line-thickness 4 --nosave --view-img

```

### Live Pose Estimation

To perform live pose estimation using the camera, run the following command:

```bash
python yolov7_pose_estimationvideolive.py --weights yolov7-w6-pose.pt --kpt-label --hide-labels --hide-conf --source video.mp4 --line-thickness 4 --nosave --view-img
```

## Alerts

We have integrated Telegram and IFTTT to receive alerts for fall detection. Here's how it works:

1. Telegram Bot: We created a Telegram bot using BotFather and obtained the bot token. This token is used to authenticate and interact with the Telegram API.

2. IFTTT Integration: We set up an applet on IFTTT (If This Then That) to create a webhook trigger for missed calls. Whenever a fall is detected, the system triggers a webhook to IFTTT.

3. IFTTT Webhook: The webhook configured in IFTTT sends a notification to our Telegram bot using the bot token and relevant chat ID. This way, we receive an alert on our Telegram app whenever a fall is detected.

Please note that you will need to set up your own Telegram bot and configure the IFTTT applet with your webhook and Telegram bot details.


## Tech 

This project utilizes the following technologies:

- **Python:** The programming language used for the implementation of the project.

- **YOLOv7-POSE:** A state-of-the-art object detection model used for fall detection and pose estimation.

- **OpenCV:** An open-source computer vision library used for image and video processing tasks.

- **PyTorch:** A deep learning framework used for training and inference of the YOLOv7-POSE model.

- **numpy:** A fundamental package for scientific computing in Python, used for numerical operations and array manipulation.

- **matplotlib:** A plotting library used for visualizing data and generating plots and figures.

- **Pillow:** A Python imaging library used for image processing and manipulation.

- **PyYAML:** A YAML parser and emitter for Python, used for configuration management.

- **tqdm:** A library for creating progress bars and monitoring the progress of tasks.

- **onnxruntime:** An inference engine for ONNX (Open Neural Network Exchange) models, used for efficient model inference.



## Screenshots

Here are some screenshots and output links related to the project:

### YOLOv7-POSE Fall Detection Output Video

[YOLOv7-POSE Fall Detection Output Video](https://drive.google.com/file/d/1d8aA7cXXmSnGh1RZORTZQWmjKUQnNUsn/view?usp=sharing)

### Loss Outputs for CNN, Improved CNN, and VGG16 Models

[Link to Loss Outputs](https://drive.google.com/file/d/11SjdFZyAgFQg5a3wG3eAR1wnuSDOk4UI/view?usp=sharing)


### Accuracy Outputs for CNN, Improved CNN, and VGG16 Models

[Link to Accuracy Outputs](https://drive.google.com/file/d/1Jpf2fQrZcpZJKAlnxI0219g1OAKR06CK/view?usp=sharing)

## Citations

If you use this project or codebase for your research or work, please consider citing it:

- Official YOLOv7 Implementation: WongKinYiu. (2021). YOLOv7: Object detection in PyTorch. GitHub repository. [Link to the repository](https://github.com/WongKinYiu/yolov7)

The YOLOv7 repository mentions the following citations:


- Wang, C. Y., Bochkovskiy, A., & Liao, H. Y. M. (2022). YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors. arXiv preprint arXiv:2207.02696. [Link to the paper](https://arxiv.org/abs/2207.02696)

- Wang, C. Y., Liao, H. Y. M., & Yeh, I. H. (2022). Designing Network Design Strategies Through Gradient Path Analysis. arXiv preprint arXiv:2211.04800. [Link to the paper](https://arxiv.org/abs/2211.04800)

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.

- Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.



