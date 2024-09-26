
# Face Mask Detection Project

<p align="center">
  <img src="img/face_mask.png" alt="Example Image" height="200" width="400" style="vertical-align:middle;"/>
</p>

## Project Overview 🎯

This project aims to detect whether a person is wearing a face mask or not using a combination of **MTCNN** for face detection and **MobileNet** for mask classification. The model is trained using TensorFlow and is capable of identifying masked and unmasked faces in images.

## Dataset 📊

- The dataset consists of images categorized into two classes:
  - **With Mask**
  - **Without Mask**

## Model Architecture 🧠

- **Face Detection**: MTCNN (Multi-Task Cascaded Convolutional Networks) is used to detect faces in images.
- **Mask Detection**: MobileNet, a lightweight deep neural network architecture, is used as the base model for classifying whether a detected face is masked or unmasked.

## Model Training ⚙️

- The model is trained using data augmentation techniques and the `ImageDataGenerator` class from Keras.
- Loss and accuracy curves are plotted to visualize the model's performance over time.

## Model Performance 📈

The model's accuracy and loss during training are recorded and can be visualized using matplotlib:

<p align="center">
  <img src="/img/Accuracy.png" alt="Real Image" width="400" />
  <img src="/img/Loss.png" alt="AI generated Image" width="400" />
</p>

### Usefulness of Project 🚀

1. **Public Spaces Monitoring**:
   - Can be deployed in public areas such as airports, train stations, and shopping malls to ensure compliance with mask-wearing policies.

2. **Workplace Safety**:
   - Used in corporate offices or industrial plants to monitor employees' compliance with health and safety protocols.

3. **Healthcare Facilities**:
   - Can assist in hospitals and clinics by ensuring that staff and visitors are wearing masks to minimize the spread of infectious diseases.

4. **Educational Institutions**:
   - Helps schools and universities monitor whether students and staff are following mask mandates, ensuring a safer learning environment.

5. **Smart Surveillance**:
   - Can be integrated with CCTV systems for automated monitoring and alerting in real-time when someone is not wearing a mask in restricted zones.

6. **Event Management**:
   - Useful for managing large gatherings such as concerts, conferences, or sports events to ensure safety measures are being followed.

  
## Project Structure 📁

```bash
face-mask-detection/
│
├── data/                    # Directory containing the dataset
│   ├── train/               # Training images
│   └── test/                # Testing images
│
├── models/                  # Trained models and weights
│   └── face_mask_detector.h5
│
├── main.ipynb               # Jupyter notebook containing the project code
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Requirements 📦

To install the required libraries, you can use the following command:

```bash
pip install -r requirements.txt
```

### Main Libraries Used:

- TensorFlow
- Keras
- OpenCV
- MTCNN
- MobileNet

## How to Use 🚀

1. **Run the Jupyter Notebook**: Open the `main.ipynb` file and execute the cells in order to train the model or make predictions on new images.
2. **Face Mask Detection**:
   - The model will detect faces in an image.
   - Each detected face will be classified as **With Mask** or **Without Mask**.


## Acknowledgements 🙏

- Thanks to the creators of the **MTCNN** and **MobileNet** architectures.
- Special thanks to the **TensorFlow** and **Keras** teams for making deep learning accessible to all.

## Contributing 🤝

Feel free to fork this repository, make your changes, and submit a pull request. For significant changes, please open an issue to discuss them first.

