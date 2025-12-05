# Software Requirements Specification (SRS)

## Project Title: Face Mask Detection System (FMDS)

| Attribute | Value |
| :--- | :--- |
| **Prepared By** | Laraib Qandeel (F22BINFT1E02142) |
| **Project Supervisor** | Sir Syed Ali Nawaz Shah |
| **Organization** | The Islamia University of Bahawalpur (IUB) |

***

## 1. Introduction

### 1.1 Purpose of the Document

This Software Requirements Specification (SRS) formally defines the complete functional, non-functional, and technical requirements for the **Face Mask Detection System (FMDS)**. The document serves as the authoritative blueprint for the system's development, ensuring all stakeholders—including the student, supervisor, and potential integrators—share a common understanding of the system's capabilities and constraints.

### 1.2 Scope of the Product

The FMDS is a specialized deep learning model designed for binary image classification. Its core function is to analyze a static image of a face and determine the presence or absence of a protective face mask. The system is a modular, back-end component intended for integration into broader monitoring or access control applications.

The scope of this final year project is limited to:
1.  The development and training of a robust Convolutional Neural Network (CNN) model.
2.  The implementation of a Python-based inference engine capable of processing static images.
3.  The achievement of a high classification accuracy on an independent test dataset.

### 1.3 Definitions, Acronyms, and Abbreviations

| Term/Acronym | Definition |
| :--- | :--- |
| **SRS** | Software Requirements Specification |
| **FMDS** | Face Mask Detection System |
| **CNN** | Convolutional Neural Network, the core deep learning architecture. |
| **Inference** | The process of using a trained model to make a prediction on new data. |
| **Epoch** | One complete cycle through the entire training dataset. |
| **Latency** | The time delay between inputting an image and receiving the classification output. |
| **RGB** | Red, Green, Blue; the three color channels of the input image. |

### 1.4 References

1.  **Project Implementation Notebook:** `Face_Mask_Detection_using_CNN_DeepLearning_Project(1)(1).ipynb` (Source of technical specifications and performance metrics).
2.  **Dataset Source:** Face Mask Dataset (Kaggle).

***

## 2. Overall Description

### 2.1 Product Perspective and System Interface

The FMDS is a standalone software module that acts as a classification service. It is designed to be integrated via a simple Application Programming Interface (API) or function call within a host application. It does not require a graphical user interface (GUI) for its core operation. The primary interface is the function signature that accepts image data and returns a prediction.

### 2.2 Product Functions

The system's operation is defined by the following sequential functions:

| ID | Function Name | Description |
| :--- | :--- | :--- |
| **F-100** | **Image Input Handling** | Accepts raw image data (e.g., NumPy array or file path) from the host system. |
| **F-200** | **Data Preprocessing** | Resizes the input image to the required **128x128x3** dimensions and normalizes pixel values to the **[0, 1]** range. |
| **F-300** | **Model Inference** | Executes the trained CNN model to perform the classification task. |
| **F-400** | **Result Output** | Returns a structured object containing the predicted class label and the confidence score. |

### 2.3 User Characteristics

The system is designed for two primary user groups:

| User Class | Technical Proficiency | Key Interaction |
| :--- | :--- | :--- |
| **Integrator/Developer** | High proficiency in Python, TensorFlow/Keras, and software integration. | Integrating the model into a larger application or deployment environment. |
| **System Operator/End-User** | Low technical proficiency; requires clear, immediate feedback. | Monitoring the final application's output (e.g., a visual alert or log entry). |

### 2.4 General Constraints

| Constraint Type | Description |
| :--- | :--- |
| **Technology** | Constrained to the Python ecosystem, specifically requiring **TensorFlow/Keras** for model execution. |
| **Performance** | The system must meet the specified accuracy and latency requirements to be viable for real-world deployment. |
| **Input Format** | The model is strictly trained for **128x128 RGB** images; all inputs must conform to this specification after preprocessing. |
| **Classification** | The system is limited to **binary classification** (Mask/No Mask) and does not currently support multi-class detection (e.g., improper mask wearing). |

***

## 3. Specific Requirements

### 3.1 Functional Requirements

#### 3.1.1 FR-1.1: Input Preprocessing
The system shall ensure that all input images are resized to a resolution of 128x128 pixels and converted to a 3-channel (RGB) format.

#### 3.1.2 FR-1.2: Classification Output
The system shall output a classification result that maps to one of the following two labels:
*   **Label 1:** Mask Worn (Positive Class)
*   **Label 0:** No Mask Worn (Negative Class)

#### 3.1.3 FR-1.3: Confidence Reporting
The system shall provide the prediction confidence as a floating-point value between 0.0 and 1.0, representing the model's certainty in the classification.

### 3.2 Non-Functional Requirements

#### 3.2.1 Performance Requirements

The model's performance, as demonstrated during the training phase, must be maintained in the deployed environment:

| Metric | Requirement | Achieved Value (from Notebook) |
| :--- | :--- | :--- |
| **Classification Accuracy** | Must exceed 90.0% on the test set. | **92.05%** |
| **Model Latency** | Target inference time of < 50ms per image on GPU-accelerated hardware. | TBD (Deployment Dependent) |
| **Training Stability** | The model must be trained for 5 epochs, demonstrating convergence without significant overfitting. | 5 Epochs |

#### 3.2.2 Security and Privacy
The system shall not perform any facial recognition or store any image data after the classification process is complete, ensuring compliance with basic privacy standards.

#### 3.2.3 Maintainability
The model architecture and training configuration shall be fully documented and stored in a version-controlled repository to facilitate future retraining and maintenance.

### 3.3 Technical Requirements (Model Architecture)

The FMDS utilizes a Sequential CNN model, compiled with the Adam optimizer and Sparse Categorical Crossentropy loss function.

| Layer Type | Output Shape | Activation | Purpose |
| :--- | :--- | :--- | :--- |
| **Conv2D (1)** | (126, 126, 32) | ReLU | Initial feature map generation. |
| **MaxPooling2D (1)** | (63, 63, 32) | N/A | Spatial down-sampling. |
| **Conv2D (2)** | (61, 61, 64) | ReLU | Deeper feature extraction. |
| **MaxPooling2D (2)** | (30, 30, 64) | N/A | Further spatial reduction. |
| **Flatten** | (57600) | N/A | Prepares data for fully connected layers. |
| **Dense (1)** | (128) | ReLU | High-level feature combination. |
| **Dropout (1)** | (128) | N/A | Regularization (Rate 0.5). |
| **Dense (2)** | (64) | ReLU | Secondary feature combination. |
| **Dropout (2)** | (64) | N/A | Regularization (Rate 0.5). |
| **Output Dense** | (2) | Sigmoid | Final classification output. |

***

## 4. Future Enhancements

The following enhancements are recommended for future iterations of the Face Mask Detection System to improve its utility, robustness, and real-world applicability.

### 4.1 Real-Time Video Stream Processing (FE-1)
The system should be extended to process live video feeds. This requires integrating a frame-by-frame processing pipeline using libraries such as OpenCV, which will introduce a new requirement for frame-rate performance (e.g., 15-30 FPS).

### 4.2 Face Localization and Bounding Box (FE-2)
A critical enhancement is the integration of a dedicated face detection algorithm (e.g., MTCNN, YOLO, or SSD) to:
1.  Locate all faces in a frame.
2.  Draw a bounding box around each detected face.
3.  Crop the face for classification by the CNN model.
4.  Overlay the classification result (Mask/No Mask) onto the bounding box in the output stream.

### 4.3 Improper Mask Detection (FE-3)
A future iteration should expand the model to a multi-class classification problem, including a third class: **"Improperly Worn Mask"** (e.g., mask covering only the mouth, or worn on the chin). This requires acquiring and labeling a new, expanded dataset.

### 4.4 Edge Device Optimization (FE-4)
To enable deployment on low-power devices (e.g., Raspberry Pi, mobile phones), the model should be optimized for size and speed. This involves:
1.  Model quantization (e.g., converting to 8-bit integers).
2.  Conversion to lightweight formats such as **TensorFlow Lite** or **ONNX**.
3.  Evaluation of model performance on embedded systems.

***

## Appendix A: Project Personnel

| Role | Name | Identifier |
| :--- | :--- | :--- |
| **Student/Author** | Laraib Qandeel | F22BINFT1E02142 |
| **Project Supervisor** | Sir Syed Ali Nawaz Shah | N/A |
| **Organization** | The Islamia University of Bahawalpur | N/A |
