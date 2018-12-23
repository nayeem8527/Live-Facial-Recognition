# Live Facial Recognition

Live Facial Recogition of person using web-cam. Using the connected webcam image to passed to our framework which recognizes the person on the frame. The framework can detect also multiple person in a single frame. Detectio can go with the speed of 17 FPS for single face and gradually goes down when faces increases in the frames.

# Model Architecture

 1. Every i’th (4th) frame in the video is passed to the MTCNN module which detects the faces in the frame and gives bonding box for each face.
2. The aligned image is then passed to facenet model to get the embedding of the face.
3. The embedding is the passed to a classifier to get the final result.

# Face Alignment Using MTCNN

Multi-task Cascaded Convolutional Networks is a three stage cascade deep neural network that predicts face and its location (bounding box).  This framework exploits the inherent correlation between detection and alignment.

**Stage 1 (Proposal Network -PNET)**: A Fully convolutional network, called Proposal Network (P-Net) is used to obtain the candidate facial windows and their bounding box regression vectors. Facial candidates are further calibrated using bounding boxes followed by a non maximal suppression.

**Stage 2: (Refine Network-RNET)**: All candidates are fed to another CNN, called Refine Network (R-Net), which further rejects a large number of false candidates, performs calibration with bounding box regression followed by non maximal suppression

**Stage 3: (Output Network- ONET)**: Similar to the second stage, but in this stage face regions are identified with more supervision. In particular, the network will output five facial landmarks positions.

# Facial Feature Extraction Using Facenet

FaceNet, learns a mapping from face images to a compact 128-D Euclidean space Once this space has been produced, tasks such as face recognition, verification and clustering can be easily implemented using standard techniques with FaceNet embeddings as feature vectors. 
We are using pre-trained model of Inception Resnet v1 trained on CASIA-Webface dataset which gives an accuracy of 98.7% 

# Classifier

The extracted features from facenet are trained on RF and SVM classifier for our created dataset.

# Preparing Dataset

For preparing your own dataset make separate folders for each person you want your model to get trained on. For each folder add nearly 15-25 images by moving the person head in different positions keeping the camera in front of it.

# Running the Code

    python face_recog_live.py

**For Training the Classifier with the dataset**  
  
    python train_classifier.py --input-dir 'INPUT IMAGE DIRECTORY' --model-path models/20170511-185253.pb --classifier-path 'PATH TO SAVE THE TRAINED MODEL' --num-threads 10 --num-epochs 5 --min-num-images-per-class 1 

# References

1. Schroff, Florian, Dmitry Kalenichenko, and James Philbin. "Facenet: A unified embedding for face recognition and clustering." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.
2. Zhang, Kaipeng, et al. "Joint face detection and alignment using multitask cascaded convolutional networks." IEEE Signal Processing Letters 23.10 (2016): 1499-1503.
