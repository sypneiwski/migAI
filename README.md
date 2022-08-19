# MigAI - a Polish Sign Language Recognition project
### Demos
https://user-images.githubusercontent.com/67709201/185696473-a26fa3de-df56-4e78-890e-c1eee58d0841.mp4



https://user-images.githubusercontent.com/67709201/185696568-a71c9db5-172a-4e51-9bb2-2a7e4f5af723.mp4

### Abstract
_migAI - "Let's sign" in Polish_

A web application capable of recognizing 20 static signs from the Polish Sign Language, developed by three students using convolutional neural networks.

### Background & motivation
During the Covid pandemic many new technologies were developed to facilitate better work-from-home environments, among them those that translate text to speech and transcribe subtitles for different languages during video calls. There are, however, no such tools of prominence for sign languages and, in particular, the Polish Sign Language.

This project was developed as part of the Computer Science BA's Software Engineering course at the University of Warsaw, by a team of three students. The goal was to create a working web application that could recognize a subset of 20 static signs from the Polish Sign Language from images of hand gestures captured by webcam, and output the results as a stream of text. The application consisted of two components: the frontend, designed by Jan Wojtach, and a machine learning model in the backend, developed by Mateusz Sypniewski and Maria Smigielska.

### Training data acquisition
The 20,000 hand gesture images (1,000 per sign) used to train the model were collected over the course of three days by two of the students in varying lighting conditions. A webcam was used to capture the images at a rate of ~60 frames a second.

### Data pre-processing
The images were first cropped to a 200px x 200px region of interest and converted to the HSV color space. Histogram backprojection[^1] was used to find the skin colored pixels, the histogram being an average of hand images taken in different lighting conditions. Gaussian and median blurs were applied to the result[^2], and finally the image was thresholded using Otsu's method, producing results like these: ![PSL sign for 'Y'](https://user-images.githubusercontent.com/67709201/185700244-adbc4f3e-38f8-4a32-9f34-996c9c9f42f2.png)

### Training
The collected data was randomly divided into training, test and validation sets (in the ratio of 8 : 1 : 1). A convolutional neural network[^3] architecture was used, with three convolutional layers, each followed by a pooling layer, and finally two densely connected linear layers. The network employed cross-entropy as the loss function and the stochastic gradient descent optimizer.

### Outcome
The model achieved very high (> 98 %) accuracy on the training data. It also performed very well in most manual tests, with ~90% accuracy. A video demo of the app recognizing the phrase "To be or not to be", and another showing all of the recognized signs in a sequence, can be found above.

### Limitations
The trained model turned out to be highly dependent on lighting conditions, and performed best in conditions similar to those in which the histogram images were taken. With particularly adverse lighting the accuracy of the model was no better than random. Skin tone and distinct hand features unique to the two students who collected the training data were also key factors, with the optimal accuracy being achieved in tests done by those students.

These limitations could be addressed by collecting data from more diverse subjects proficient in sign language, as well as perfecting the method for skin segmentation.

### Acknowledgements
The project was developed under the guidance of Grzegorz Grudzinski, a lecturer at the University of Warsaw, and employed the use of OpenCV and PyTorch libraries. The original documentation of the project (in Polish) can be found [here](https://drive.google.com/drive/folders/1DmidYspVy9fUVdAPWSyX-HHTBMgZOeHC).

[^1]:Swain MJ, Ballard DH (1992) Indexing via color histograms. Active Perception and Robot Vision: Springer.
[^2]:H. Gupta, S. Oza, A. Sharma, M. Shukla (2020) Sign Language Interpreter using Deep Learning, [GitHub](https://github.com/harshbg/Sign-Language-Interpreter-using-Deep-Learning)
[^3]:Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner (1998) Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324

