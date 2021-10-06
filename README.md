# Word_Line_Segmentataion(Lines and words Segmentation of IAM Forms dataset using OpenCV)

Offline Handwritten Text Recognition (HTR) systems transcribe text contained in scanned images into digital text.

In general, developed HTR consists of several parts, which are responsible for processing pages with full text (scanned or photographed), dividing them into lines, splitting the resulting lines into words and following recognition of words from them.

For solve the problem of recognition, it was decided to use Neural Network (NN). It consists of convolutional NN (CNN) layers, recurrent NN (RNN) layers and a final Connectionist Temporal Classification (CTC) layer.

But in this notebook only the task of page segmentation is highlighted.

# Word Segmentation
Word Segmentation is used to identify the words present in the paragraph using bounding box coordinates.

## Image Pre-Processing ##
The pre-processing is a series of operations performed of scanned input image. It essentially enhances the image rendering for suitable segmentation. The role of pre-processing is to segment the interesting pattern from the background.Grey-scaling (images are turned to black and white for the model to accurately detect the presence of handwritten text).

#### Steps in Pre-Processing: ####
* Load the paragraph image in Grayscale Mode.
```
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(10,10))
plt.imshow(gray)
```
![Screenshot from 2021-10-07 02-28-55](https://user-images.githubusercontent.com/67474853/136282225-9e6c4386-e1b9-4ccb-a956-e2c3710f92c7.png)
* Performing Thresholding operation on the grayscaled image: The basic Thresholding technique is Binary Thresholding. For every pixel, the same threshold value is applied. If the pixel value is smaller than the threshold, it is set to 0, otherwise, it is set to a maximum value.
```
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
plt.figure(figsize=(10,10))
plt.imshow(thresh, cmap='gray', vmax=1, vmin=0)
```
![Screenshot from 2021-10-07 02-42-28](https://user-images.githubusercontent.com/67474853/136284097-4a902e47-d6cf-4132-bd66-e1c0b8746621.png)
* Performing Dilation operation on the threshed image:A kernel(a matrix of odd size(25,130) is convolved with the image.A pixel element in the original image is ‘1’ if at least one pixel under the kernel is ‘1’.It increases the white region in the image or the size of the foreground object increases.
```
kernel = np.ones((25,20), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
plt.figure(figsize=(10,10))
plt.imshow(img_dilation, cmap='gray', vmax=1, vmin=0)
```
![Screenshot from 2021-10-07 02-49-03](https://user-images.githubusercontent.com/67474853/136284741-2f997fbe-8105-468e-8216-800f1b02c239.png)
* Find and draw Contours: 'Contours‘ is a Python list of all the contours in the image. Each individual contour is a Numpy array of (x, y) coordinates of boundary points of the object.findContour() function helps in extracting the contours from the image. It works best on binary images, so we should first apply thresholding techniques, Sobel edges, etc.
```
ctrs,hier = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#sort contours from left to right and top to bottom
#sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
#it only sorts by y position of the bbox. to sort by x and y you'd use:
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * img.shape[1] )

plt.figure(figsize=(15,15))
current_axis = plt.gca()

lst4 = []

for k, ctr in enumerate(sorted_ctrs):
    sub_list1 = []
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)
    sub_list1.append(y)
    sub_list1.append(x)
    sub_list1.append(w)
    sub_list1.append(h)
   
    lst4.append(sub_list1)
    current_axis.add_patch(Rectangle((x, y), w, h, edgecolor = 'g', fill=False, linewidth=2)) 
    
fin_boxes = np.array(lst4)
print(img.shape)
print(fin_boxes)
plt.imshow(img)
```
![download](https://user-images.githubusercontent.com/67474853/136285952-af5b606a-20f1-4102-aad8-1d1fa7dacb49.png)

# Future Work
* Combine Word Segmentation and Text Recognition Model by passing the bounding box coordinates in the predict function of CRNN-CTC text recognition Model.
* To recognize whole paragraph at once Scan, Attend and Read: End-to-End Handwritten Paragraph Recognition.
