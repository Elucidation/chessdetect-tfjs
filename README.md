## U-Net++ Chessboard Segmentation and 4 Corners Web Predictor

This is an HTML/JavaScript web application that runs a pre-trained U-Net++ segmentation/heatmap model directly in the browser using TensorFlow.js. The model runs locally on device, no server is used.

![latest_224_model_heatmaps_result](https://github.com/user-attachments/assets/1a0deba5-a425-4eed-b715-cd5904497096)

It provides an interactive way to test and visualize the output of the chessboard segmentation + 4 corner heatmap U-Net++ model on different inputs from a website.

[**Try it out Live**](https://elucidation.github.io/chessdetect-tfjs/) or watch a video of it in use below:


https://github.com/user-attachments/assets/f32c9319-21a1-4f7c-b17a-33c4baf608e5

**Features:**

* Loads a quantized uint8 U-Net++ model previously converted to the TensorFlow.js graph model format (`model.json` + weights).
* The model was trained on 10,000 custom synthetic images that [I made with Blender](https://youtu.be/ybKiTbZaJAw?si=b5dMPWt5Md34XuKk)
* Accepts input via either:
    * **File Upload:** Select an image file from your device.
    * **Live Camera Feed:** Use your device's camera in real-time.
* Performs model prediction directly in the browser.
* Supports switching between available cameras (e.g., front/back on mobile).
* Displays the input (uploaded image or live camera feed) alongside a combined visualization showing the predicted segmentation mask and corner heatmaps overlaid on the input.
* Includes a slider to control the transparency (alpha) of the segmentation/heatmap overlay.


**Model**

It takes in 128x128 RGB images and puts out 128x128x5 binary segmentation + 4 gaussian heatmaps for the 4 corners of the chessboard

I have a small Youtube series explaining how this model works: [Finding Chessboards with U-Net ML Models - Decoding Chessboards | Part 4](https://youtu.be/BVt12vzp_iM?si=SfXGMvcKfdSI8SZ3)

* INPUT 128x128x3
* A series of double convolutions and max pool steps in the encode phase, then the reverse back up the decode phase, with filter sizes [16, 32, 64, 128, 256]
* OUTPUT 128x128x5

```
Model: "Unetpp_16_32_64_128_256"

==================================================================================================
Total params: 555509 (2.12 MB)
Trainable params: 553845 (2.11 MB)
Non-trainable params: 1664 (6.50 KB)
__________________________________________________________________________________________________
```

The model itself is a U-Net++ architecture (U-Net with skip/merge connections from all the layers back up) based off of https://arxiv.org/abs/1807.10165 
