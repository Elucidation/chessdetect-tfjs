// --- DOM Elements ---
const imageInput = document.getElementById("imageInput");
const predictButton = document.getElementById("predictButton");
const inputCanvas = document.getElementById("inputCanvas");
const outputCombinedCanvas = document.getElementById("outputCombined");
const messageArea = document.getElementById("messageArea");
const modelStatus = document.getElementById("modelStatus");
const alphaSlider = document.getElementById("alphaSlider");
const alphaValueSpan = document.getElementById("alphaValue");
const modeFileButton = document.getElementById("modeFileButton");
const modeCameraButton = document.getElementById("modeCameraButton");
const fileInputSection = document.getElementById("fileInputSection");
const cameraInputSection = document.getElementById("cameraInputSection");
const startCameraButton = document.getElementById("startCameraButton");
const stopCameraButton = document.getElementById("stopCameraButton");
const switchCameraButton = document.getElementById("switchCameraButton");
const videoElement = document.getElementById("videoFeed");
const camErrorMsg = document.getElementById("camErrorMsg");
const inputCanvasLabel = document.getElementById("inputCanvasLabel");
const togglePeaksButton = document.getElementById("togglePeaks");
const fpsDisplay = document.getElementById("fpsDisplay");

// Get 2D contexts
const outCombCtx = outputCombinedCanvas.getContext("2d"); // Still needed for drawing base image & overlays

// --- Config ---
// v1.11 Best, Unet++ with Attention Gates, 128x128 size, ~8.6 MB
const MODEL_URL = "./m128_att_v3_q8/model.json";
const TARGET_IMG_SIZE = 128; // Model's expected input size

const TARGET_FPS = 45; // Predict FPS, not draw FPS
const MS_PER_FRAME = 1000 / TARGET_FPS;

// --- State ---
let loadedImage = null;
let model = null;
let lastPredictionTensor = null; // Store the last tensor for alpha adjustments
let isPredicting = false;
let isCameraMode = false;
let isCameraRunning = false;
let drawPeakOutlines = true;
let videoStream = null;
let animationFrameId = null;
let lastPredictTime = 0;
let lastPredictOnlyTime = 0; // Only time for last predict
let predictFrameCount = 0; // Track Predict update count for predict FPS
let lastFpsUpdate = 0;
let videoDevices = [];
let currentDeviceId = null;
let devicesEnumerated = false;

// --- Temporary Canvases (Still needed as targets for tf.browser.toPixels) ---
let tempCombinedCanvas = null;

// --- Status Update Function ---
function setStatus(text, showSpinner = false) {
  modelStatus.innerHTML =
    text + (showSpinner ? ' <div class="loader"></div>' : "");
}
function setMessage(text = "") {
  messageArea.textContent = text;
}
function setCamError(text = "") {
  camErrorMsg.textContent = text;
}

// --- Mode Switching ---
modeFileButton.addEventListener("click", () => switchMode("file"));
modeCameraButton.addEventListener("click", () => switchMode("camera"));
function switchMode(mode) {
  if (mode === "file") {
    isCameraMode = false;
    stopCamera(); // Stop camera if switching away
    fileInputSection.classList.remove("hidden");
    cameraInputSection.classList.add("hidden");
    modeFileButton.classList.replace("bg-gray-300", "bg-purple-600");
    modeFileButton.classList.replace("text-gray-700", "text-white");
    modeFileButton.classList.add("ring-2", "ring-purple-600", "z-10");
    modeCameraButton.classList.replace("bg-purple-600", "bg-gray-300");
    modeCameraButton.classList.replace("text-white", "text-gray-700");
    modeCameraButton.classList.remove("ring-2", "ring-purple-600", "z-10");
    resetUI(); // Reset UI elements for file mode
  } else {
    // camera mode
    isCameraMode = true;
    fileInputSection.classList.add("hidden");
    cameraInputSection.classList.remove("hidden");
    modeCameraButton.classList.replace("bg-gray-300", "bg-purple-600");
    modeCameraButton.classList.replace("text-gray-700", "text-white");
    modeCameraButton.classList.add("ring-2", "ring-purple-600", "z-10");
    modeFileButton.classList.replace("bg-purple-600", "bg-gray-300");
    modeFileButton.classList.replace("text-white", "text-gray-700");
    modeFileButton.classList.remove("ring-2", "ring-purple-600", "z-10");
    resetUI(); // Reset UI elements for camera mode
  }
}

// --- Camera Controls ---
startCameraButton.addEventListener("click", () => startCamera());
stopCameraButton.addEventListener("click", stopCamera);
switchCameraButton.addEventListener("click", switchCamera);
async function getCameraDevices() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    console.warn("enumerateDevices() is not supported.");
    return;
  }
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    videoDevices = devices.filter((device) => device.kind === "videoinput");
    console.log("Available video devices:", videoDevices);
    devicesEnumerated = true;
    switchCameraButton.disabled = !(videoDevices.length > 1 && isCameraRunning);
  } catch (err) {
    console.error("Error enumerating devices:", err);
    setCamError(`Error listing cameras: ${err.message}`);
  }
}
async function startCamera(deviceId = null) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setCamError("getUserMedia() not supported (HTTPS required on mobile).");
    return;
  }
  if (!model) {
    setCamError("Model not loaded yet.");
    return;
  }
  if (isCameraRunning) return;
  setCamError("");
  setMessage("Starting camera...");
  startCameraButton.disabled = true;
  stopCameraButton.disabled = true;
  switchCameraButton.disabled = true;
  let constraints = { video: {} };
  if (deviceId) {
    constraints.video.deviceId = { exact: deviceId };
    console.log("Attempting camera with deviceId:", deviceId);
  } else {
    constraints.video.facingMode = "environment";
    console.log("Attempting camera with facingMode: environment");
  }
  try {
    let stream = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (err) {
      console.warn(
        `Failed getting camera with constraints: ${JSON.stringify(
          constraints
        )}. Error: ${err.name}`
      );
      if (deviceId || constraints.video.facingMode === "environment") {
        console.log("Falling back to facingMode: user");
        constraints.video = { facingMode: "user" };
        try {
          stream = await navigator.mediaDevices.getUserMedia(constraints);
        } catch (err2) {
          console.warn(
            `Failed getting user camera. Falling back to any video.`
          );
          constraints.video = true;
          stream = await navigator.mediaDevices.getUserMedia(constraints);
        }
      } else {
        throw err;
      }
    }
    videoStream = stream;
    videoElement.srcObject = videoStream;
    const currentTrack = videoStream.getVideoTracks()[0];
    if (currentTrack) {
      currentDeviceId = currentTrack.getSettings().deviceId;
      console.log("Using deviceId:", currentDeviceId);
    }
    videoElement.onloadedmetadata = () => {
      videoElement
        .play()
        .then(async () => {
          console.log("Video playback started.");
          isCameraRunning = true;
          stopCameraButton.disabled = false;
          if (!devicesEnumerated) {
            await getCameraDevices();
          } else {
            switchCameraButton.disabled = !(
              videoDevices.length > 1 && isCameraRunning
            );
          } // Update button state
          setMessage("Camera running. Starting predictions...");
          lastPredictTime = performance.now();
          predictFrameCount = 0;
          lastFpsUpdate = lastPredictTime;
          predictLoop(); // Start the prediction loop
        })
        .catch((err) => {
          console.error("Video play() failed:", err);
          setCamError(`Autoplay failed: ${err.message}.`);
          stopCameraButton.disabled = false;
        });
    };
    videoElement.onerror = (e) => {
      console.error("Video Element Error:", e);
      setCamError("Error playing video stream.");
      stopCamera();
    };
  } catch (err) {
    console.error("Error accessing camera:", err);
    setCamError(`Error accessing camera: ${err.name}`);
    setMessage("");
    startCameraButton.disabled = !model;
    stopCameraButton.disabled = true;
    switchCameraButton.disabled = true;
  }
}
function stopCamera() {
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
  if (videoStream) {
    videoStream.getTracks().forEach((track) => track.stop());
  }
  videoElement.srcObject = null;
  isCameraRunning = false;
  currentDeviceId = null;
  startCameraButton.disabled = !model;
  stopCameraButton.disabled = true;
  switchCameraButton.disabled = true;
  setMessage("Camera stopped.");
  clearOutputCanvases();
  fpsDisplay.textContent = "-";
}
async function switchCamera() {
  if (!isCameraRunning || videoDevices.length <= 1) {
    return;
  }
  console.log("Attempting to switch camera...");
  const currentIndex = videoDevices.findIndex(
    (device) => device.deviceId === currentDeviceId
  );
  const nextIndex = (currentIndex + 1) % videoDevices.length;
  const nextDeviceId = videoDevices[nextIndex].deviceId;
  console.log(`Switching from ${currentDeviceId} to ${nextDeviceId}`);
  stopCamera(); // Stop current stream first
  await new Promise((resolve) => setTimeout(resolve, 100)); // Short delay might help hardware release
  await startCamera(nextDeviceId); // Start with the new device ID
}

// --- Model Loading ---
async function loadAppModel() {
  setStatus("Loading model...", true);
  imageInput.disabled = true;
  predictButton.disabled = true;
  startCameraButton.disabled = true;
  switchCameraButton.disabled = true;
  try {
    await tf.setBackend("webgpu");
    model = await tf.loadGraphModel(MODEL_URL);
    setStatus("Warming up...", true);
    // Warmup with the correct input size
    const warmupTensor = tf.zeros([1, TARGET_IMG_SIZE, TARGET_IMG_SIZE, 3]);
    const warmupResult = model.predict(warmupTensor);
    // Dispose warmup results (handle potential array output)
    if (Array.isArray(warmupResult)) warmupResult.forEach((t) => t.dispose());
    else warmupResult.dispose();
    warmupTensor.dispose();
    setStatus(`Ready ${MODEL_URL.split('/')[1]}`);
    console.log(`Graph Model ${MODEL_URL} loaded.`);
    // Enable controls now that model is ready
    imageInput.disabled = false;
    startCameraButton.disabled = false;
    // Predict button is enabled only when an image is loaded in file mode
  } catch (error) {
    console.error("Failed to load graph model:", error);
    setStatus(`Model Load Error!`);
    setMessage("Model loading failed. Check console/path.");
  }
}

// --- Image Loading & Display (File Mode) ---
imageInput.addEventListener("change", async (event) => {
  if (isCameraMode) return; // Ignore if in camera mode
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = async () => {
        loadedImage = img;
        clearOutputCanvases(); // Clear previous predictions
        if (lastPredictionTensor) {
          // Dispose old tensor if a new image is loaded
          tf.dispose(lastPredictionTensor);
          lastPredictionTensor = null;
        }
        if (model && !isPredicting) {
          setMessage("Image loaded. Auto-predicting...");
          await runPrediction(); // Automatically predict on new image load
        } else if (model) {
          // Enable predict button only if model loaded but not auto-predicting (shouldn't happen often)
          predictButton.disabled = false;
          setMessage("Image loaded. Ready to predict.");
        } else {
          setMessage("Image loaded. Model not ready.");
        }
      };
      img.onerror = () => {
        displayError("Error loading image.");
      };
      img.src = e.target.result;
    };
    reader.onerror = () => {
      displayError("Error reading file.");
    };
    reader.readAsDataURL(file);
  } else {
    // If no file selected (e.g., user cancelled), reset relevant parts
    resetUI();
  }
});

// --- Prediction Trigger (Manual Button - File Mode) ---
predictButton.addEventListener("click", async () => {
  if (isCameraMode || !loadedImage || !model) {
    displayError("Switch to File mode and load image first.");
    return;
  }
  if (isPredicting) return; // Don't predict if already predicting
  await runPrediction();
});

// --- Core Prediction Logic (Refactored) ---
async function runPrediction(sourceElement = null) {
  const currentSource = sourceElement || loadedImage;
  if (!currentSource || !model || isPredicting) {
    return; // Exit if no source, model, or already predicting
  }
  isPredicting = true; // Set prediction flag

  // Update UI for file mode prediction start
  if (!isCameraMode) {
    setStatus("Processing...", true);
    predictButton.disabled = true;
    imageInput.disabled = true; // Disable while predicting
    setMessage("Preprocessing & Predicting...");
    await new Promise((resolve) => setTimeout(resolve, 10)); // Allow UI to update
  }

  let inputTensor = null;
  let outputTensor = null; // Keep the raw tensor output from the model

  try {
    // Preprocess the source image/video frame into a tensor
    inputTensor = preprocessSource(currentSource, TARGET_IMG_SIZE);

    const tStart = performance.now();

    // Run model prediction
    const predictionResult = tf.tidy(() => model.predict(inputTensor));

    // Get output tensor or make it if in pieces
    // predictionResult expected to be either = [HxWx5] or [HxWx1, HxWx4] for segmentation/corners
    if (!Array.isArray(predictionResult)) {
      outputTensor = predictionResult;
    }
    else if (predictionResult.length == 1) {
      outputTensor = predictionResult[0];
    }
    else {
      outputTensor = tf.concat([predictionResult[0], predictionResult[1]], axis=3);
    }

    // Squeeze the batch dimension if present (e.g., shape [1, H, W, C] -> [H, W, C])
    if (outputTensor.shape.length === 4 && outputTensor.shape[0] === 1) {
      const squeezedTensor = outputTensor.squeeze([0]);
      outputTensor.dispose(); // Dispose the original tensor with batch dim
      outputTensor = squeezedTensor; // Use the squeezed tensor
    }
    // outputTensor should now have shape [H, W, C] (e.g., [128, 128, 5])

    const predictTime = performance.now() - tStart;
    lastPredictOnlyTime = predictTime;

    // Dispose the previously stored tensor before keeping the new one
    if (lastPredictionTensor) {
      tf.dispose(lastPredictionTensor);
      lastPredictionTensor = null;
    }
    // Keep a cloned tensor so it's not disposed by tf.tidy to be used by alpha slider
    lastPredictionTensor = tf.keep(outputTensor.clone());

    // Draw the combined output using the tensor
    await drawCombined(
      currentSource,
      outputTensor,
      parseFloat(alphaSlider.value),
      outputCombinedCanvas
    );

    if (!isCameraMode) {
      setMessage(`Prediction took ${predictTime.toFixed(0)}ms`);
    }
  } catch (error) {
    console.error("Prediction Error:", error);
    if (!isCameraMode) displayError(`Processing error: ${error.message}`);
    // Ensure tensor is disposed on error
    if (lastPredictionTensor) {
      tf.dispose(lastPredictionTensor);
      lastPredictionTensor = null;
    }
  } finally {
    isPredicting = false; // Clear prediction flag
    // Update UI for file mode prediction end
    if (!isCameraMode) {
      setStatus("Model Ready");
      // Re-enable controls only if model and image are loaded
      predictButton.disabled = !model || !loadedImage;
      imageInput.disabled = !model;
    }
    // Dispose tensors used specifically in this run (input and the local outputTensor ref)
    // lastPredictionTensor is intentionally kept
    if (inputTensor) inputTensor.dispose();
    if (outputTensor) outputTensor.dispose();
  }
}

// --- Real-time Prediction Loop (Camera Mode) ---
async function predictLoop() {
  if (!isCameraRunning || !model) {
    // Stop loop if camera off or model gone
    setMessage(model ? "Camera stopped." : "Camera stopped, model unloaded.");
    fpsDisplay.textContent = "-";
    return;
  }

  const now = performance.now();
  const elapsed = now - lastPredictTime;

  // Update FPS counter approx every second
  
  if (now - lastFpsUpdate > 1000) {
    const timeTakenMs = now - lastFpsUpdate; // Time in ms.
    const perStepAvgMs = timeTakenMs / predictFrameCount;
    const fps = predictFrameCount / (timeTakenMs / 1000);
    // fpsDisplay.textContent = `${fps.toFixed(1)} | ${perStepAvgMs.toFixed(1)}ms/step | ${lastPredictOnlyTime.toFixed(1)}ms/predict`;
    fpsDisplay.textContent = fps.toFixed(1);
    predictFrameCount = 0;
    lastFpsUpdate = now;
  }

  // Throttle predictions based on TARGET_FPS
  if (elapsed > MS_PER_FRAME && !isPredicting) {
    predictFrameCount++;
    lastPredictTime = now; // Update time *before* prediction starts
    // Ensure video frame is ready before trying to predict
    if (videoElement.readyState >= videoElement.HAVE_CURRENT_DATA) {
      setMessage(" "); // Clear previous messages in camera mode
      await runPrediction(videoElement); // Run prediction on the current video frame
    }
  }

  // Request the next frame
  animationFrameId = requestAnimationFrame(predictLoop);
}

// --- Unified Preprocessing Function ---
function preprocessSource(sourceElement, targetSize) {
  return tf.tidy(() => {
    const srcWidth = sourceElement.videoWidth || sourceElement.width;
    const srcHeight = sourceElement.videoHeight || sourceElement.height;

    if (!srcWidth || !srcHeight) {
      throw new Error("Source element has invalid dimensions.");
    }

    // Create tensor from pixels
    let tensor = tf.browser.fromPixels(sourceElement).toFloat();

    // Calculate cropping to maintain aspect ratio for square target
    const aspectRatio = srcWidth / srcHeight;
    const targetAspectRatio = 1.0; // targetSize / targetSize
    let cropWidth = srcWidth;
    let cropHeight = srcHeight;
    if (aspectRatio > targetAspectRatio) {
      // Wider than target
      cropWidth = srcHeight * targetAspectRatio;
    } else {
      // Taller than target
      cropHeight = srcWidth / targetAspectRatio;
    }
    const offsetX = (srcWidth - cropWidth) / 2;
    const offsetY = (srcHeight - cropHeight) / 2;

    // Define the crop region relative to original dimensions [y1, x1, y2, x2]
    const cropBox = [
      [
        offsetY / srcHeight,
        offsetX / srcWidth,
        (offsetY + cropHeight) / srcHeight,
        (offsetX + cropWidth) / srcWidth,
      ],
    ];

    // Crop and resize using tf.image.cropAndResize
    // Input needs batch dimension, so expand dims
    const cropped = tf.image.cropAndResize(
      tensor.expandDims(0), // Add batch dim: [1, H, W, C]
      cropBox, // Boxes normalized coordinates
      [0], // Box indices (only one box)
      [targetSize, targetSize] // Target output size [height, width]
    ); // Output shape: [1, targetSize, targetSize, 3]

    // Normalize to 0-1 range
    const normalized = cropped.div(tf.scalar(255.0));

    // Dispose the initial tensor from pixels
    tensor.dispose();

    // Return the preprocessed tensor (still has batch dimension)
    // Shape: [1, targetSize, targetSize, 3]
    return normalized;
  });
}

// --- Visualization Functions ---
/**
 * Calculates peak locations from heatmap channels and draws markers onto an image tensor.
 *
 * @param {tf.Tensor} imageTensor - The base image tensor (Int32, Shape: [H, W, 3]) to draw on.
 * @param {tf.Tensor} heatmapChannelsReshaped - Heatmap prediction tensor reshaped (Float32, Shape: [H*W, 4]).
 * @param {tf.Tensor} peakMarkerColor - Color tensor for the markers (Int32, Shape: [1, 3]).
 * @param {number} predHeight - The height of the prediction tensor.
 * @param {number} predWidth - The width of the prediction tensor.
 * @returns {tf.Tensor} A new tensor with peak markers drawn, same shape and type as imageTensor.
 *                     Remember to dispose the input imageTensor if it's no longer needed after calling this.
 */
function applyPeakMarkers(
  imageTensor,
  heatmapChannelsReshaped,
  peakMarkerColor,
  predHeight,
  predWidth
) {
  return tf.tidy(() => {
    // Tidy up intermediate tensors created within this function
    // Get heatmap peaks - Find the argmax for each channel (axis=0)
    const flatIndices = heatmapChannelsReshaped.argMax(0); // Shape [4]

    // Convert flat indices to [y, x] coordinates on GPU
    const yCoords = tf.floor(flatIndices.div(predWidth)).cast("int32"); // Shape [4]
    const xCoords = flatIndices.mod(predWidth).cast("int32"); // Shape [4]
    const xyCoords = tf.stack([yCoords, xCoords], 1); // Shape [4, 2] ~[[y,x],...] peaks

    // Define offsets for the sprite shape (5x5 hollow square)
    // prettier-ignore
    const spriteOffsets = tf.tensor2d(
        [[-2, -2],[-2, -1],[-2, 0],[-2, 1],[-2, 2],[-1, -2],[-1, 2],[0, -2],
         [0, 2],[1, -2],[1, 2],[2, -2],[2, -1],[2, 0],[2, 1],[2, 2]], [16, 2], "int32");

    // Calculate all coordinates for the sprites using broadcasting
    // xyCoords [4, 1, 2] + spriteOffsets [1, 16, 2] -> allCoords [4, 16, 2]
    const allCoords = xyCoords.expandDims(1).add(spriteOffsets.expandDims(0));
    const allCoordsFlat = allCoords.reshape([-1, 2]); // Shape [4 * 16, 2] = [64, 2]

    // Clip coordinates to stay within bounds [0, H-1] and [0, W-1]
    const clippedCoords = tf.tidy(() => {
      const yClamped = allCoordsFlat
        .slice([0, 0], [-1, 1])
        .clipByValue(0, predHeight - 1);
      const xClamped = allCoordsFlat
        .slice([0, 1], [-1, 1])
        .clipByValue(0, predWidth - 1);
      return tf.concat([yClamped, xClamped], 1); // Shape [64, 2]
    });

    // Tile the color for each point in the sprites
    const numPoints = allCoordsFlat.shape[0]; // 64 points total (4 peaks * 16 points/sprite)
    const peakUpdates = tf.tile(peakMarkerColor, [numPoints, 1]); // Shape [64, 3]

    // Splat markers onto the image tensor and keep the result
    return tf.tensorScatterUpdate(imageTensor, clippedCoords, peakUpdates);
  });
}

async function drawCombined(
  baseSourceElement,
  predictionTensor,
  alpha,
  targetCanvas
) {
  if (!baseSourceElement || !predictionTensor) return;

  const targetCtx = targetCanvas.getContext("2d");
  const targetWidth = targetCanvas.width;
  const targetHeight = targetCanvas.height;

  const [predHeight, predWidth, channels] = predictionTensor.shape;
  if (channels !== 5) {
    console.error(`Prediction tensor has ${channels} channels, expected 5.`);
    return;
  }

  // --- Use only ONE temporary canvas ---
  if (
    !tempCombinedCanvas ||
    tempCombinedCanvas.width !== predWidth ||
    tempCombinedCanvas.height !== predHeight
  ) {
    tempCombinedCanvas = document.createElement("canvas");
    tempCombinedCanvas.width = predWidth;
    tempCombinedCanvas.height = predHeight;
  }

  const heatmapColorsTensor = tf.tensor2d(
    [
      [255, 0, 0],
      [0, 255, 0],
      [0, 0, 255],
      [255, 255, 0],
    ],
    [4, 3],
    "float32"
  );
  const peakMarkerColor = tf.tensor2d([[255, 0, 255]], [1, 3], "int32"); // Magenta

  // --- Generate Combined Visualization Tensor ---
  // NOTE : Most of this is running on GPU via webgpu, avoid gpu<-> data transfer where possible.
  // This is the difference between 0.2ms and 20-30ms frame updates/
  const finalDrawTensor = tf.tidy(() => {
    // Calculate desired display alphas
    const segDisplayAlpha = Math.min(1.0, alpha + 0.2);

    // --- Segmentation Mask (Channel 0) ---
    const segChannel = tf.slice(
      predictionTensor,
      [0, 0, 0],
      [predHeight, predWidth, 1]
    );
    const segChannelScaled = segChannel.mul(175); // Segmentation as not fully white gray
    const segRgb = tf.concat(
      [segChannelScaled, segChannelScaled, segChannelScaled],
      2
    ); // Shape [H, W, 3]

    // --- Heatmap (Channels 1-4) ---
    const heatmapChannels = tf.slice(
      predictionTensor,
      [0, 0, 1],
      [predHeight, predWidth, 4]
    ); // Shape: [H, W, 4]
    const heatmapChannelsReshaped = heatmapChannels.reshape([
      predHeight * predWidth,
      4,
    ]); // Shape: [H*W, 4]
    const heatmapCombined = tf.matMul(
      heatmapChannelsReshaped,
      heatmapColorsTensor
    );
    const heatmapRgb = heatmapCombined.reshape([predHeight, predWidth, 3]); // Shape [H, W, 3]

    // --- Blend Segmentation and Heatmap using Tensor Ops ---
    // Multiply each by their intended display alpha
    const segWeighted = segRgb.mul(segDisplayAlpha);
    const heatWeighted = heatmapRgb.mul(alpha);

    // Add them together (simple additive blending)
    // NOTE: This is different from overlay compositing!
    // Adjust blending logic here if needed (e.g., screen, multiply)
    const blendedRgb = segWeighted.add(heatWeighted);

    // Clip blended values and convert to Int32 for drawing or further modification
    let finalInt = blendedRgb.clipByValue(0, 255).toInt();

    if (drawPeakOutlines) {
      const finalIntWithPeaks = applyPeakMarkers(
        finalInt,
        heatmapChannelsReshaped,
        peakMarkerColor,
        predHeight,
        predWidth
      );
      finalInt.dispose(); // Dispose the tensor without peaks
      finalInt = finalIntWithPeaks; // Assign the new tensor with peaks
    }

    return finalInt; // Shape finalInt = [H, W, 3]
  });

  // NOTE: Key webgpu optimization, tf.browser.draw avoids gpu->cpu->gpu data transfer for drawing
  // const a = performance.now();
  const results = await tf.browser.draw(finalDrawTensor, tempCombinedCanvas);
  // console.info(`Took ${(performance.now() - a).toFixed(2)} ms to await draw`);

  // --- Dispose intermediate tensors ---
  finalDrawTensor.dispose();
  heatmapColorsTensor.dispose();
  peakMarkerColor.dispose();

  // --- Draw Base Image ---
  targetCtx.clearRect(0, 0, targetWidth, targetHeight);
  targetCtx.imageSmoothingEnabled = true;
  const baseWidth = baseSourceElement.videoWidth || baseSourceElement.width;
  const baseHeight = baseSourceElement.videoHeight || baseSourceElement.height;
  if (!baseWidth || !baseHeight) {
    console.warn("Base source has no dimensions.");
    return;
  }
  const aspectRatio = baseWidth / baseHeight;
  const targetAspectRatio = targetWidth / targetHeight;
  let cropWidth = baseWidth;
  let cropHeight = baseHeight;
  if (aspectRatio > targetAspectRatio) {
    cropWidth = baseHeight * targetAspectRatio;
  } else {
    cropHeight = baseWidth / targetAspectRatio;
  }
  const offsetX = (baseWidth - cropWidth) / 2;
  const offsetY = (baseHeight - cropHeight) / 2;
  targetCtx.drawImage(
    baseSourceElement,
    offsetX,
    offsetY,
    cropWidth,
    cropHeight,
    0,
    0,
    targetWidth,
    targetHeight
  );

  // --- Draw SINGLE Overlay ---
  targetCtx.imageSmoothingEnabled = false; // Disable smoothing for pixelated overlay
  targetCtx.globalAlpha = alpha; // Draw the pre-blended overlay fully opaque
  targetCtx.drawImage(
    tempCombinedCanvas,
    0,
    0,
    predWidth,
    predHeight,
    0,
    0,
    targetWidth,
    targetHeight
  ); // Scale prediction canvas to target size

  // Reset alpha
  targetCtx.globalAlpha = 1.0;
}

// --- Alpha Slider ---
alphaSlider.addEventListener("input", async (event) => {
  // Added async
  const alpha = parseFloat(event.target.value);
  alphaValueSpan.textContent = alpha.toFixed(2); // Update displayed value

  const source = isCameraMode ? videoElement : loadedImage;

  // Use the stored tensor (lastPredictionTensor) to redraw immediately
  // This avoids re-running the model just for an alpha change
  if (source && lastPredictionTensor) {
    // Avoid drawing if a prediction is currently running (esp. in camera mode)
    // This prevents potential race conditions or drawing inconsistencies
    if (!isPredicting) {
      await drawCombined(
        source,
        lastPredictionTensor,
        alpha,
        outputCombinedCanvas
      );
    }
  }
});

// --- Toggle Peaks Button ---
togglePeaksButton.addEventListener("click", async () => {
  drawPeakOutlines = !drawPeakOutlines; // Toggle the state

  // Update button text/style for visual feedback (optional but recommended)
  if (drawPeakOutlines) {
    togglePeaksButton.textContent = "Finding Corners";
    togglePeaksButton.classList.replace("bg-yellow-600", "bg-green-600");
    togglePeaksButton.classList.replace(
      "hover:bg-yellow-700",
      "hover:bg-green-700"
    );
    
  } else {
    togglePeaksButton.textContent = "Hiding Corners";
    togglePeaksButton.classList.replace("bg-green-600", "bg-yellow-600");
    togglePeaksButton.classList.replace(
      "hover:bg-green-700",
      "hover:bg-yellow-700"
    );
    
  }

  // Redraw the output canvas if a prediction exists
  const source = isCameraMode ? videoElement : loadedImage;
  if (source && lastPredictionTensor && !isPredicting) {
    setMessage("Toggling peaks display..."); // Provide feedback
    await drawCombined(
      source,
      lastPredictionTensor,
      parseFloat(alphaSlider.value),
      outputCombinedCanvas
    );
    setMessage(""); // Clear feedback
  }
});

// Clears only the output visualization canvases
function clearOutputCanvases() {
  outCombCtx.clearRect(
    0,
    0,
    outputCombinedCanvas.width,
    outputCombinedCanvas.height
  );
}

// Reset UI elements and state, disposing the stored tensor
function resetUI() {
  predictButton.disabled = true; // Disable manual predict initially
  loadedImage = null;

  // Dispose the stored prediction tensor if it exists
  if (lastPredictionTensor) {
    tf.dispose(lastPredictionTensor);
    lastPredictionTensor = null;
  }

  // Clear canvases based on mode
  clearOutputCanvases();

  setMessage(""); // Clear any messages
  alphaSlider.value = 0.7; // Reset slider
  alphaValueSpan.textContent = "0.7";
  imageInput.disabled = !model; // Disable file input if model not loaded
  imageInput.value = ""; // Clear file input selection

  // Reset toggle button text
  togglePeaksButton.textContent = "Finding Corners";
  togglePeaksButton.classList.add("bg-green-600", "hover:bg-green-700"); // Ensure correct initial color
  togglePeaksButton.classList.remove("bg-yellow-600", "hover:bg-yellow-700"); // Ensure correct initial color
  

  // Reset camera button states appropriately
  if (model) {
    startCameraButton.disabled = false;
  }
  stopCameraButton.disabled = true;
  switchCameraButton.disabled = true;
}

// Display error message and reset state, disposing tensor
function displayError(message) {
  setMessage(message);
  predictButton.disabled = true; // Disable predict on error
  loadedImage = null;
  // Dispose the stored prediction tensor on error
  if (lastPredictionTensor) {
    tf.dispose(lastPredictionTensor);
    lastPredictionTensor = null;
  }
  console.error(message);
  imageInput.disabled = !model; // Re-enable based on model status
}

// --- Load Model & Init ---
loadAppModel(); // Start loading the model immediately
switchMode("file"); // Start in file mode by default
