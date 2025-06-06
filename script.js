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
// const MODEL_URL = "./m128_att_v3_q8/model.json";

// Tiny UNet Model, quantized uint8, ~645 KB
const MODEL_URL = "./4_unet_q8/model.json";


const TARGET_IMG_SIZE = 128; // Model's expected input size
const TARGET_FPS = 100; // Max FPS, but really camera FPS constrains to 30 most of the time.
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
let lastPredictOnlyTime = 0; // Only time for last predict (Doesn't handle async accurately)
let currentVideoTime = 0;
let lastProcessedVideoTime = -1;
let predictFrameCount = 0; // Track Predict update count for predict FPS
let lastFpsUpdate = 0;
let videoDevices = [];
let currentDeviceId = null;
let devicesEnumerated = false;

// --- Temporary Canvas for model output  ---
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
    tf.enableProdMode(); // Likely doesn't affect anything, but just in case.
    model = await tf.loadGraphModel(MODEL_URL);
    setStatus("Warming up...", true);
    // Warmup with the correct input size
    const warmupTensor = tf.zeros([1, TARGET_IMG_SIZE, TARGET_IMG_SIZE, 3]);
    const warmupResult = model.predict(warmupTensor);
    // Create output canvas and size to model output
    tempCombinedCanvas = document.createElement("canvas");
    tempCombinedCanvas.width = TARGET_IMG_SIZE;
    tempCombinedCanvas.height = TARGET_IMG_SIZE;
    // Dispose warmup results
    if (Array.isArray(warmupResult)) {
        warmupResult.forEach(t => t.dispose());
    } else {
        // This case should ideally not happen if the model output is consistent
        console.warn("Warmup result was not an array as expected.");
        if (warmupResult) warmupResult.dispose();
    }
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
          lastPredictionTensor.forEach(t => tf.dispose(t));
          lastPredictionTensor = null;
        }
        if (model && !isPredicting) {
          setMessage("Image loaded. Auto-predicting...");

          // Update UI for file mode prediction start
          setStatus("Processing...", true);
          predictButton.disabled = true;
          imageInput.disabled = true; // Disable while predicting
          setMessage("Preprocessing & Predicting...");

          // await runPrediction(); // Automatically predict on new image load
          const tm = await tf.time(() => runPrediction());
          setMessage(`Prediction took ${tm.wallMs.toFixed(0)}ms (${tm.kernelMs.toFixed(0)}ms in kernel)`);
          // Update UI for file mode prediction end
          setStatus("Model Ready");
          // Re-enable controls only if model and image are loaded
          predictButton.disabled = !model || !loadedImage;
          imageInput.disabled = !model;
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

  predictButton.disabled = true;
  setStatus("Processing...", true);
  const tm = await tf.time(() => runPrediction());
  setMessage(`Prediction took ${tm.wallMs.toFixed(0)}ms (${tm.kernelMs.toFixed(0)}ms in kernel)`);
  predictButton.disabled = false;
  setStatus("Model Ready");
});

// --- Core Prediction Logic ---
async function runPrediction(sourceElement = null) {
  const currentSource = sourceElement || loadedImage;
  if (!currentSource || !model || isPredicting) {
    return; // Exit if no source, model, or already predicting
  }
  isPredicting = true; // Set prediction flag
  let prediction_tensors = null; // This will hold the "kept" result from tf.tidy: [segTensor, heatTensor]

  try {
    // Preprocess the source. preprocessSource uses its own tf.tidy,
    // so inputTensorForPrediction is "kept" from that tidy.
    const inputTensorForPrediction = preprocessSource(currentSource, TARGET_IMG_SIZE);
    // Run model prediction. Expecting an array: [segTensorWithBatch, heatTensorWithBatch]
    let [segTensor, heatTensor] = model.predict(inputTensorForPrediction);
    inputTensorForPrediction.dispose();

    // Squeeze batch dimension if present for each tensor
    segTensor = segTensor.squeeze([0]); // [H, W, 1]
    heatTensor = heatTensor.squeeze([0]); // [H, W, 4]

    // These two tensors will be "kept" and returned from tf.tidy as an array
    prediction_tensors = [segTensor, heatTensor];

    if (lastPredictionTensor) {
      lastPredictionTensor.forEach(t => tf.dispose(t)); // Dispose elements of the array
      lastPredictionTensor = null;
    }
    // Clone the "kept" tensors from tidy for later use
    lastPredictionTensor = [
        tf.keep(prediction_tensors[0].clone()),
        tf.keep(prediction_tensors[1].clone())
    ];

    await drawCombined(
      currentSource,
      prediction_tensors, // Pass the array of tensors [segTensor, heatTensor]
      parseFloat(alphaSlider.value),
      outputCombinedCanvas
    );
  } catch (error) {
    console.error("Prediction Error:", error);
    if (!isCameraMode) displayError(`Processing error: ${error.message}`);
    if (lastPredictionTensor) {
      lastPredictionTensor.forEach(t => tf.dispose(t));
      lastPredictionTensor = null;
    }
    if (prediction_tensors && Array.isArray(prediction_tensors)) prediction_tensors.forEach(t => tf.dispose(t));

  } finally {
    isPredicting = false; // Clear prediction flag
    if (prediction_tensors && Array.isArray(prediction_tensors)) prediction_tensors.forEach(t => tf.dispose(t));
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

  // Throttle predictions based on TARGET_FPS
  if (elapsed > MS_PER_FRAME && !isPredicting) {
    const currentVideoTime = videoElement.currentTime;
    // Only process if a new video frame is available
    if ((videoElement.readyState >= videoElement.HAVE_CURRENT_DATA) && 
        (currentVideoTime > lastProcessedVideoTime)) {
      predictFrameCount++;
      lastPredictTime = now; // Update time *before* prediction starts
      lastProcessedVideoTime = currentVideoTime; // Update last video frame time
      setMessage(" "); // Clear previous messages in camera mode
      await runPrediction(videoElement); // Run prediction on the current video frame
    }
  }

  // Update FPS counter approx every second
  if (now - lastFpsUpdate > 1000) {
    const timeTakenMs = now - lastFpsUpdate; // Time in ms.
    const perStepAvgMs = timeTakenMs / predictFrameCount;
    const fps = 1000 * predictFrameCount / timeTakenMs;

    const camera_framerate = videoElement.srcObject.getVideoTracks()[0].getSettings().frameRate;
    fpsDisplay.textContent = `${fps.toFixed(1)} / ${camera_framerate} | ${perStepAvgMs.toFixed(1)}ms/step | ${predictFrameCount} steps`;
    // fpsDisplay.textContent = tMsg;
    // fpsDisplay.textContent = fps.toFixed(1);
    predictFrameCount = 0;
    lastFpsUpdate = now;
  }
  

  // Request the next frame
  animationFrameId = requestAnimationFrame(predictLoop);
}

// --- Preprocessing ---
function preprocessSource(sourceElement, targetSize) {
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
      [targetSize, targetSize], // Target output size [height, width]
    ); // Output shape: [1, targetSize, targetSize, 3]

    // Normalize to 0-1 range
    const normalized = cropped.div(tf.scalar(255.0));

    // Dispose the initial tensor from pixels
    tensor.dispose();

    // Return the preprocessed tensor (still has batch dimension)
    // Shape: [1, targetSize, targetSize, 3]
    return normalized;
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
  predictionTensors, // Now an array [segTensor, heatTensor]
  alpha,
  targetCanvas
) {
  if (!baseSourceElement || !predictionTensors || !Array.isArray(predictionTensors) || predictionTensors.length !== 2) {
    console.warn("drawCombined: Invalid inputs", baseSourceElement, predictionTensors);
    return;
  }

  const [segTensor, heatTensor] = predictionTensors; // Destructure
  if (!segTensor || !heatTensor) {
    console.warn("drawCombined: One or both tensors in predictionTensors are undefined.");
    return;
  }

  const targetCtx = targetCanvas.getContext("2d");
  const targetWidth = targetCanvas.width;
  const targetHeight = targetCanvas.height;

  // const [predHeight, predWidth] = segTensor.shape.slice(0, 2); // e.g. [128, 128], assuming H,W match
  const predWidth = TARGET_IMG_SIZE;
  const predHeight = TARGET_IMG_SIZE;

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
  const finalDrawTensor = tf.tidy(() => {
    // Calculate desired display alphas
    const segDisplayAlpha = Math.min(1.0, alpha + 0.2);

    // --- Segmentation Mask ---
    const segChannelScaled = segTensor.mul(175); // Segmentation as not fully white gray
    const segRgb = tf.concat(
      [segChannelScaled, segChannelScaled, segChannelScaled],
      2
    ); // Shape [H, W, 3]

    // --- Heatmap ---
    const heatmapChannelsReshaped = heatTensor.reshape([predHeight * predWidth, 4]); // Shape: [H*W, 4]

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
  const results = await tf.browser.draw(finalDrawTensor, tempCombinedCanvas);

  // --- Dispose intermediate tensors ---
  // finalDrawTensor.dispose();
  heatmapColorsTensor.dispose();
  peakMarkerColor.dispose();

  // --- Draw Base Image ---
  // targetCtx.clearRect(0, 0, targetWidth, targetHeight); // Will be overwriting anyway
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
  drawPeakOutlines = !drawPeakOutlines;

  if (drawPeakOutlines) {
    togglePeaksButton.textContent = "Finding Corners";
    togglePeaksButton.classList.remove("toggle-peaks-inactive");
    togglePeaksButton.classList.add("toggle-peaks-active");
  } else {
    togglePeaksButton.textContent = "Hiding Corners";
    togglePeaksButton.classList.remove("toggle-peaks-active");
    togglePeaksButton.classList.add("toggle-peaks-inactive");
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
    lastPredictionTensor.forEach(t => tf.dispose(t));
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
  togglePeaksButton.classList.remove("toggle-peaks-inactive");
  togglePeaksButton.classList.add("toggle-peaks-active");

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
    lastPredictionTensor.forEach(t => tf.dispose(t));
    lastPredictionTensor = null;
  }
  console.error(message);
  imageInput.disabled = !model; // Re-enable based on model status
}

// --- Load Model & Init ---
loadAppModel(); // Start loading the model immediately
switchMode("file"); // Start in file mode by default
