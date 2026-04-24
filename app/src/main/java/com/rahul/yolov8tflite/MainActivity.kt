package com.rahul.yolov8tflite

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.rahul.yolov8tflite.Constants.DEPTH_MODEL_PATH
import com.rahul.yolov8tflite.Constants.LABELS_PATH
import com.rahul.yolov8tflite.Constants.MODEL_PATH
import com.rahul.yolov8tflite.Constants.SEGMENTATION_MODEL_PATH
import com.rahul.yolov8tflite.databinding.ActivityMainBinding
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), Detector.DetectorListener {
    private lateinit var binding: ActivityMainBinding
    private val isFrontCamera = false

    @Volatile
    private var isDetecting = false

    @Volatile
    private var lastFrameForOcr: Bitmap? = null

    @Volatile
    private var isWalkingMode: Boolean = false

    @Volatile
    private var lastGuidanceTime: Long = 0L

    @Volatile
    private var lastGuidanceText: String? = null

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var detector: Detector? = null
    private var depthEstimator: DepthEstimator? = null
    private var segmentor: Segmentor? = null

    @Volatile
    private var lastDepthMap: FloatArray? = null
    @Volatile
    private var depthWidth: Int = 0
    @Volatile
    private var depthHeight: Int = 0
    @Volatile
    private var lastDepthTime: Long = 0L

    @Volatile
    private var lastSegMap: Array<IntArray>? = null
    @Volatile
    private var lastSegTime: Long = 0L

    private lateinit var textRecognizer: TextRecognizer
    private var tts: TextToSpeech? = null
    @Volatile
    private var isTtsReady: Boolean = false

    private lateinit var cameraExecutor: ExecutorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        textRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = tts?.setLanguage(Locale.getDefault())
                isTtsReady = result != TextToSpeech.LANG_MISSING_DATA && result != TextToSpeech.LANG_NOT_SUPPORTED
            } else {
                isTtsReady = false
            }
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        cameraExecutor.execute {
            detector = Detector(baseContext, MODEL_PATH, LABELS_PATH, this)
            depthEstimator = DepthEstimator(baseContext, DEPTH_MODEL_PATH)
            segmentor = Segmentor(baseContext, SEGMENTATION_MODEL_PATH)
        }

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        bindListeners()
    }

    private fun bindListeners() {
        binding.detectToggle.apply {
            isChecked = false
            setBackgroundColor(ContextCompat.getColor(baseContext, R.color.gray))

            setOnCheckedChangeListener { _, checked ->
                isDetecting = checked
                val colorRes = if (checked) R.color.red else R.color.gray
                setBackgroundColor(ContextCompat.getColor(baseContext, colorRes))

                if (!checked) {
                    binding.inferenceTime.text = ""
                    binding.overlay.clear()
                }
            }
        }

        binding.walkingModeToggle.apply {
            isChecked = false
            setBackgroundColor(ContextCompat.getColor(baseContext, R.color.gray))

            setOnCheckedChangeListener { _, checked ->
                isWalkingMode = checked
                val colorRes = if (checked) R.color.red else R.color.gray
                setBackgroundColor(ContextCompat.getColor(baseContext, colorRes))

                if (checked && !binding.detectToggle.isChecked) {
                    binding.detectToggle.isChecked = true
                }

                if (!checked) {
                    lastGuidanceText = null
                    lastGuidanceTime = 0L
                }
            }
        }

        binding.readTextButton.setOnClickListener {
            val bitmap = lastFrameForOcr
            if (bitmap == null) {
                Toast.makeText(this, "No frame available yet", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val image = InputImage.fromBitmap(bitmap, 0)
            textRecognizer.process(image)
                .addOnSuccessListener { visionText ->
                    val text = visionText.text
                    if (text.isBlank()) {
                        Toast.makeText(this, "No text detected", Toast.LENGTH_SHORT).show()
                    } else {
                        if (tts == null || !isTtsReady) {
                            Toast.makeText(this, "Text-to-speech is initializing, please wait...", Toast.LENGTH_SHORT).show()
                        } else {
                            tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "OCR_TEXT")
                        }
                    }
                }
                .addOnFailureListener { e ->
                    Toast.makeText(
                        this,
                        "Text recognition failed: ${e.message}",
                        Toast.LENGTH_SHORT
                    ).show()
                }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider  = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera initialization failed.")

        val rotation = binding.viewFinder.display.rotation

        val cameraSelector = CameraSelector
            .Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview =  Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(rotation)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetRotation(binding.viewFinder.display.rotation)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        imageAnalyzer?.setAnalyzer(cameraExecutor) { imageProxy ->
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees
            val width = imageProxy.width
            val height = imageProxy.height
            val buffer = imageProxy.planes[0].buffer
            buffer.rewind()

            val bitmapBuffer = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            bitmapBuffer.copyPixelsFromBuffer(buffer)
            imageProxy.close()

            val matrix = Matrix().apply {
                postRotate(rotationDegrees.toFloat())
                if (isFrontCamera) {
                    postScale(-1f, 1f, width.toFloat(), height.toFloat())
                }
            }

            val rotatedBitmap = Bitmap.createBitmap(bitmapBuffer, 0, 0, width, height, matrix, true)
            lastFrameForOcr = rotatedBitmap

            if (isDetecting) {
                detector?.detect(rotatedBitmap)
            }

            if (isWalkingMode) {
                val now = System.currentTimeMillis()
                if (now - lastDepthTime > 1000L) {
                    lastDepthTime = now
                    depthEstimator?.estimateDepth(rotatedBitmap)?.let { result ->
                        lastDepthMap = result.data
                        depthWidth = result.width
                        depthHeight = result.height
                    }
                }

                if (now - lastSegTime > 1000L) {
                    lastSegTime = now
                    segmentor?.segment(rotatedBitmap)?.let { seg ->
                        lastSegMap = seg
                    }
                }
            }
        }

        cameraProvider.unbindAll()

        try {
            camera = cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )

            preview?.setSurfaceProvider(binding.viewFinder.surfaceProvider)
        } catch(exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()) {
        if (it[Manifest.permission.CAMERA] == true) { startCamera() }
    }

    override fun onDestroy() {
        super.onDestroy()
        detector?.close()
        depthEstimator?.close()
        segmentor?.close()
        if (::textRecognizer.isInitialized) {
            textRecognizer.close()
        }
        tts?.stop()
        tts?.shutdown()
        cameraExecutor.shutdown()
    }

    override fun onResume() {
        super.onResume()
        if (allPermissionsGranted()){
            startCamera()
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }
    }

    companion object {
        private const val TAG = "Camera"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = mutableListOf (
            Manifest.permission.CAMERA
        ).toTypedArray()
    }

    override fun onEmptyDetect() {
        runOnUiThread {
            binding.overlay.clear()
        }
    }

    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        runOnUiThread {
            if (!isDetecting) {
                binding.inferenceTime.text = ""
                binding.overlay.clear()
                return@runOnUiThread
            }

            binding.inferenceTime.text = "${inferenceTime}ms"
            binding.overlay.apply {
                setResults(boundingBoxes)
                invalidate()
            }

            if (isWalkingMode) {
                provideWalkingGuidance(boundingBoxes)
            }
        }
    }

    private fun provideWalkingGuidance(boundingBoxes: List<BoundingBox>) {
        if (tts == null || !isTtsReady) return

        val now = System.currentTimeMillis()
        if (now - lastGuidanceTime < 1000L) return

        lastGuidanceTime = now

        val depth = lastDepthMap
        val w = depthWidth
        val h = depthHeight
        val seg = lastSegMap

        val depthThresholdClose = 0.6f
        val depthThresholdVeryClose = 0.8f

        var guidance: String? = null

        if (depth != null && w > 0 && h > 0) {
            // 1. Obstacle detection in center
            val center = sampleDepth(depth, w, h, 0.5f, 0.5f)
            if (center > depthThresholdVeryClose) {
                guidance = "Obstacle very close. Stop."
            }

            // 5. Wall detection – wide area in front
            if (guidance == null) {
                val midY = ((h - 1) * 0.5f).toInt()
                val startX = ((w - 1) * 0.2f).toInt()
                val endX = ((w - 1) * 0.8f).toInt()
                var count = 0
                for (x in startX..endX) {
                    val idx = midY * w + x
                    if (idx in depth.indices && depth[idx] > depthThresholdClose) count++
                }
                if (count > (endX - startX + 1) * 0.6f) {
                    guidance = "Wall ahead. Turn slightly left or right."
                }
            }

            // 4. Danger detection – object + depth
            if (guidance == null) {
                val centerBoxes = boundingBoxes.filter { it.cx in 0.4f..0.6f }
                val closest = centerBoxes.maxByOrNull { it.h }
                if (closest != null && center > depthThresholdClose) {
                    val label = closest.clsName
                    guidance = when {
                        label.contains("car", true) || label.contains("vehicle", true) ->
                            "Warning, vehicle very near in front."
                        label.contains("person", true) || label.contains("human", true) ->
                            "Person very near in front."
                        else ->
                            "Warning, $label very close in front."
                    }
                }
            }

            // Segmentation-based person / obstacle hints (DeepLab)
            if (guidance == null && seg != null) {
                val centerClass = sampleSeg(seg, 0.5f, 0.5f)
                if (centerClass == 15) {
                    guidance = "Person ahead."
                }
            }

            // 2 & 3. Smart navigation + free path detection (depth + segmentation)
            if (guidance == null) {
                val leftDepth = sampleDepth(depth, w, h, 0.25f, 0.5f)
                val rightDepth = sampleDepth(depth, w, h, 0.75f, 0.5f)

                val bottomY = ((h - 1) * 0.8f).toInt()
                val pathStartX = ((w - 1) * 0.4f).toInt()
                val pathEndX = ((w - 1) * 0.6f).toInt()
                var sum = 0f
                var n = 0
                for (x in pathStartX..pathEndX) {
                    val idx = bottomY * w + x
                    if (idx in depth.indices) {
                        sum += depth[idx]
                        n++
                    }
                }
                val avg = if (n > 0) sum / n else 0f

                // Segmentation: check bottom band for obstacles (non-background)
                var segObstacle = false
                var segMoveLeft = false
                var segMoveRight = false
                if (seg != null) {
                    val segH = seg.size
                    val segW = if (segH > 0) seg[0].size else 0
                    if (segW > 0 && segH > 0) {
                        val segBottomY = (0.85f * (segH - 1)).toInt()
                        val segStartX = (0.35f * (segW - 1)).toInt()
                        val segEndX = (0.65f * (segW - 1)).toInt()
                        for (x in segStartX..segEndX) {
                            if (seg[segBottomY][x] != 0) {
                                segObstacle = true
                                break
                            }
                        }

                        val segLeft = seg[(0.8f * (segH - 1)).toInt()][(0.2f * (segW - 1)).toInt()]
                        val segRight = seg[(0.8f * (segH - 1)).toInt()][(0.8f * (segW - 1)).toInt()]
                        if (segLeft != 0 && segRight == 0) segMoveRight = true
                        if (segRight != 0 && segLeft == 0) segMoveLeft = true
                    }
                }

                guidance = when {
                    center > depthThresholdClose -> {
                        when {
                            segMoveLeft -> "Move left."
                            segMoveRight -> "Move right."
                            else -> if (leftDepth < rightDepth) "Move left." else "Move right."
                        }
                    }
                    avg > depthThresholdClose || segObstacle -> {
                        "Obstacle ahead. Path blocked."
                    }
                    else -> {
                        "Path clear. You can go forward."
                    }
                }
            }
        } else {
            // Fallback: old bounding-box-only logic
            if (boundingBoxes.isEmpty()) {
                guidance = "Path looks clear. You can walk forward."
            } else {
                val centerLeft = 0.33f
                val centerRight = 0.67f

                val wideBlocking = boundingBoxes.filter { box ->
                    val width = box.x2 - box.x1
                    val height = box.y2 - box.y1
                    width > 0.8f && height > 0.6f
                }

                val centerBlocking = boundingBoxes.filter { box ->
                    box.cx in centerLeft..centerRight
                }

                guidance = when {
                    wideBlocking.isNotEmpty() -> {
                        val obj = wideBlocking.maxBy { it.h }
                        "Stop. ${obj.clsName} is blocking the way. Turn slightly left or right to find a free path."
                    }
                    centerBlocking.isNotEmpty() -> {
                        val obj = centerBlocking.maxBy { it.h }
                        val direction = if (obj.cx < 0.5f) "right" else "left"
                        "Stop. ${obj.clsName} ahead in front. Move a little $direction."
                    }
                    else -> {
                        val centerLeftB = 0.33f
                        val centerRightB = 0.67f
                        val leftObjects = boundingBoxes.filter { it.cx < centerLeftB }
                        val rightObjects = boundingBoxes.filter { it.cx > centerRightB }

                        when {
                            leftObjects.isNotEmpty() && rightObjects.isNotEmpty() ->
                                "Objects on both sides. Move carefully straight ahead."
                            leftObjects.isNotEmpty() ->
                                "Clear on the right. You can move slightly right."
                            rightObjects.isNotEmpty() ->
                                "Clear on the left. You can move slightly left."
                            else ->
                                "Path looks clear. You can walk forward."
                        }
                    }
                }
            }
        }

        guidance?.let { speakGuidance(it) }
    }

    private fun sampleDepth(depth: FloatArray, width: Int, height: Int, xNorm: Float, yNorm: Float): Float {
        if (width <= 0 || height <= 0) return 0f
        val xn = xNorm.coerceIn(0f, 1f)
        val yn = yNorm.coerceIn(0f, 1f)
        val x = (xn * (width - 1)).toInt()
        val y = (yn * (height - 1)).toInt()
        val idx = y * width + x
        return if (idx in depth.indices) depth[idx] else 0f
    }

    private fun sampleSeg(seg: Array<IntArray>, xNorm: Float, yNorm: Float): Int {
        val h = seg.size
        if (h == 0) return 0
        val w = seg[0].size
        if (w == 0) return 0
        val xn = xNorm.coerceIn(0f, 1f)
        val yn = yNorm.coerceIn(0f, 1f)
        val x = (xn * (w - 1)).toInt()
        val y = (yn * (h - 1)).toInt()
        return seg[y][x]
    }

    private fun speakGuidance(text: String) {
        if (text == lastGuidanceText) return
        lastGuidanceText = text
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "WALK_GUIDE")
    }
}
