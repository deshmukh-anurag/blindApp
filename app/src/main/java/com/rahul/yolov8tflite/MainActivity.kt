package com.rahul.yolov8tflite

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.VibrationEffect
import android.os.Vibrator
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
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

// ─────────────────────────────────────────────────────────────────────────────
// App mode enum
// ─────────────────────────────────────────────────────────────────────────────
private enum class AppMode { IDLE, DETECTING, WALKING }
private enum class GuidanceLevel { SAFE, CAUTION, DANGER, IDLE }

class MainActivity : AppCompatActivity(), Detector.DetectorListener {

    private lateinit var binding: ActivityMainBinding
    private val isFrontCamera = false

    // ── Current mode ─────────────────────────────────────────────────────────
    @Volatile private var appMode: AppMode = AppMode.IDLE

    // ── Camera / ML ──────────────────────────────────────────────────────────
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var detector: Detector? = null
    private var depthEstimator: DepthEstimator? = null
    private var segmentor: Segmentor? = null

    // ── Last frame (for on-demand OCR) ───────────────────────────────────────
    @Volatile private var lastFrameForOcr: Bitmap? = null

    // ── Depth / segmentation cache ───────────────────────────────────────────
    @Volatile private var lastDepthMap: FloatArray? = null
    @Volatile private var depthWidth: Int = 0
    @Volatile private var depthHeight: Int = 0
    @Volatile private var lastDepthTime: Long = 0L

    @Volatile private var lastSegMap: Array<IntArray>? = null
    @Volatile private var lastSegTime: Long = 0L

    // ── TTS & OCR ────────────────────────────────────────────────────────────
    private lateinit var textRecognizer: TextRecognizer
    private var tts: TextToSpeech? = null
    @Volatile private var isTtsReady = false

    // ── Walking guidance throttle ─────────────────────────────────────────────
    @Volatile private var lastGuidanceTime: Long = 0L
    @Volatile private var lastGuidanceText: String? = null

    // ── Auto-describe (Detect mode) ───────────────────────────────────────────
    private val autoDescribeHandler = Handler(Looper.getMainLooper())
    private val autoDescribeIntervalMs = 4_000L
    @Volatile private var latestBoxes: List<BoundingBox> = emptyList()

    private val autoDescribeRunnable = object : Runnable {
        override fun run() {
            if (appMode == AppMode.DETECTING) {
                describeScene(latestBoxes)
                autoDescribeHandler.postDelayed(this, autoDescribeIntervalMs)
            }
        }
    }

    // ── Vibration ─────────────────────────────────────────────────────────────
    private var vibrator: Vibrator? = null

    private val uiHandler = Handler(Looper.getMainLooper())
    private lateinit var cameraExecutor: ExecutorService

    // ─────────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ─────────────────────────────────────────────────────────────────────────
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        @Suppress("DEPRECATION")
        vibrator = getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator

        // TTS — uses system voice, works offline
        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = tts?.setLanguage(Locale.getDefault())
                isTtsReady = result != TextToSpeech.LANG_MISSING_DATA &&
                             result != TextToSpeech.LANG_NOT_SUPPORTED
                if (isTtsReady) {
                    tts?.setSpeechRate(0.9f)
                    tts?.setPitch(1.0f)
                }
            }
        }

        // ML Kit OCR — on-device model, fully offline
        textRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

        cameraExecutor = Executors.newSingleThreadExecutor()
        cameraExecutor.execute {
            detector       = Detector(baseContext, MODEL_PATH, LABELS_PATH, this)
            depthEstimator = DepthEstimator(baseContext, DEPTH_MODEL_PATH)
            segmentor      = Segmentor(baseContext, SEGMENTATION_MODEL_PATH)
        }

        if (allPermissionsGranted()) startCamera()
        else ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)

        bindListeners()
    }

    override fun onResume() {
        super.onResume()
        if (allPermissionsGranted()) startCamera()
        else requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
    }

    override fun onDestroy() {
        super.onDestroy()
        autoDescribeHandler.removeCallbacks(autoDescribeRunnable)
        detector?.close()
        depthEstimator?.close()
        segmentor?.close()
        if (::textRecognizer.isInitialized) textRecognizer.close()
        tts?.stop()
        tts?.shutdown()
        cameraExecutor.shutdown()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Button / listener wiring
    // ─────────────────────────────────────────────────────────────────────────
    private fun bindListeners() {

        // ── OCR button ────────────────────────────────────────────────────────
        binding.ocrButton.setOnClickListener {
            val bitmap = lastFrameForOcr
            if (bitmap == null) {
                toast(getString(R.string.no_frame)); return@setOnClickListener
            }
            binding.statusText.text = getString(R.string.status_reading)
            showGuidance("Scanning for text…", GuidanceLevel.IDLE)

            val image = InputImage.fromBitmap(bitmap, 0)
            textRecognizer.process(image)
                .addOnSuccessListener { visionText ->
                    val text = visionText.text.trim()
                    if (text.isBlank()) {
                        toast(getString(R.string.no_text_found))
                        hideGuidance()
                    } else {
                        showGuidance(text, GuidanceLevel.IDLE)
                        speak(text, "OCR_TEXT")
                    }
                    binding.statusText.text = getString(R.string.status_idle)
                }
                .addOnFailureListener { e ->
                    toast("OCR failed: ${e.localizedMessage}")
                    hideGuidance()
                }
        }

        // ── Detect button ─────────────────────────────────────────────────────
        binding.detectButton.setOnClickListener {
            if (appMode == AppMode.DETECTING) setMode(AppMode.IDLE)
            else                              setMode(AppMode.DETECTING)
        }

        // ── Walk button ───────────────────────────────────────────────────────
        binding.walkButton.setOnClickListener {
            if (appMode == AppMode.WALKING) {
                setMode(AppMode.IDLE)
            } else {
                setMode(AppMode.WALKING)
                speak("Navigation mode activated. I will guide you.", "WALK_START")
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Mode management
    // ─────────────────────────────────────────────────────────────────────────
    private fun setMode(mode: AppMode) {
        appMode = mode
        autoDescribeHandler.removeCallbacks(autoDescribeRunnable)

        runOnUiThread {
            // Reset all button backgrounds
            binding.ocrButton.background    = ContextCompat.getDrawable(this, R.drawable.bg_mode_button)
            binding.detectButton.background = ContextCompat.getDrawable(this, R.drawable.bg_mode_button)
            binding.walkButton.background   = ContextCompat.getDrawable(this, R.drawable.bg_mode_button)

            when (mode) {
                AppMode.IDLE -> {
                    binding.overlay.clear()
                    binding.inferenceTime.text = ""
                    binding.statusText.text    = getString(R.string.status_idle)
                    binding.modeLabel.text     = getString(R.string.select_mode)
                    hideGuidance()
                    hideDirectionView()
                    lastGuidanceText = null
                    lastGuidanceTime = 0L
                }
                AppMode.DETECTING -> {
                    binding.detectButton.background =
                        ContextCompat.getDrawable(this, R.drawable.bg_mode_button_active)
                    binding.statusText.text = getString(R.string.status_detecting)
                    binding.modeLabel.text  = "Object Detection"
                    hideDirectionView()
                    hideGuidance()
                    autoDescribeHandler.postDelayed(autoDescribeRunnable, autoDescribeIntervalMs)
                }
                AppMode.WALKING -> {
                    binding.walkButton.background =
                        ContextCompat.getDrawable(this, R.drawable.bg_mode_button_walk_active)
                    binding.detectButton.background =
                        ContextCompat.getDrawable(this, R.drawable.bg_mode_button_active)
                    binding.statusText.text = getString(R.string.status_walking)
                    binding.modeLabel.text  = "Navigation Mode"
                    showDirectionView(DirectionView.Direction.NONE)
                    lastGuidanceText = null
                    lastGuidanceTime = 0L
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Camera setup
    // ─────────────────────────────────────────────────────────────────────────
    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            cameraProvider = future.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val provider = cameraProvider ?: return
        val rotation = binding.viewFinder.display?.rotation ?: return

        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(rotation)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetRotation(rotation)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        imageAnalyzer?.setAnalyzer(cameraExecutor) { imageProxy ->
            val rotDeg = imageProxy.imageInfo.rotationDegrees
            val w = imageProxy.width
            val h = imageProxy.height

            val buffer = imageProxy.planes[0].buffer
            buffer.rewind()
            val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            bmp.copyPixelsFromBuffer(buffer)
            imageProxy.close()

            val matrix = Matrix().apply {
                postRotate(rotDeg.toFloat())
                if (isFrontCamera) postScale(-1f, 1f, w.toFloat(), h.toFloat())
            }
            val rotated = Bitmap.createBitmap(bmp, 0, 0, w, h, matrix, true)
            lastFrameForOcr = rotated

            if (appMode == AppMode.DETECTING || appMode == AppMode.WALKING) {
                detector?.detect(rotated)
            }

            if (appMode == AppMode.WALKING) {
                val now = System.currentTimeMillis()
                if (now - lastDepthTime > 1000L) {
                    lastDepthTime = now
                    depthEstimator?.estimateDepth(rotated)?.let {
                        lastDepthMap = it.data; depthWidth = it.width; depthHeight = it.height
                    }
                }
                if (now - lastSegTime > 1000L) {
                    lastSegTime = now
                    segmentor?.segment(rotated)?.let { lastSegMap = it }
                }
            }
        }

        provider.unbindAll()
        try {
            camera = provider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
            preview?.setSurfaceProvider(binding.viewFinder.surfaceProvider)
        } catch (e: Exception) {
            Log.e(TAG, "Camera bind failed", e)
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Detector callbacks
    // ─────────────────────────────────────────────────────────────────────────
    override fun onEmptyDetect() {
        latestBoxes = emptyList()
        runOnUiThread {
            binding.overlay.clear()
            if (appMode == AppMode.WALKING) provideWalkingGuidance(emptyList())
        }
    }

    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        latestBoxes = boundingBoxes
        runOnUiThread {
            if (appMode == AppMode.IDLE) {
                binding.overlay.clear(); binding.inferenceTime.text = ""; return@runOnUiThread
            }
            binding.inferenceTime.text = "${inferenceTime}ms"
            binding.overlay.setResults(boundingBoxes)
            if (appMode == AppMode.WALKING) provideWalkingGuidance(boundingBoxes)
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Auto-describe (Detect mode)
    // ─────────────────────────────────────────────────────────────────────────
    private fun describeScene(boxes: List<BoundingBox>) {
        if (!isTtsReady) return
        if (boxes.isEmpty()) {
            speak("No objects detected in the scene.", "AUTO_DESC"); return
        }
        val grouped = boxes.groupBy { it.clsName }
            .map { (cls, list) -> if (list.size == 1) cls else "${list.size} ${cls}s" }
        speak("I can see: ${grouped.joinToString(", ")}.", "AUTO_DESC")
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Walking guidance logic
    // ─────────────────────────────────────────────────────────────────────────
    private fun provideWalkingGuidance(boundingBoxes: List<BoundingBox>) {
        if (tts == null || !isTtsReady) return
        val now = System.currentTimeMillis()
        if (now - lastGuidanceTime < 1500L) return
        lastGuidanceTime = now

        val depth = lastDepthMap
        val w = depthWidth; val h = depthHeight
        val seg = lastSegMap

        val threshClose     = 0.60f
        val threshVeryClose = 0.80f

        var guidance: String? = null
        var direction = DirectionView.Direction.FORWARD
        var level     = GuidanceLevel.SAFE

        if (depth != null && w > 0 && h > 0) {
            val center = sampleDepth(depth, w, h, 0.5f, 0.5f)

            // ① Very close obstacle — STOP
            if (center > threshVeryClose) {
                guidance = "Stop! Obstacle very close. Do not move."
                direction = DirectionView.Direction.STOP; level = GuidanceLevel.DANGER
                vibrateStop()
            }

            // ② Wide wall ahead
            if (guidance == null) {
                val midY = ((h - 1) * 0.5f).toInt()
                val sX = ((w - 1) * 0.2f).toInt(); val eX = ((w - 1) * 0.8f).toInt()
                var count = 0
                for (x in sX..eX) { val idx = midY * w + x; if (idx in depth.indices && depth[idx] > threshClose) count++ }
                if (count > (eX - sX + 1) * 0.6f) {
                    val lD = sampleDepth(depth, w, h, 0.2f, 0.5f)
                    val rD = sampleDepth(depth, w, h, 0.8f, 0.5f)
                    if (lD < rD) { guidance = "Wall ahead. Turn left."; direction = DirectionView.Direction.LEFT }
                    else          { guidance = "Wall ahead. Turn right."; direction = DirectionView.Direction.RIGHT }
                    level = GuidanceLevel.CAUTION; vibrateCaution()
                }
            }

            // ③ Named obstacle + depth
            if (guidance == null) {
                val centerBoxes = boundingBoxes.filter { it.cx in 0.35f..0.65f }
                val closest = centerBoxes.maxByOrNull { it.h }
                if (closest != null && center > threshClose) {
                    val label = closest.clsName
                    guidance = when {
                        label.contains("car", true) || label.contains("vehicle", true) ->
                            "Warning! Vehicle very near ahead."
                        label.contains("person", true) || label.contains("human", true) ->
                            "Person very close in front."
                        else -> "Warning! $label very close ahead."
                    }
                    direction = DirectionView.Direction.STOP; level = GuidanceLevel.DANGER; vibrateCaution()
                }
            }

            // ④ Segmentation person
            if (guidance == null && seg != null && sampleSeg(seg, 0.5f, 0.5f) == 15) {
                guidance = "Person directly ahead. Please go around."; direction = DirectionView.Direction.STOP; level = GuidanceLevel.CAUTION
            }

            // ⑤ Smart navigation
            if (guidance == null) {
                val leftD  = sampleDepth(depth, w, h, 0.25f, 0.5f)
                val rightD = sampleDepth(depth, w, h, 0.75f, 0.5f)

                val bY = ((h - 1) * 0.8f).toInt()
                val pS = ((w - 1) * 0.4f).toInt(); val pE = ((w - 1) * 0.6f).toInt()
                var sum = 0f; var n = 0
                for (x in pS..pE) { val idx = bY * w + x; if (idx in depth.indices) { sum += depth[idx]; n++ } }
                val avg = if (n > 0) sum / n else 0f

                var segObs = false; var segL = false; var segR = false
                if (seg != null) {
                    val sH = seg.size; val sW = if (sH > 0) seg[0].size else 0
                    if (sW > 0 && sH > 0) {
                        val byS = (0.85f * (sH - 1)).toInt()
                        val sxS = (0.35f * (sW - 1)).toInt(); val exS = (0.65f * (sW - 1)).toInt()
                        for (x in sxS..exS) { if (seg[byS][x] != 0) { segObs = true; break } }
                        val lC = seg[(0.8f * (sH - 1)).toInt()][(0.2f * (sW - 1)).toInt()]
                        val rC = seg[(0.8f * (sH - 1)).toInt()][(0.8f * (sW - 1)).toInt()]
                        if (lC != 0 && rC == 0) segR = true
                        if (rC != 0 && lC == 0) segL = true
                    }
                }

                when {
                    center > threshClose -> {
                        when {
                            segL -> { guidance = "Move left.";  direction = DirectionView.Direction.LEFT }
                            segR -> { guidance = "Move right."; direction = DirectionView.Direction.RIGHT }
                            else -> if (leftD < rightD) {
                                guidance = "Obstacle ahead. Move left."; direction = DirectionView.Direction.LEFT
                            } else {
                                guidance = "Obstacle ahead. Move right."; direction = DirectionView.Direction.RIGHT
                            }
                        }
                        level = GuidanceLevel.CAUTION
                    }
                    avg > threshClose || segObs -> {
                        guidance = "Caution. Path partially blocked."; direction = DirectionView.Direction.STOP; level = GuidanceLevel.CAUTION
                    }
                    else -> {
                        guidance = "Path clear. Walk forward."; direction = DirectionView.Direction.FORWARD; level = GuidanceLevel.SAFE
                    }
                }
            }

        } else {
            // Fallback — bounding boxes only
            if (boundingBoxes.isEmpty()) {
                guidance = "Path looks clear. Walk forward."; direction = DirectionView.Direction.FORWARD; level = GuidanceLevel.SAFE
            } else {
                val wide   = boundingBoxes.filter { (it.x2 - it.x1) > 0.8f && (it.y2 - it.y1) > 0.6f }
                val center = boundingBoxes.filter { it.cx in 0.33f..0.67f }
                when {
                    wide.isNotEmpty() -> {
                        val obj = wide.maxBy { it.h }
                        guidance = "Stop. ${obj.clsName} blocking the way. Turn left or right."
                        direction = DirectionView.Direction.STOP; level = GuidanceLevel.DANGER; vibrateStop()
                    }
                    center.isNotEmpty() -> {
                        val obj = center.maxBy { it.h }
                        val d = if (obj.cx < 0.5f) "right" else "left"
                        guidance = "Stop. ${obj.clsName} ahead. Move $d."
                        direction = if (d == "left") DirectionView.Direction.LEFT else DirectionView.Direction.RIGHT
                        level = GuidanceLevel.CAUTION; vibrateCaution()
                    }
                    else -> {
                        val lO = boundingBoxes.filter { it.cx < 0.33f }
                        val rO = boundingBoxes.filter { it.cx > 0.67f }
                        when {
                            lO.isNotEmpty() && rO.isNotEmpty() -> {
                                guidance = "Objects on both sides. Walk carefully."
                                direction = DirectionView.Direction.FORWARD; level = GuidanceLevel.CAUTION
                            }
                            lO.isNotEmpty() -> {
                                guidance = "Clear on the right. Move slightly right."
                                direction = DirectionView.Direction.RIGHT; level = GuidanceLevel.CAUTION
                            }
                            rO.isNotEmpty() -> {
                                guidance = "Clear on the left. Move slightly left."
                                direction = DirectionView.Direction.LEFT; level = GuidanceLevel.CAUTION
                            }
                            else -> {
                                guidance = "Path looks clear. Walk forward."
                                direction = DirectionView.Direction.FORWARD; level = GuidanceLevel.SAFE
                            }
                        }
                    }
                }
            }
        }

        guidance?.let {
            showDirectionView(direction)
            showGuidance(it, level)
            speakGuidance(it)
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // UI helpers
    // ─────────────────────────────────────────────────────────────────────────
    private fun showGuidance(text: String, level: GuidanceLevel) {
        runOnUiThread {
            binding.guidanceText.visibility = View.VISIBLE
            binding.guidanceText.text = text
            val bgRes = when (level) {
                GuidanceLevel.SAFE    -> R.drawable.bg_guidance_safe
                GuidanceLevel.CAUTION -> R.drawable.bg_guidance_caution
                GuidanceLevel.DANGER  -> R.drawable.bg_guidance_danger
                GuidanceLevel.IDLE    -> R.drawable.bg_guidance_default
            }
            binding.guidanceText.background = ContextCompat.getDrawable(this, bgRes)
        }
    }

    private fun hideGuidance()      { runOnUiThread { binding.guidanceText.visibility = View.GONE } }

    private fun showDirectionView(dir: DirectionView.Direction) {
        runOnUiThread {
            binding.directionView.visibility = View.VISIBLE
            binding.directionView.setDirection(dir)
        }
    }

    private fun hideDirectionView() { runOnUiThread { binding.directionView.visibility = View.GONE } }

    private fun flashSpeakingDot() {
        runOnUiThread {
            binding.speakingDot.visibility = View.VISIBLE
            uiHandler.postDelayed({ binding.speakingDot.visibility = View.INVISIBLE }, 600)
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // TTS helpers
    // ─────────────────────────────────────────────────────────────────────────
    private fun speak(text: String, utteranceId: String) {
        if (!isTtsReady) { toast(getString(R.string.tts_not_ready)); return }
        flashSpeakingDot()
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, utteranceId)
    }

    private fun speakGuidance(text: String) {
        if (text == lastGuidanceText) return
        lastGuidanceText = text
        speak(text, "WALK_GUIDE")
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Vibration helpers
    // ─────────────────────────────────────────────────────────────────────────
    @Suppress("DEPRECATION")
    private fun vibrateStop() {
        vibrator?.let { v ->
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O)
                v.vibrate(VibrationEffect.createWaveform(longArrayOf(0, 200, 100, 200), -1))
            else v.vibrate(longArrayOf(0, 200, 100, 200), -1)
        }
    }

    @Suppress("DEPRECATION")
    private fun vibrateCaution() {
        vibrator?.let { v ->
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O)
                v.vibrate(VibrationEffect.createOneShot(120, VibrationEffect.DEFAULT_AMPLITUDE))
            else v.vibrate(120)
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Depth / segmentation sampling
    // ─────────────────────────────────────────────────────────────────────────
    private fun sampleDepth(d: FloatArray, w: Int, h: Int, xN: Float, yN: Float): Float {
        if (w <= 0 || h <= 0) return 0f
        val x = (xN.coerceIn(0f, 1f) * (w - 1)).toInt()
        val y = (yN.coerceIn(0f, 1f) * (h - 1)).toInt()
        val idx = y * w + x
        return if (idx in d.indices) d[idx] else 0f
    }

    private fun sampleSeg(seg: Array<IntArray>, xN: Float, yN: Float): Int {
        val sh = seg.size; if (sh == 0) return 0
        val sw = seg[0].size; if (sw == 0) return 0
        val x = (xN.coerceIn(0f, 1f) * (sw - 1)).toInt()
        val y = (yN.coerceIn(0f, 1f) * (sh - 1)).toInt()
        return seg[y][x]
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Permissions
    // ─────────────────────────────────────────────────────────────────────────
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
            if (it[Manifest.permission.CAMERA] == true) startCamera()
        }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()

    companion object {
        private const val TAG = "VisionAI"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
