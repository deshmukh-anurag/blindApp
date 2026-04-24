package com.rahul.yolov8tflite

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class DepthEstimator(
    context: Context,
    modelPath: String,
) {

    private var interpreter: Interpreter

    private var inputWidth = 0
    private var inputHeight = 0

    private var outputWidth = 0
    private var outputHeight = 0

    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(0f, 255f))
        .add(CastOp(DataType.FLOAT32))
        .build()

    init {
        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }

        val model = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(model, options)

        val inputShape = interpreter.getInputTensor(0)?.shape()
        if (inputShape != null) {
            // Expect [1, height, width, 3] or [1, 3, height, width]
            if (inputShape.size >= 4) {
                if (inputShape[3] == 3) {
                    inputHeight = inputShape[1]
                    inputWidth = inputShape[2]
                } else {
                    inputHeight = inputShape[2]
                    inputWidth = inputShape[3]
                }
            }
        }

        val outputShape = interpreter.getOutputTensor(0)?.shape()
        if (outputShape != null) {
            // Common MiDaS-style: [1, h, w, 1] or [1, h, w]
            when (outputShape.size) {
                4 -> {
                    outputHeight = outputShape[1]
                    outputWidth = outputShape[2]
                }
                3 -> {
                    outputHeight = outputShape[1]
                    outputWidth = outputShape[2]
                }
                else -> {
                    // Fallback: best effort
                    outputHeight = outputShape.lastOrNull() ?: 0
                    outputWidth = 1
                }
            }
        }
    }

    fun close() {
        interpreter.close()
    }

    data class DepthResult(
        val data: FloatArray,
        val width: Int,
        val height: Int,
    )

    fun estimateDepth(frame: Bitmap): DepthResult? {
        if (inputWidth == 0 || inputHeight == 0 || outputWidth == 0 || outputHeight == 0) return null

        val resized = Bitmap.createScaledBitmap(frame, inputWidth, inputHeight, false)

        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(resized)
        val processed = imageProcessor.process(tensorImage)
        val imageBuffer = processed.buffer

        val output = TensorBuffer.createFixedSize(
            intArrayOf(1, outputHeight, outputWidth, 1),
            DataType.FLOAT32
        )

        interpreter.run(imageBuffer, output.buffer)

        return DepthResult(
            data = output.floatArray,
            width = outputWidth,
            height = outputHeight,
        )
    }
}
