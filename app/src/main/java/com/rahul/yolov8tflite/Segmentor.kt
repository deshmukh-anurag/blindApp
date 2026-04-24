package com.rahul.yolov8tflite

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder

class Segmentor(
    context: Context,
    modelPath: String,
) {

    private val interpreter: Interpreter
    private val inputSize = 257

    init {
        val model = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(model)
    }

    fun close() {
        interpreter.close()
    }

    fun segment(bitmap: Bitmap): Array<IntArray> {
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        val input = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        input.order(ByteOrder.nativeOrder())

        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                val pixel = resized.getPixel(x, y)
                val r = (pixel shr 16 and 0xFF) / 255.0f
                val g = (pixel shr 8 and 0xFF) / 255.0f
                val b = (pixel and 0xFF) / 255.0f
                input.putFloat(r)
                input.putFloat(g)
                input.putFloat(b)
            }
        }

        // Output: [1, 257, 257, 21] class scores
        val output = Array(1) { Array(inputSize) { Array(inputSize) { FloatArray(21) } } }

        interpreter.run(input, output)

        val result = Array(inputSize) { IntArray(inputSize) }

        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                var maxIndex = 0
                var maxValue = output[0][y][x][0]
                for (c in 1 until 21) {
                    val v = output[0][y][x][c]
                    if (v > maxValue) {
                        maxValue = v
                        maxIndex = c
                    }
                }
                result[y][x] = maxIndex
            }
        }

        return result
    }
}
