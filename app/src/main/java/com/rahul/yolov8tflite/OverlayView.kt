package com.rahul.yolov8tflite

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

/**
 * Overlay view that draws bounding boxes over detected objects.
 *
 * Features:
 *  - Colour-coded boxes per object class (cycling a vivid palette)
 *  - Rounded-corner rectangles for a modern look
 *  - Label pill with class name + confidence percentage
 *  - Semi-transparent fill inside each box (subtle depth cue)
 *  - Corner "targeting" tick marks for a HUD aesthetic
 *  - Confidence-based stroke thickness
 */
class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results = listOf<BoundingBox>()

    // ── Vivid palette that reads clearly over a dark camera scene ──────────
    private val palette = intArrayOf(
        0xFF4A90E2.toInt(), // bright blue
        0xFF3DDC84.toInt(), // android green
        0xFFFF9800.toInt(), // orange
        0xFFE91E63.toInt(), // pink
        0xFF9C27B0.toInt(), // purple
        0xFF00BCD4.toInt(), // cyan
        0xFFF44336.toInt(), // red
        0xFFFFEB3B.toInt(), // yellow
        0xFF8BC34A.toInt(), // light green
        0xFFFF5722.toInt(), // deep orange
        0xFF03A9F4.toInt(), // light blue
        0xFFCDDC39.toInt(), // lime
    )

    // Stable colour per class name so the same object always gets the same colour
    private val classColorMap = HashMap<String, Int>()
    private var colorCounter = 0

    private fun colorForClass(cls: String): Int {
        val idx = classColorMap.getOrPut(cls) { colorCounter++ % palette.size }
        return palette[idx % palette.size]
    }

    // ── Paints ──────────────────────────────────────────────────────────────
    private val boxStrokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeJoin = Paint.Join.ROUND
        strokeCap  = Paint.Cap.ROUND
    }

    private val boxFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val labelBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val labelTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color    = Color.WHITE
        style    = Paint.Style.FILL
        typeface = Typeface.create(Typeface.DEFAULT_BOLD, Typeface.BOLD)
        textSize = 34f
    }

    private val tickPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style     = Paint.Style.STROKE
        strokeWidth = 5f
        strokeCap = Paint.Cap.ROUND
    }

    private val cornerRadius = 14f
    private val labelRadius  = 10f
    private val labelPadH    = 12f
    private val labelPadV    = 7f
    private val tickLen      = 20f

    // ── Public API ──────────────────────────────────────────────────────────
    fun clear() {
        results = listOf()
        invalidate()
    }

    fun setResults(boxes: List<BoundingBox>) {
        results = boxes
        invalidate()
    }

    // ── Drawing ─────────────────────────────────────────────────────────────
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        for (box in results) {
            val left   = box.x1 * width
            val top    = box.y1 * height
            val right  = box.x2 * width
            val bottom = box.y2 * height
            val rect   = RectF(left, top, right, bottom)

            val color = colorForClass(box.clsName)

            // 1. Subtle fill
            boxFillPaint.color = color
            boxFillPaint.alpha = 28
            canvas.drawRoundRect(rect, cornerRadius, cornerRadius, boxFillPaint)

            // 2. Stroke — confidence drives thickness and opacity
            boxStrokePaint.color = color
            boxStrokePaint.alpha = (180 + box.cnf * 75).toInt().coerceIn(180, 255)
            boxStrokePaint.strokeWidth = 3.5f + box.cnf * 4f
            canvas.drawRoundRect(rect, cornerRadius, cornerRadius, boxStrokePaint)

            // 3. Corner ticks
            tickPaint.color = color
            tickPaint.alpha = 220
            drawCornerTicks(canvas, rect)

            // 4. Label pill
            val label = "${box.clsName}  ${(box.cnf * 100).toInt()}%"
            val textW  = labelTextPaint.measureText(label)
            val ascent = -labelTextPaint.ascent()
            val descent = labelTextPaint.descent()
            val pillW = textW + labelPadH * 2
            val pillH = ascent + descent + labelPadV * 2

            val pillL = left.coerceAtMost(width  - pillW)
            val pillT = (top - pillH - 4f).coerceAtLeast(0f)
            val pillRect = RectF(pillL, pillT, pillL + pillW, pillT + pillH)

            labelBgPaint.color = color
            labelBgPaint.alpha = 225
            canvas.drawRoundRect(pillRect, labelRadius, labelRadius, labelBgPaint)

            labelTextPaint.color = Color.WHITE
            canvas.drawText(label, pillL + labelPadH, pillT + labelPadV + ascent, labelTextPaint)
        }
    }

    private fun drawCornerTicks(canvas: Canvas, r: RectF) {
        // Top-left
        canvas.drawLine(r.left,           r.top + tickLen, r.left,           r.top, tickPaint)
        canvas.drawLine(r.left,           r.top,           r.left + tickLen, r.top, tickPaint)
        // Top-right
        canvas.drawLine(r.right - tickLen, r.top,           r.right,          r.top, tickPaint)
        canvas.drawLine(r.right,           r.top,           r.right, r.top + tickLen, tickPaint)
        // Bottom-left
        canvas.drawLine(r.left,            r.bottom - tickLen, r.left,            r.bottom, tickPaint)
        canvas.drawLine(r.left,            r.bottom,           r.left + tickLen,  r.bottom, tickPaint)
        // Bottom-right
        canvas.drawLine(r.right - tickLen, r.bottom, r.right, r.bottom, tickPaint)
        canvas.drawLine(r.right,           r.bottom, r.right, r.bottom - tickLen, tickPaint)
    }

    companion object {
        @Suppress("unused")
        private const val BOUNDING_RECT_TEXT_PADDING = 8 // kept for API compatibility
    }
}
