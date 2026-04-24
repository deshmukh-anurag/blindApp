package com.rahul.yolov8tflite

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

/**
 * Custom view that draws a large directional arrow indicator for the blind navigation mode.
 * States: FORWARD (safe green), LEFT (orange turn), RIGHT (orange turn), STOP (red X), NONE (hidden).
 */
class DirectionView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    enum class Direction { FORWARD, LEFT, RIGHT, STOP, NONE }

    private var currentDirection = Direction.NONE

    // Colors
    private val colorSafe    = Color.parseColor("#3DDC84")
    private val colorCaution = Color.parseColor("#FF9800")
    private val colorDanger  = Color.parseColor("#F44336")
    private val colorIdle    = Color.parseColor("#2A3A5A")

    private val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val ringPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    private val shapePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.FILL
    }

    private val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 14f
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }

    fun setDirection(dir: Direction) {
        if (currentDirection == dir) return
        currentDirection = dir
        invalidate()
    }

    private fun accentColor(): Int = when (currentDirection) {
        Direction.FORWARD -> colorSafe
        Direction.LEFT, Direction.RIGHT -> colorCaution
        Direction.STOP -> colorDanger
        Direction.NONE -> colorIdle
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val cx = width / 2f
        val cy = height / 2f
        val r  = min(width, height) / 2f - 6f
        val ac = accentColor()

        // Outer translucent ring
        ringPaint.color = ac
        ringPaint.alpha = 60
        canvas.drawCircle(cx, cy, r, ringPaint)

        // Filled circle
        bgPaint.color = ac
        bgPaint.alpha = 190
        canvas.drawCircle(cx, cy, r - 10f, bgPaint)

        // Inner white ring
        ringPaint.alpha = 80
        ringPaint.color = Color.WHITE
        ringPaint.strokeWidth = 3f
        canvas.drawCircle(cx, cy, r - 10f, ringPaint)

        when (currentDirection) {
            Direction.FORWARD -> drawArrow(canvas, cx, cy, r, 0f)
            Direction.LEFT    -> drawArrow(canvas, cx, cy, r, -90f)
            Direction.RIGHT   -> drawArrow(canvas, cx, cy, r,  90f)
            Direction.STOP    -> drawStopX(canvas, cx, cy, r)
            Direction.NONE    -> {}
        }
    }

    /** Draws a chunky upward arrow, rotated by [angleDeg] around the centre. */
    private fun drawArrow(canvas: Canvas, cx: Float, cy: Float, r: Float, angleDeg: Float) {
        canvas.save()
        canvas.rotate(angleDeg, cx, cy)

        val headH  = r * 0.55f   // arrow head height
        val headW  = r * 0.55f   // arrow head half-width
        val shaftW = r * 0.22f   // shaft half-width
        val shaftB = r * 0.55f   // shaft bottom from centre

        val path = Path().apply {
            // tip
            moveTo(cx, cy - headH)
            // head right slope
            lineTo(cx + headW, cy)
            // step down to shaft right
            lineTo(cx + shaftW, cy)
            // shaft right side down
            lineTo(cx + shaftW, cy + shaftB)
            // shaft bottom right-to-left
            lineTo(cx - shaftW, cy + shaftB)
            // shaft left side up
            lineTo(cx - shaftW, cy)
            // step out to head left
            lineTo(cx - headW, cy)
            close()
        }

        shapePaint.color = Color.WHITE
        shapePaint.alpha = 240
        canvas.drawPath(path, shapePaint)
        canvas.restore()
    }

    /** Draws an X (stop symbol). */
    private fun drawStopX(canvas: Canvas, cx: Float, cy: Float, r: Float) {
        val arm = r * 0.42f
        strokePaint.strokeWidth = r * 0.18f
        strokePaint.color = Color.WHITE
        strokePaint.alpha = 240
        canvas.drawLine(cx - arm, cy - arm, cx + arm, cy + arm, strokePaint)
        canvas.drawLine(cx + arm, cy - arm, cx - arm, cy + arm, strokePaint)
    }
}
