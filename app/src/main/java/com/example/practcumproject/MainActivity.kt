package com.example.practcumproject

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.SystemClock
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay
import java.io.File
import java.util.concurrent.TimeUnit


private val Aqua = Color(0xFF00E5FF)

private val DarkColors = darkColorScheme(
    primary = Aqua,
    background = Color.Black,
    surface = Color.Black,
    onPrimary = Color.Black,
    onBackground = Color.White,
    onSurface = Color.White,
    outline = Aqua
)


class MainActivity : ComponentActivity(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null

    private var accValues = FloatArray(3)
    private var gyroValues = FloatArray(3)

    private var currentLabel by mutableStateOf<String?>(null)
    private var isRecording by mutableStateOf(false)
    private var startTime by mutableStateOf(0L)

    private val sensorDataList = mutableListOf<SensorData>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        setContent {
            MaterialTheme(colorScheme = DarkColors) {
                Surface(modifier = Modifier.fillMaxSize(), color = Color.Black) {
                    if (currentLabel == null) {
                        HomeScreen { startSession(it) }
                    } else {
                        ActivityScreen(
                            activityName = currentLabel!!,
                            isRecording = isRecording,
                            startTime = startTime,
                            onBack = { stopAndReset() },
                            onStop = { isRecording = false },
                            onSave = {
                                saveDataToCSV()
                                stopAndReset()
                            }
                        )
                    }
                }
            }
        }
    }

    private fun startSession(label: String) {
        currentLabel = label
        sensorDataList.clear()
        startTime = SystemClock.elapsedRealtime()
        isRecording = true
    }

    private fun stopAndReset() {
        isRecording = false
        currentLabel = null
    }

    override fun onResume() {
        super.onResume()
        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        gyroscope?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (!isRecording || currentLabel == null) return

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> accValues = event.values.clone()
            Sensor.TYPE_GYROSCOPE -> gyroValues = event.values.clone()
        }

        sensorDataList.add(
            SensorData(
                System.currentTimeMillis(),
                accValues[0], accValues[1], accValues[2],
                gyroValues[0], gyroValues[1], gyroValues[2],
                currentLabel!!
            )
        )
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}


    private fun saveDataToCSV() {
        if (sensorDataList.isEmpty()) return

        val file = File(getExternalFilesDir(null), "har_data.csv")
        val writer = file.bufferedWriter()

        writer.write("timestamp,accX,accY,accZ,gyroX,gyroY,gyroZ,label\n")
        sensorDataList.forEach {
            writer.write(
                "${it.timestamp},${it.accX},${it.accY},${it.accZ}," +
                        "${it.gyroX},${it.gyroY},${it.gyroZ},${it.label}\n"
            )
        }
        writer.close()

        Toast.makeText(this, "Session saved", Toast.LENGTH_SHORT).show()
    }
}


@Composable
fun HomeScreen(onSelect: (String) -> Unit) {
    var customActivity by remember { mutableStateOf("") }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
            .padding(top = 48.dp, start = 20.dp, end = 20.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {

        Text(
            text = "Activity Recorder",
            fontSize = 34.sp,
            fontWeight = FontWeight.ExtraBold,
            color = Aqua
        )

        Text(
            text = "Select an activity to start recording",
            fontSize = 16.sp,
            color = Color.White.copy(alpha = 0.8f)
        )

        listOf("🚶 Walking", "🪑 Sitting", "🧍 Standing", "🏃 Running").forEach {
            ActivityCard(it) { onSelect(it.substringAfter(" ")) }
        }

        OutlinedTextField(
            value = customActivity,
            onValueChange = { customActivity = it },
            label = { Text("Custom activity", color = Color.White) },
            textStyle = LocalTextStyle.current.copy(color = Color.White),
            modifier = Modifier.fillMaxWidth(),
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = Aqua,
                unfocusedBorderColor = Color.White,
                cursorColor = Aqua
            )
        )

        Button(
            onClick = {
                if (customActivity.isNotBlank()) {
                    onSelect(customActivity.trim())
                    customActivity = ""
                }
            },
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(containerColor = Aqua)
        ) {
            Text("Start", color = Color.Black)
        }
    }
}

@Composable
fun ActivityCard(label: String, onClick: () -> Unit) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick() },
        colors = CardDefaults.cardColors(containerColor = Color.Black),
        border = BorderStroke(1.dp, Aqua),
        shape = RoundedCornerShape(18.dp)
    ) {
        Text(
            text = label,
            color = Color.White,
            fontSize = 20.sp,
            modifier = Modifier.padding(20.dp)
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ActivityScreen(
    activityName: String,
    isRecording: Boolean,
    startTime: Long,
    onBack: () -> Unit,
    onStop: () -> Unit,
    onSave: () -> Unit
) {
    var elapsed by remember { mutableStateOf(0L) }

    LaunchedEffect(isRecording) {
        while (isRecording) {
            elapsed = SystemClock.elapsedRealtime() - startTime
            delay(1000)
        }
    }

    val minutes = TimeUnit.MILLISECONDS.toMinutes(elapsed)
    val seconds = TimeUnit.MILLISECONDS.toSeconds(elapsed) % 60

    Column(modifier = Modifier.fillMaxSize().background(Color.Black)) {

        TopAppBar(
            title = { Text(activityName, color = Aqua) },
            navigationIcon = {
                IconButton(onClick = onBack) {
                    Icon(Icons.Default.ArrowBack, contentDescription = "Back", tint = Color.White)
                }
            },
            colors = TopAppBarDefaults.topAppBarColors(containerColor = Color.Black)
        )

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(24.dp)
        ) {

            Text(
                String.format("%02d:%02d", minutes, seconds),
                fontSize = 52.sp,
                fontWeight = FontWeight.ExtraBold,
                color = Aqua
            )

            Text(if (isRecording) "Recording" else "Stopped", color = Color.White)

            OutlinedButton(
                onClick = onStop,
                enabled = isRecording,
                border = BorderStroke(1.dp, Aqua),
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Stop Recording", color = Color.White)
            }

            Button(
                onClick = onSave,
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(containerColor = Aqua)
            ) {
                Text("Save Session", color = Color.Black)
            }
        }
    }
}
