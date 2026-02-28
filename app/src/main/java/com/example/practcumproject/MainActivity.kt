package com.example.practcumproject

import android.content.Context
import android.hardware.*
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
import java.io.BufferedWriter
import java.io.FileWriter
import java.util.concurrent.TimeUnit

private val Aqua = Color(0xFF00E5FF)

class MainActivity : ComponentActivity(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null

    private var accValues = FloatArray(3)
    private var gyroValues = FloatArray(3)

    private var currentLabel by mutableStateOf<String?>(null)
    private var isRecording by mutableStateOf(false)
    private var startTime by mutableStateOf(0L)

    private val sensorDataList = mutableListOf<String>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        setContent {
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

        val row = "${System.currentTimeMillis()}," +
                "${accValues[0]},${accValues[1]},${accValues[2]}," +
                "${gyroValues[0]},${gyroValues[1]},${gyroValues[2]}," +
                currentLabel

        sensorDataList.add(row)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun saveDataToCSV() {

        if (sensorDataList.isEmpty()) {
            Toast.makeText(this, "No data to save", Toast.LENGTH_SHORT).show()
            return
        }

        val file = File(filesDir, "har_dataset.csv")
        val fileExists = file.exists()

        val writer: BufferedWriter = BufferedWriter(FileWriter(file, true))

        if (!fileExists) {
            writer.write("timestamp,accX,accY,accZ,gyroX,gyroY,gyroZ,label\n")
        }

        sensorDataList.forEach {
            writer.write(it + "\n")
        }

        writer.flush()
        writer.close()

        sensorDataList.clear()

        Toast.makeText(this, "Session appended successfully", Toast.LENGTH_SHORT).show()
    }
}

@Composable
fun HomeScreen(onSelect: (String) -> Unit) {

    var customActivity by remember { mutableStateOf("") }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {

        Text(
            text = "Activity Recorder",
            fontSize = 32.sp,
            fontWeight = FontWeight.ExtraBold,
            color = Aqua
        )

        Spacer(modifier = Modifier.height(8.dp))

        listOf("Walking", "Sitting", "Standing", "Running").forEach { activity ->

            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { onSelect(activity) },
                colors = CardDefaults.cardColors(
                    containerColor = Color(0xFF1E1E1E)
                ),
                border = BorderStroke(1.dp, Aqua),
                shape = RoundedCornerShape(16.dp)
            ) {
                Text(
                    text = activity,
                    modifier = Modifier.padding(20.dp),
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Medium,
                    color = Color.White
                )
            }
        }

        Spacer(modifier = Modifier.height(12.dp))


        OutlinedTextField(
            value = customActivity,
            onValueChange = { customActivity = it },
            label = { Text("Custom Activity", color = Color.White) },
            modifier = Modifier.fillMaxWidth(),
            textStyle = LocalTextStyle.current.copy(color = Color.White),
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = Aqua,
                unfocusedBorderColor = Color.Gray,
                cursorColor = Aqua
            ),
            shape = RoundedCornerShape(14.dp)
        )

        Button(
            onClick = {
                if (customActivity.isNotBlank()) {
                    onSelect(customActivity.trim())
                    customActivity = ""
                }
            },
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(
                containerColor = Aqua,
                contentColor = Color.Black
            ),
            shape = RoundedCornerShape(14.dp)
        ) {
            Text(
                text = "Start Custom Activity",
                fontWeight = FontWeight.Bold
            )
        }
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
                    Icon(Icons.Default.ArrowBack, contentDescription = null, tint = Color.White)
                }
            }
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
                fontSize = 48.sp,
                fontWeight = FontWeight.Bold,
                color = Aqua
            )

            Text(
                if (isRecording) "Recording..." else "Stopped",
                color = Color.White
            )

            Button(
                onClick = onStop,
                enabled = isRecording,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Stop")
            }

            Button(
                onClick = onSave,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Save Session")
            }
        }
    }
}