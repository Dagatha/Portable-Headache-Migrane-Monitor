import uasyncio as asyncio
from machine import Pin, SPI, I2C
import time
import math
import network
import urequests

# Wi-Fi credentials
def connect_wifi():
    ssid = 'OleBallAndChang'
    password = 'wHgDEr3(n,HgkEn]%9{4'
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(ssid, password)
    print("Connecting to Wi-Fi...")
    while not wlan.isconnected():
        time.sleep(1)
    print("Connected:", wlan.ifconfig())

# SPI setup for MCP3008
spi = SPI(0, baudrate=1000000, polarity=0, phase=0,
          sck=Pin(18), mosi=Pin(19), miso=Pin(16))
cs = Pin(17, Pin.OUT)

# I2C setup for SEN0546 (CHT8305)
i2c = I2C(0, scl=Pin(21), sda=Pin(20), freq=100000)

# Constants
V_REF = 3.3
R_FIXED = 10000
V_REF_DB = 0.00631
headache_active = False

# Discord webhook URL
DISCORD_WEBHOOK_URL = 'https://discord.com/api/webhooks/1392979421979349083/ItHV6qipj-ub8Fp8ims6eq1NGNQmpNkMj2SGPyi20xHcoMmM2RxqmlVIwBFyL5sUltnB'

def send_discord_alert(message: str):
    payload = {"content": message}
    try:
        response = urequests.post(DISCORD_WEBHOOK_URL, json=payload)
        print("Discord alert sent:", response.status_code)
        response.close()
    except Exception as e:
        print("Discord alert failed:", e)

def read_adc(channel):
    if not 0 <= channel <= 7:
        return -1
    cs.value(0)
    buf = bytearray(3)
    buf[0] = 1
    buf[1] = (8 + channel) << 4
    buf[2] = 0
    spi.write_readinto(buf, buf)
    cs.value(1)
    result = ((buf[1] & 3) << 8) | buf[2]
    return result

def moving_average(data, window_size):
    if len(data) < window_size:
        return sum(data) / len(data)
    return sum(data[-window_size:]) / window_size

def detect_peak(data, threshold=0.5):
    if len(data) < 3:
        return False
    return data[-2] > data[-3] and data[-2] > data[-1] and data[-2] > threshold

def read_temperature_humidity():
    address = 0x40
    try:
        i2c.writeto(address, b'\x00')
        time.sleep_ms(20)
        buf = i2c.readfrom(address, 4)
        if len(buf) != 4:
            raise ValueError("Incomplete data received")
        temp_raw = (buf[0] << 8) | buf[1]
        hum_raw = (buf[2] << 8) | buf[3]
        temperature = ((temp_raw * 165.0) / 65535.0) - 40.0
        humidity = (hum_raw / 65535.0) * 100.0
        return temperature, humidity
    except Exception as e:
        print("Temp/Humidity read error:", e)
        return None, None

async def monitor_pulse(shared):
    samples = []
    smoothed = []
    peak_times = []
    window_size = 5
    min_interval_ms = 300
    sampling_interval = 0.1
    contact_threshold = 0.5
    while True:
        if headache_active:
            raw = read_adc(0)
            voltage = (raw / 1023) * V_REF
            samples.append(voltage)
            avg = moving_average(samples, window_size)
            smoothed.append(avg)
            shared["pulse"] = avg
            if avg > contact_threshold:
                now = time.ticks_ms()
                if detect_peak(smoothed, threshold=contact_threshold):
                    if len(peak_times) == 0 or time.ticks_diff(now, peak_times[-1]) > min_interval_ms:
                        peak_times.append(now)
                if len(peak_times) > 11:
                    peak_times = peak_times[-11:]
                if len(peak_times) >= 2:
                    intervals = [time.ticks_diff(peak_times[i], peak_times[i - 1]) for i in range(1, len(peak_times))]
                    avg_interval = sum(intervals) / len(intervals)
                    bpm = 60000 / avg_interval
                    print("BPM: {:.1f}".format(bpm))
            else:
                print("Pulse sensor not in contact.")
        await asyncio.sleep(sampling_interval)

async def monitor_gsr(shared):
    gsr_conductance_history = []
    window_size = 5
    sampling_interval = 0.1
    alpha = 0.5  
    smoothed_conductance = 0  

    while True:
        if headache_active:
            raw = read_adc(1)
            voltage = (raw / 1023) * V_REF
            try:
                resistance = R_FIXED * (V_REF / voltage - 1)
                conductance = 1 / resistance * 1e6
            except ZeroDivisionError:
                conductance = 0
            
            # Update the conductance history
            gsr_conductance_history.append(conductance)
            if len(gsr_conductance_history) > window_size:
                gsr_conductance_history.pop(0)
            
            # Calculate the exponential moving average
            if smoothed_conductance == 0:
                smoothed_conductance = conductance  # Initialize on the first reading
            else:
                smoothed_conductance = alpha * conductance + (1 - alpha) * smoothed_conductance
            
            # Calculate slope based on smoothed values
            if len(gsr_conductance_history) >= 2:
                slope = (smoothed_conductance - gsr_conductance_history[-2]) / sampling_interval
            else:
                slope = 0
            
            shared["gsr"] = slope
            print("GSR Conductance Slope: {:.2f} µS/s".format(slope))
        
        await asyncio.sleep(sampling_interval)

async def monitor_microphone(shared):
    sampling_interval = 0.1
    burst_samples = 100
    burst_delay = 0.001
    while True:
        if headache_active:
            max_val = 0
            min_val = 1023
            for _ in range(burst_samples):
                raw = read_adc(2)
                max_val = max(max_val, raw)
                min_val = min(min_val, raw)
                await asyncio.sleep(burst_delay)
            peak_to_peak = max_val - min_val
            loudness_v = (peak_to_peak / 1023) * V_REF
            try:
                loudness_db = 20 * math.log10(loudness_v / V_REF_DB)
            except ValueError:
                loudness_db = 0
            shared["sound_db"] = loudness_db
            print("Loudness: {:.1f} dB".format(loudness_db))
        await asyncio.sleep(sampling_interval)

async def monitor_temperature(shared):
    while True:
        if headache_active:
            temp, hum = read_temperature_humidity()
            if temp is not None:
                shared["temp"] = temp
                shared["hum"] = hum
                print("Temperature: {:.2f} °C | Humidity: {:.1f} %RH".format(temp, hum))
        await asyncio.sleep(1)

async def monitor_light(shared):
    light_samples = []
    window_size = 5
    sampling_interval = 0.1
    while True:
        if headache_active:
            raw = read_adc(3)
            voltage = (raw / 1023) * V_REF
            light_samples.append(voltage)
            if len(light_samples) > window_size:
                light_samples.pop(0)
            avg_voltage = moving_average(light_samples, window_size)
            lux = 500 * (avg_voltage / V_REF)
            shared["light"] = lux
            print("Estimated Light Intensity: {:.1f} lux".format(lux))
        await asyncio.sleep(sampling_interval)

async def predict_headache(shared):
    while True:
        if headache_active:
            try:
                payload = {
                    "timestamp": time.time(),
                    "pulse": round(shared.get("pulse", 0.0), 2),
                    "gsr_slope": round(shared.get("gsr", 0.0), 2),
                    "temperature": round(shared.get("temp", 0.0), 2),
                    "humidity": round(shared.get("hum", 0.0), 2),
                    "sound_db": round(shared.get("sound_db", 0.0), 1),
                    "light_lux": round(shared.get("light", 0.0), 1),
                    "label": 0
                }
                response = urequests.post("http://192.168.68.66:8000/predict", json=payload)
                result = response.json()
                response.close()
                prediction = result.get("prediction", 0)
                probability = result.get("probability", 0.0)
                print("Prediction:", prediction, "Probability:", probability)
                if prediction == 1 and probability > 0.8:
                    send_discord_alert("🚨 Headache Imminent Find Safety!")
            except Exception as e:
                print("Prediction error:", e)
        await asyncio.sleep(5)

async def monitor_button():
    global headache_active
    button = Pin(15, Pin.IN, Pin.PULL_UP)
    debounce_time = 200
    last_press = 0
    while True:
        if button.value() == 0:
            now = time.ticks_ms()
            if time.ticks_diff(now, last_press) > debounce_time:
                headache_active = not headache_active
                status = "ON" if headache_active else "OFF"
                print("System toggled:", status)
                last_press = now
            try:
                if headache_active:
                    urequests.post("http://192.168.68.66:8000/system_on")
                else:
                    urequests.post("http://192.168.68.66:8000/system_off")
            except Exception as e:
                print("Client sync failed:", e)
        await asyncio.sleep(0.05)

async def main():
    shared = {}
    await asyncio.gather(
        monitor_pulse(shared),
        monitor_gsr(shared),
        monitor_microphone(shared),
        monitor_temperature(shared),
        monitor_light(shared),
        predict_headache(shared),
        monitor_button()
    )

try:
    connect_wifi()
    asyncio.run(main())
except KeyboardInterrupt:
    print("Program stopped manually.")

