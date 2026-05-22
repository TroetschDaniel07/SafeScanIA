# lectorOBDII.py
import obd
import time

# Conexión al emulador
connection = obd.OBD(
    portstr="socket://localhost:35000", 
    baudrate=38400, 
    protocol="6",      # Protocolo ISO 15765-4 CAN
    timeout=30
)

if not connection.is_connected():
    print("Error: No se pudo conectar al emulador")
    print("¿Ejecutaste 'python -m elm -n 35000' en otra terminal?")
    exit(1)

print("✅ Conectado al emulador ELM327\n")

# Esperar un momento para estabilizar
time.sleep(1)

# 1️⃣ Solicitar códigos de fallo (DTCs)
print("🔍 Solicitando códigos DTC...")
dtcs = connection.query(obd.commands.GET_DTC)

if dtcs and dtcs.value:
    print(f"📋 Códigos DTC detectados: {dtcs.value}")
else:
    print("📋 No se detectaron códigos DTC")

# 2️⃣ Probar otros comandos básicos
print("\n📊 Leyendo datos del vehículo:")

# RPM
rpm = connection.query(obd.commands.RPM)
if rpm and rpm.value is not None:
    print(f"   RPM: {rpm.value}")

# Velocidad
speed = connection.query(obd.commands.SPEED)
if speed and speed.value is not None:
    print(f"   Velocidad: {speed.value} km/h")

# Temperatura del refrigerante (corregido)
coolant = connection.query(obd.commands.COOLANT_TEMP)
if coolant and coolant.value is not None:
    # Usar .magnitude para obtener el valor numérico sin las unidades problemáticas
    print(f"   Temp. refrigerante: {coolant.value.magnitude:.1f} °C")
else:
    print("   Temp. refrigerante: No disponible")

# 3️⃣ Comandos adicionales para probar
print("\n🎮 Probando comandos adicionales:")

# Carga del motor
load = connection.query(obd.commands.ENGINE_LOAD)
if load and load.value is not None:
    print(f"   Carga del motor: {load.value}%")

# Temperatura del aire de admisión
intake_temp = connection.query(obd.commands.INTAKE_TEMP)
if intake_temp and intake_temp.value is not None:
    print(f"   Temp. aire admisión: {intake_temp.value.magnitude:.1f} °C")

# Presión absoluta del colector
map_pressure = connection.query(obd.commands.INTAKE_PRESSURE)
if map_pressure and map_pressure.value is not None:
    print(f"   Presión del colector: {map_pressure.value} kPa")

connection.close()
print("\n🔌 Conexión cerrada")