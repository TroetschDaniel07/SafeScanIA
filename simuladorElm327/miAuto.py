# mi_auto.py - Múltiples DTCs para simular Check Engine
ObdMessage = {
    # =========================================================
    # COMANDO 03: Leer códigos DTC almacenados
    # =========================================================
    '03': {  
        # Simula tres códigos de fallo activos:
        # P0102 + P0300 + P0420
        'Response': '43 01 02 33 03 00 00 04 20 00',
        'Descr': 'DTCs: P0102 (MAF Circuit Low), P0300 (Random Misfire), P0420 (Catalyst)'
    },
    
    # =========================================================
    # COMANDO 07: Leer códigos DTC pendientes (no confirmados)
    # =========================================================
    '07': {
        'Response': '47 01 35 00 00 00 00 00',
        'Descr': 'DTC Pendiente: P0135 (O2 Sensor Heater Circuit)'
    },
    
    # =========================================================
    # DATOS DE SENSORES (para que el lector tenga más datos)
    # =========================================================
    '010C': {  # RPM
        'Response': '41 0C 1A F8',  # 6900 RPM
        'Descr': 'Engine RPM: 6900'
    },
    '010D': {  # Velocidad
        'Response': '41 0D 50',  # 80 km/h
        'Descr': 'Vehicle Speed: 80 km/h'
    },
    '0105': {  # Temperatura refrigerante
        'Response': '41 05 53',  # 83°C
        'Descr': 'Coolant Temp: 83°C'
    },
}