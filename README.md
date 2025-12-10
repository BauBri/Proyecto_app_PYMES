# Estimador de datos con Lógica Difusa

Aplicación desarrollada en Streamlit para apoyar la toma de decisiones en pequeñas y medianas empresas mediante el uso de lógica difusa.  
El sistema permite leer datos, construir números difusos, realizar cálculos básicos y generar visualizaciones de funciones de membresía de forma interactiva.

## Características principales

- Implementación de números difusos triangulares y trapezoidales.  
- Cálculos fundamentales mediante aritmética difusa.  
- Lectura y procesamiento de datos para generar estimaciones bajo incertidumbre.  
- Interfaz web ligera desarrollada con Streamlit.  
- Arquitectura modular distribuida en:
  - `base_difusa.py` – Operaciones y definiciones de números difusos.  
  - `lectura_datos.py` – Manejo y preprocesamiento de datos.  
  - `streamlit_app.py` – Interfaz principal y visualización.

## Requisitos

Instalar las dependencias desde el archivo `requirements.txt`:

   ```
   $ pip install -r requirements.txt
   ```

## Ejecución local

Ejecutar la aplicación con:

   ```
   $ streamlit run streamlit_app.py
   ```

## Estructura del repositorio

- `.streamlit/` – Configuración de la interfaz.  
- `base_difusa.py` – Núcleo matemático del sistema.  
- `lectura_datos.py` – Carga y transformación de datos.  
- `streamlit_app.py` – Aplicación principal.  
- `requirements.txt` – Dependencias del entorno.  
- `unam_logo.png` – Elemento visual institucional.  

## Licencia

Este proyecto se distribuye bajo los términos de la licencia incluida en `LICENSE`.
