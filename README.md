# RV-MAP (Rescue Vision Mapping)

**Sistema de Mapeo por Visión de Rescate y Análisis de Transitabilidad**

RV-MAP es una arquitectura algorítmica de visión computacional diseñada para el soporte logístico en entornos colapsados. El núcleo de *software* ingesta telemetría óptica capturada por vehículos aéreos no tripulados (UAV) y, mediante inferencia tensorial basada en YOLOv8, segmenta las estructuras destruidas, vehículos y escombros. Posteriormente, aplica operaciones de dilatación morfológica para generar mapas de costes bidimensionales que delimitan las áreas de tránsito seguro frente a zonas de exclusión.

## Características Técnicas

* **Inferencia Neuronal:** Extracción de características topológicas en tiempo real utilizando la arquitectura YOLOv8.
* **Geometría de Penalización:** Implementación de gradientes de riesgo cónicos euclidianos para calcular márgenes de seguridad físicos en torno a obstáculos dinámicos.
* **Auditoría Visual:** Proyección de contornos geométricos sobre la matriz de transitabilidad para validación táctica humana.
* **Despliegue Interactivo:** Interfaz de línea de comandos (CLI) parametrizable y optimizada para operar en plataformas de *Edge Computing*.

## Despliegue del Entorno Computacional

Para instanciar el entorno de ejecución, se recomienda aislar las dependencias mediante un entorno virtual (`venv` o `conda`). Ejecute los siguientes comandos en la terminal de su sistema:

```bash
# 1. Clonar el repositorio
git clone [https://github.com/TU_USUARIO/RV-MAP.git](https://github.com/TU_USUARIO/RV-MAP.git)
cd RV-MAP

# 2. Instalar dependencias matriciales y de aprendizaje profundo
pip install -r requirements.txt
